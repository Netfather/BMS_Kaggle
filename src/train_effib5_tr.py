# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         train_effib5_tr
# Description:  此文件用于实现 将数据集进行训练
# # 2021年5月18日 V1. 加入学习率计划 加入数据增强 加入从头训练模式和非从头训练模式
# #                  由于efb5模型大概比 res101 快了1.5倍 因此考虑每5000个Iteration 查看一次Val 保存一次结果
# 2021年5月20日 V2. 修复do_val 函数的错误 原函数不能正确反应 lb的真实分数
#                  扩大 输入的分辨率 从 224 扩大到 384  中间不一样的地方 加入 torch的dwonasmple层强行转为一样 如此使用原来的函数进行FineTune
#                  使用fold5 进行训练
#              V3. 确认do_valid修正无误 目前是可以反应真实的lb分数的
#                  放弃使用 anti-focal loss  改回原来的  bce loss 目前还没有达到 finetune的情况
#                  修改 FixNumberSampler 使得 现在不再是从0到固定地点 而是在指定区间内进行测试
#                  修改 训练流程  每次先进行一次 valid 然后再开始进行下一步的训练
# 2021年5月22日 V5. 修正do_valid函数问题 cv和lb之间的巨大差距源于二者的评估方式完全不一致。 原来的do_valid函数采用的是force teachiing的方式
#                   这种方式为了提升速度，每次喂入的是正确的结果 因此cv极小   但是测试中用的是unforce teaching的方式 因此这里统一修改为 unforce teaching
#                   loss暂时不变
# 2021年5月24日 V6.  再一步调低学习率 这次范围为 5e-7到1e-6 从1800000pth开始
#              V7.  使用SGDM 进行最终的收敛   根据Vit的相关指引   512 batch 他们使用 0.001到 0.003进行收敛
#                   24batch  使用1e-4 到 5e-5收敛比较合理
# 2021年5月25日 V8.  保持Radam优化器不变，再一次调小学习率  目前为 1e-6 到 5e-6
#                   调整为5e-7到1e-6查看结果
# 2021年5月27日 V9.  已经无法下降了 终止EB5模型的训练。
# 2021年5月29日 V10. 扩大分辨率 重启训练  扩大为上限384  1920000_model 包括本身 之前都为320  记得修改Util中的相关设定
# 2021年5月29日 V11. 修正avepool的问题
# 2021年5月30日 V12. 修正学习率的问题 128batch 下 学习率3e-5为最终选择   我的batchsize为20 根据等效缩放原则 为  4.68 10 -6  差不多为5e-6 那目前学习率没问题  就是训练时常不够
# 2021年5月31日 V13. 2192000_model.pth调大学习率  尝试冲出局部最优解！！！   更新：并非学习率的问题，可能是旋转的问题，关闭旋转试一试
# 2021年6月1日  V14. 修正使用 focal-loss来进行最后的FineTune 查看结果
# 2021年6月1日  V15. 为了给tit模型腾出位置  终止此模型的训练。 并查看单模型的结果  tit时间已经来不及 放弃！
# 2021年6月2日  V16. 2296000_model.pth 之后 重新开启混精度 调低为320 扩大batchsize 调大学习率  重新开启RandomRotate 最后尝试
# Author:       Administrator
# Date:         2021/5/18
# -------------------------------------------------------------------------------
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import pandas as pd
import torch
from utils import model_ef
import Levenshtein
from torch.nn import functional as F
from timeit import default_timer as timer
from utils import util
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from utils.network_model.lookahead import Lookahead
from utils.network_model.radam import RAdam
import torch.optim.lr_scheduler as torchopt_lrs
from random import randrange
from utils.log_writers.log import get_logger
import albumentations as A
import gc
from utils.learning_schdule_box.pytorch_cosin_warmup import CosineAnnealingWarmupRestarts


import random

import json

# 一些必要的设置参数定义
device = torch.device('cuda:0')
Open_Parral_Training = False  #开启双显卡训练
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
manual_seed = randrange(10000)

# 一些必要的超参数定义
# 读入哪个fold的 数据 并将结果输出到 model 对应的fold文件夹下
fold = 4
out_dir = '../model/efb_trans/fold{}'.format(fold)

Train_From_Beginning = False
if Train_From_Beginning:
    initial_checkpoint = '../model/1360000_model.pth'
else:
    initial_checkpoint = '../model/efb_trans/fold4/checkpoint/2296000_model.pth'

start_lr = 1e-5
batch_size = 28

MAX_VALIDATION_RANGE = [20000,30000]

# 最大迭代 batchsize 次数
num_iteration = 80000 * 1000
# 每 iter_log 打印
iter_log = 1000
# 每 iter_valid 次验证
iter_valid = 8000
# 每 iter_save 次保存
iter_save = 8000 # 1*1000

# 词典中特定标识符
STOI = {
    '<sos>': 190,
    '<eos>': 191,
    '<pad>': 192,
}
# 词典预定义的图片大小  词典大小  以及一个输出最长为多少
image_size = 320  # 224 -》 320 ——》 384 -》320
vocab_size = 193
max_length = 300

#是否开启 混精度 amp
is_mixed_precision = False #False  #

# 初始化 日志头
logdir = "../model/efb_trans/fold{}/log".format(fold)
logger = get_logger(logdir,OutputOnConsole= True,log_initial= "efb_tr",logfilename="train_efb_tr_avepool")

# 图像增强手段 2021年5月18日 新增
def train_transforms():
    return A.Compose([
        A.Resize(image_size, image_size),
        A.RandomRotate90(p=0.8),#修正 扩大旋转90度的范围  # 更新 暂时关闭  # 更新 开启
        A.OneOf([
            # 高斯噪点
            A.IAAAdditiveGaussianNoise(p = 1),
            A.GaussNoise(p = 1),
            # 模糊相关操作
            A.MotionBlur(p=1),
            A.MedianBlur(blur_limit=3, p=1),
            A.Blur(blur_limit=3, p=1),
        ], p=0.8),
    ], p=1.)

def val_transforms():
    return A.Compose([
        A.Resize(image_size,image_size),
    ], p=1.)

def test_transforms():
    return A.Compose([
        A.Resize(image_size,image_size),
    ], p=1.)
###################################################################################################
# 设定训练的fold
logger.info('** fold_number setting **\n')
logger.info('fold_number : \n%s\n' % (fold))
logger.info('\n')

###################################################################################################
# 记录设定的Valid 范围
logger.info('** Valid_data num range **\n')
logger.info('MAX_VALIDATION_RANGE : \n%s\n' % (MAX_VALIDATION_RANGE))
logger.info('\n')

###################################################################################################
# 设定随机种子，方便复现代码
def set_seeds(seed=manual_seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seeds(manual_seed)
logger.info('** set_seeds setting **\n')
logger.info('manual_seed : \n%s\n' % (manual_seed))
logger.info('\n')

###################################################################################################
import torch.cuda.amp as amp
class AmpNet(model_ef.Net):
    @torch.cuda.amp.autocast()
    def forward(self, *args):
        return super(AmpNet, self).forward(*args)


###################################################################################################
# 测试验证集指标  输入 网络  词典  和验证加载器  返回一个列表  第一个为验证集的loss 第二个为验证集的编辑距离 也就是lb_scor的得分
def do_valid(net, tokenizer, valid_loader):

    valid_probability = []
    valid_truth = []
    valid_length = []
    valid_txt_modify = []
    valid_num = 0

    net.eval()
    start_timer = timer()
    for t, batch in enumerate(valid_loader):
        batch_size = len(batch['index'])
        image  = batch['image' ].to(device)
        token  = batch['token' ].to(device)
        length = batch['length']

        with torch.no_grad():
            # logit = net(image, token, length)
            # probability = F.softmax(logit,-1)
            k = net.forward_argmax_decode(image)

        valid_num += batch_size
        # valid_probability.append(probability.data.cpu().numpy())
        valid_truth.append(token.data.cpu().numpy())
        valid_txt_modify.append(k.data.cpu().numpy())
        valid_length.extend(length)
        print('\r %8d / %d  %s'%(valid_num, len(valid_loader.sampler),time_to_str(timer() - start_timer,'sec')),end='',flush=True)

    assert(valid_num == len(valid_loader.sampler)) #len(valid_loader.dataset))

    #----------------------
    # probability = np.concatenate(valid_probability)
    #predict = probability.argmax(-1)
    truth   = np.concatenate(valid_truth)
    length  = valid_length
    predict_modify = np.concatenate(valid_txt_modify)

    #----
    # p = probability[:,:-1].reshape(-1,vocab_size)
    # t = truth[:,1:].reshape(-1)
    #
    # non_pad = np.where(t!=STOI['<pad>'])[0] #& (t!=STOI['<sos>'])
    # p = p[non_pad]
    # t = t[non_pad]
    # loss = util.np_loss_cross_entropy(p, t)
    loss = 0.0
    #----
    # V2 修正 原 valid 有问题
    lb_score = 0
    if 1:
        score = []
        for i, (p, t) in enumerate(zip(predict_modify, truth)):
            t = truth[i][1:length[i] - 1]
            p = predict_modify[i][0:length[i] - 2]
            t = tokenizer.one_sequence_to_text(t)
            p = tokenizer.one_sequence_to_text(p)
            s = Levenshtein.distance(p, t)
            score.append(s)
        lb_score = np.mean(score)
    del valid_probability,valid_truth,valid_length,truth,p,t,s,score,predict_modify,valid_txt_modify
    gc.collect()
    return [loss, lb_score]

###################################################################################################
# 打印信息 设定 返回一个 当前信息的字符串 输出
def message(mode='print'):
    if mode == ('print'):
        asterisk = ' '
        loss = batch_loss
    if mode == ('log'):
        asterisk = '*' if iteration % iter_save == 0 else ' '
        loss = train_loss

    text = \
        '%0.7f  %5.4f%s %4.2f  | ' % (rate, iteration / 10000, asterisk, epoch,) + \
        '%4.3f  %5.2f  | ' % (*valid_loss,) + \
        '%4.3f  %4.3f  | ' % (*loss,) + \
        '%s' % (time_to_str(timer() - start_timer, 'min'))

    return text


# 输出当前所用时间
def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)

    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)

    else:
        raise NotImplementedError



## setup  ----------------------------------------
# 在output目标路径下 新建一个文件夹名字为 checkpoint
for f in ['checkpoint']:
    os.makedirs(out_dir + '/' + f, exist_ok=True)

## dataset ------------------------------------
# 读入 fold对应的pickle
df_train = pd.read_pickle('../data/df_train{}.pkl'.format(fold))
df_valid = pd.read_pickle('../data/df_valid{}.pkl'.format(fold))

# 加载数据
tokenizer = util.load_tokenizer()
train_dataset = util.BmsDataset(df_train,tokenizer,augment= train_transforms())
logger.info("Load train_dataset OK! The lengths is {}".format(len(train_dataset)))
valid_dataset = util.BmsDataset(df_valid,tokenizer,augment=val_transforms())
logger.info("Load valid_dataset OK! The lengths is {}".format(len(valid_dataset)))
assert(MAX_VALIDATION_RANGE[1] <= len(valid_dataset))  #判定Range是否超越了 规定的范围 如果超越则返回报错


logger.info('** auguement setting **\n')
logger.info('train_transforms : \n%s\n' % (train_transforms()))
logger.info('val_transforms : \n%s\n' % (val_transforms()))
logger.info('\n')

train_loader = DataLoader(
    train_dataset,
    sampler = RandomSampler(train_dataset),
    batch_size=batch_size,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
    collate_fn=util.collate_fn,
)
valid_loader = DataLoader(
    valid_dataset,
    # sampler=SequentialSampler(valid_dataset),
    sampler=util.FixNumSampler(valid_dataset, length=MAX_VALIDATION_RANGE),
    batch_size=128,
    drop_last=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=util.collate_fn,
)

logger.info('train_dataset : \n%s\n' % (train_dataset))
logger.info('valid_dataset : \n%s\n' % (valid_dataset))

## net ----------------------------------------
logger.info('** net setting **\n')
if is_mixed_precision:
    scaler = amp.GradScaler()
    net = AmpNet().to(device)
else:
    net = model_ef.Net().to(device)

if initial_checkpoint is not None:
    f = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
    start_iteration = f['iteration']
    start_epoch     = f['epoch']
    state_dict = f['state_dict']
    # 查询一i西安 state_dict中所有的encoder 然后删除   只保留权重中的 Transformer部分
    # 如果是从头开始训练 我们就要删除 给予权重中的encoder部分 否则就直接读入
    if Train_From_Beginning:
        for element in list(state_dict.keys()):
            if (element[0:7] == "encoder"):
                del state_dict[element]

        net.load_state_dict(state_dict, strict=False)  # True
    else:
        net.load_state_dict(state_dict, strict=True)  # True
else:
    start_iteration = 0
    start_epoch = 0


logger.info('\tinitial_checkpoint = %s\n' % initial_checkpoint)
logger.info('\n')


logger.info('\tTrain_From_Beginning = %s\n' % Train_From_Beginning)
logger.info('\n')


logger.info('\tOpen_Parral_traing = %s\n' % Open_Parral_Training)
logger.info('\n')
if Open_Parral_Training:
    net = torch.nn.DataParallel(net)

# 定义优化器
innew_potimizer = RAdam(filter(lambda p: p.requires_grad, net.parameters()), lr=start_lr)
# 2021年5月24日 V7 新增 使用SGDM进行最终的收敛
# innew_potimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
#                                lr= start_lr,
#                                momentum= 0.90,
#                                nesterov=False,
#                                )
# scheduler=torchopt_lrs.OneCycleLR(optimizer=innew_potimizer, pct_start=0.2, div_factor=1e2, max_lr=5e-4,steps_per_epoch=len(train_loader), epochs=10)
scheduler = CosineAnnealingWarmupRestarts(innew_potimizer,
                                          first_cycle_steps=3000,
                                          cycle_mult=1.0,
                                          max_lr=5e-5,
                                          min_lr=1e-5,
                                          warmup_steps=50,
                                          gamma=1.0)
# 别忘记打开 scheduler()!!!!
optimizer = Lookahead(innew_potimizer, alpha=0.5, k=5)

logger.info('optimizer\n  %s\n' % (optimizer))
logger.info('\n')

logger.info('scheduler\n  %s\n' % (scheduler))
logger.info('\n')

## start training here! ##############################################
logger.info('** start training here! **\n')
logger.info('   is_mixed_precision = %s \n' % str(is_mixed_precision))
logger.info('   batch_size = %d\n' % (batch_size))
logger.info('                      |----- VALID ---|---- TRAIN/BATCH --------------\n')
logger.info('rate     iter   epoch | loss  lb(lev) | loss0  loss1  | time          \n')
logger.info('----------------------------------------------------------------------\n')
            # 0.00000   0.00* 0.00  | 0.000  0.000  | 0.000  0.000  |  0 hr 00 min

# ----
valid_loss = np.zeros(2, np.float32)
train_loss = np.zeros(2, np.float32)
batch_loss = np.zeros_like(train_loss)
sum_train_loss = np.zeros_like(train_loss)
sum_train = 0
loss0 = torch.FloatTensor([0]).to(device).sum()
loss1 = torch.FloatTensor([0]).to(device).sum()
# loss2 = torch.FloatTensor([0]).cuda().sum()

start_timer = timer()
iteration = start_iteration
epoch = start_epoch
rate = 0
while iteration < num_iteration:

    for t, batch in enumerate(train_loader):

        if (iteration % iter_save == 0):
            if iteration != start_iteration:
                torch.save({
                    'state_dict': net.state_dict(),
                    'iteration': iteration,
                    'epoch': epoch,
                }, out_dir + '/checkpoint/{}_model.pth'.format(iteration))

        if (iteration % iter_valid == 0):
            if iteration != start_iteration:
                valid_loss = do_valid(net, tokenizer, valid_loader)


        if (iteration % iter_log == 0):
            if iteration != start_iteration:
                print('\r', end='', flush=True)
                logger.info(message(mode='log') + '\n')

        # learning rate schduler ------------
        rate = util.get_learning_rate(optimizer)

        # one iteration update  -------------
        batch_size = len(batch['index'])
        image  = batch['image'].to(device)
        # print(image.shape)
        token  = batch['token'].to(device)
        length = batch['length']

        # ----
        net.train()
        optimizer.zero_grad()

        if is_mixed_precision:
            # https://pytorch.org/docs/master/amp.html
            # https://pytorch.org/docs/master/notes/amp_examples.html#amp-examples
            with amp.autocast():
                #assert(False)
                logit = net(image, token, length)
                loss0 = model_ef.seq_anti_focal_cross_entropy_loss(logit, token, length)
                # loss0 = model_ef.seq_focal_cross_entropy_loss(logit, token, length)  #
                # loss0 = model_ef.seq_cross_entropy_loss(logit, token, length)#
            scaler.scale(loss0).backward()
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

        else:
            logit = net(image, token, length)
            loss0 = model_ef.seq_anti_focal_cross_entropy_loss(logit, token, length)
            # loss0 = model_ef.seq_focal_cross_entropy_loss(logit, token, length)  #

            # loss0 = model_ef.seq_cross_entropy_loss(logit, token, length)
            (loss0).backward()
            scheduler.step()
            optimizer.step()

        # print statistics  --------
        epoch += 1 / len(train_loader)
        iteration += 1

        # batch_loss = np.array([loss0.item(), loss1.item(), loss2.item()])
        batch_loss = np.array([loss0.item(), loss1.item()])
        sum_train_loss += batch_loss
        sum_train += 1
        # 求最近 100 个 batch 的平均 loss
        if iteration % 100 == 0:
            train_loss = sum_train_loss / (sum_train + 1e-12)
            sum_train_loss[...] = 0
            sum_train = 0

        print('\r', end='', flush=True)
        print(message(mode='print'), end='', flush=True)

logger.info('\n')


# main #################################################################
if __name__ == '__main__':
    pass


