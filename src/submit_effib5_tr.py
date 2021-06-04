# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         submit_effib5_tr
# Description:  此文件用于生成 针对 efficientB5-transform的提交文件
# Author:       Administrator
# 2021年5月21日        V1. 由于 lb 和  cv之间存在巨大 gap  这里提交的时候 尝试直接使用resize  而不是用任何rotate 查看结果
#
# Date:         2021/5/19
# -------------------------------------------------------------------------------
import os
os.environ['CUDA_VISIBLE_DEVICES']  ="2"
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
from torch.utils.data.sampler import SequentialSampler
from utils.log_writers.log import get_logger
from torch.nn.parallel.data_parallel import data_parallel
from random import randrange
from imp import reload
from utils import util
reload(util)

# 一些必要的设置参数定义
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
device = torch.device('cuda:0')
Open_Parral_Training = False  #开启双显卡训练
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
manual_seed = randrange(10000)
Record_log_file = "train_efb_tr-2021-05-26-20-40.log"   # 本次提交模型的是在哪个log文件被记载的

local_cv = {
    "Fold" : 4,
    "Local LB(fixed)" :  "1.99",
    "Val_loss" : "x.xxxx"
}
# ----------------
is_mixed_precision = False #False  #
is_norm_ichi = False #True
# model_name = "res_trans"
model_name = "efb_trans"

if model_name == "efb_trans":
    fold = 4
    out_dir = '../model/efb_trans/fold{}'.format(fold)
    initial_checkpoint = out_dir + '/checkpoint/1920000_model.pth'  # None #

# 初始化日志头
logdir = "../submit/efb_trans/fold{}".format(fold)
logger = get_logger(logdir,OutputOnConsole= True,log_initial= "sub_efb_tr",logfilename="sub_efb_tr_noro_1920000")


# 模型输出位
submit_dir = '../submit/efb_trans/fold{}'.format(fold)
os.makedirs(submit_dir, exist_ok=True)

# 词典预定义的图片大小  词典大小  以及一个输出最长为多少
image_size = 384
#image_size = 224
vocab_size = 193
max_length = 300

###################################################################################################
import torch.cuda.amp as amp
if is_mixed_precision:
    class AmpNet(model_ef.Net):
        @torch.cuda.amp.autocast()
        def forward(self, *args):
            return super(AmpNet, self).forward_argmax_decode(*args)
else:
    AmpNet = model_ef.Net


# start here ! ###################################################################################
def do_predict(net, tokenizer, valid_loader):

    text = []

    start_timer = timer()
    valid_num = 0
    for t, batch in enumerate(valid_loader):
        batch_size = len(batch['image'])
        image = batch['image'].to(device)

        net.eval()
        with torch.no_grad():
            k = net.forward_argmax_decode(image)
            k = k.data.cpu().numpy()
            k = tokenizer.predict_to_inchi(k)
            text.extend(k)

        valid_num += batch_size
        print('\r %8d / %d  %s' % (valid_num, len(valid_loader.dataset), util.time_to_str(timer() - start_timer, 'sec')),
              end='', flush=True)


    #assert(valid_num == len(valid_loader.dataset))
    # assert(valid_num == 5000)
    print('')
    return text


if 1:

    ## setup  ----------------------------------------
    mode = 'local'

    ## 在log中写入必要信息  必须写入此次submit 的一系列超参数 所用模型 模型名字等关键参数
    logger.info('** Submit Model setting **\n')
    logger.info('Model Name : \n%s\n' % (model_name))
    logger.info('Checkpoint Pth \n%s\n' % (initial_checkpoint))
    logger.info('Record Log File  \n%s\n' % (Record_log_file))
    logger.info('Local CV  \n%s\n' % (local_cv))
    logger.info('\n')



    ## dataset ------------------------------------
    tokenizer = util.load_tokenizer()
    df_valid = pd.read_pickle('../data/df_test.pkl')

    valid_dataset = util.BmsDataset(df_valid, tokenizer, mode = 'denoise_test', augment=util.rot_augment,test_resize_shape= image_size)
    valid_loader  = DataLoader(
        valid_dataset,
        # sampler = SequentialSampler(valid_dataset),
        # Only For test
        sampler = util.FixNumSampler(valid_dataset, [0,30]),
        batch_size  = 32, #32,
        drop_last   = False,
        num_workers = 8,
        pin_memory  = True,
        collate_fn  = lambda batch: util.collate_fn(batch,False)
    )

    start_timer = timer()

    tokenizer = util.load_tokenizer()
    net = AmpNet().to(device)
    net.load_state_dict(torch.load(initial_checkpoint)['state_dict'], strict=True)

    #---
    predict = do_predict(net, tokenizer, valid_loader)

    logger.info('time taken : %s\n\n' % util.time_to_str(timer() - start_timer, 'min'))
    #----
    # df_submit = pd.DataFrame()
    # df_submit.loc[:, 'image_id'] = df_valid.image_id.values
    # df_submit.loc[:, 'InChI'] = predict  #
    # df_submit.to_csv(submit_dir + '/submit_no_rotate.csv', index=False)
    #
    # logger.info('submit_dir : %s\n' % (submit_dir))
    # logger.info('initial_checkpoint : %s\n' % (initial_checkpoint))
    # logger.info('df_submit : %s\n' % str(df_submit.shape))
    # logger.info('%s\n' % str(df_submit))
    # logger.info('\n')

if __name__ == '__main__':
    pass
