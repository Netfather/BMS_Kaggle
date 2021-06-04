# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         submit_ensamble
# Description:  此文件用于进行模型融合
# Author:       Administrator
# Date:         2021/5/29
# -------------------------------------------------------------------------------
# ModelsEnsamble
import os
os.environ['CUDA_VISIBLE_DEVICES']  ="0"
import numpy as np
import pandas as pd
import torch
from utils import model_ef,model_b6
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

is_mixed_precision = False #False  #
is_norm_ichi = False #True

## model_1 efb
Model1_Record_log_file = "train_efb_tr_avepool-2021-06-02-11-21.log"   # 本次提交模型的是在哪个log文件被记载的

Model1_local_cv = {
    "Fold" : 4,
    "Local LB(fixed)" :  "0.92",
    "Val_loss" : "x.xxxx"
}
# ----------------
# model_name = "res_trans"
Model1_model_name = "efb_trans"

if Model1_model_name == "efb_trans":
    Model1_fold = 4
    Model1_out_dir = '../model/efb_trans/fold{}'.format(Model1_fold)
    Model1_initial_checkpoint = Model1_out_dir + '/checkpoint/2368000_model.pth'  # None #

## model_2 eb6
Model2_Record_log_file = "train_eb6_tr_avepool-2021-06-02-11-18.log"   # 本次提交模型的是在哪个log文件被记载的

Model2_local_cv = {
    "Fold": 5,
    "Local LB(fixed)": "1.08",
    "Val_loss": "x.xxxx"
}
# ----------------
# model_name = "res_trans"
Model2_model_name = "eb6_trans"

if Model2_model_name == "eb6_trans":
    Model2_fold = 5
    Model2_out_dir = '../model/eb6_trans/fold{}'.format(Model2_fold)
    Model2_initial_checkpoint = Model2_out_dir + '/checkpoint/2064000_model.pth'  # None #

#################################################################
# 初始化日志头
logdir = "../submit/model_ensm_final320"
logger = get_logger(logdir,OutputOnConsole= True,log_initial= "sub_model_ensm_final320",logfilename="sub_model_ensm_final320")


# 模型输出位
submit_dir = '../submit/model_ensm_final320'
os.makedirs(submit_dir, exist_ok=True)

# 词典预定义的图片大小  词典大小  以及一个输出最长为多少
# image_size = 384  ########################################记得修改回来！！！！！
image_size = 320  ########################################记得修改回来！！！！！
#image_size = 224
vocab_size = 193
max_length = 300

###################################################################################################
import torch.cuda.amp as amp
# 由于不需要混精度了 因此关闭这个选项
# if is_mixed_precision:
#     class AmpNet(model_ef.Net):
#         @torch.cuda.amp.autocast()
#         def forward(self, *args):
#             return super(AmpNet, self).forward_argmax_decode(*args)
# else:
#     AmpNet = model_ef.Net


# start here ! ###################################################################################
def do_predict(models, tokenizer, valid_loader):

    text = []

    start_timer = timer()
    valid_num = 0
    for t, batch in enumerate(valid_loader):
        batch_size = len(batch['image'])
        image = batch['image'].to(device)

        with torch.no_grad():
            k = util.ModelsEnsamble(models,image)
        # k = net.forward_argmax_decode(image)
            k = k.data.cpu().numpy()
            k = tokenizer.predict_to_inchi(k)
            text.extend(k)
            #
            # k1 = k1.data.cpu().numpy()
            # k1 = tokenizer.predict_to_inchi(k1)
            # text.extend(k1)
            # k2 = k2.data.cpu().numpy()
            # k2 = tokenizer.predict_to_inchi(k2)
            # text.extend(k2)

        valid_num += batch_size
        print('\r %8d / %d  %s' % (valid_num, len(valid_loader.dataset), util.time_to_str(timer() - start_timer, 'sec')),
              end='', flush=True)


    #assert(valid_num == len(valid_loader.dataset))
    # assert(valid_num == 5000)
    print('')
    # print(text) # Only For test
    return text


if 1:

    ## setup  ----------------------------------------
    mode = 'local'

    ## 在log中写入必要信息  必须写入此次submit 的一系列超参数 所用模型 模型名字等关键参数
    logger.info('** Submit Model1 setting **\n')
    logger.info('Model Name : \n%s\n' % (Model1_model_name))
    logger.info('Checkpoint Pth \n%s\n' % (Model1_initial_checkpoint))
    logger.info('Record Log File  \n%s\n' % (Model1_Record_log_file))
    logger.info('Local CV  \n%s\n' % (Model1_local_cv))
    logger.info('\n')


    logger.info('** Submit Model2 setting **\n')
    logger.info('Model Name : \n%s\n' % (Model2_model_name))
    logger.info('Checkpoint Pth \n%s\n' % (Model2_initial_checkpoint))
    logger.info('Record Log File  \n%s\n' % (Model2_Record_log_file))
    logger.info('Local CV  \n%s\n' % (Model2_local_cv))
    logger.info('\n')


    ## dataset ------------------------------------
    tokenizer = util.load_tokenizer()
    df_valid = pd.read_pickle('../data/df_test.pkl')

    valid_dataset = util.BmsDataset(df_valid, tokenizer, mode = 'denoise_test', augment=util.rot_augment,test_resize_shape= image_size)
    valid_loader  = DataLoader(
        valid_dataset,
        sampler = SequentialSampler(valid_dataset),
        # Only For test
        # sampler = util.FixNumSampler(valid_dataset, [0,30]),

        batch_size  = 64, #32,
        drop_last   = False,
        num_workers = 8,
        pin_memory  = True,
        collate_fn  = lambda batch: util.collate_fn(batch,False)
    )

    start_timer = timer()

    tokenizer = util.load_tokenizer()

    net1 = model_ef.Net().to(device)
    net1.load_state_dict(torch.load(Model1_initial_checkpoint)['state_dict'], strict=True)

    net2= model_b6.Net().to(device)
    net2.load_state_dict(torch.load(Model2_initial_checkpoint)['state_dict'], strict=True)
    #---
    predict = do_predict([net1,net2], tokenizer, valid_loader)

    logger.info('time taken : %s\n\n' % util.time_to_str(timer() - start_timer, 'min'))
    #----
    df_submit = pd.DataFrame()
    df_submit.loc[:, 'image_id'] = df_valid.image_id.values
    df_submit.loc[:, 'InChI'] = predict  #
    df_submit.to_csv(submit_dir + '/submit.csv', index=False)

    logger.info('submit_dir : %s\n' % (submit_dir))
    logger.info('df_submit : %s\n' % str(df_submit.shape))
    logger.info('%s\n' % str(df_submit))
    logger.info('\n')

if __name__ == '__main__':
    pass
