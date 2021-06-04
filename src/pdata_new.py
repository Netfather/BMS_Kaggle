# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         pdata_new
# Description:  按照标签的token长度 来重新设计 每一个的pickle
#               由于val有限  直接指定
# Author:       Administrator
# Date:         2021/5/21
# -------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import gc
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
import cv2
from matplotlib import pyplot as plt
from imp import reload
from utils import util
reload(util)
import albumentations as A
import random

image_size = 224
def train_transforms():
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.OneOf([
            # 高斯噪点
            A.IAAAdditiveGaussianNoise(p = 1),
            A.GaussNoise(p = 1),
            # 模糊相关操作
            A.MotionBlur(p=1),
            A.MedianBlur(blur_limit=3, p=1),
            A.Blur(blur_limit=3, p=1),
        ], p=0.5),
        A.Resize(image_size, image_size),
    ], p=1.)

def val_transforms():
    return A.Compose([
        A.Resize(image_size,image_size),
    ], p=1.)

# 必要的超参数
Kfold = 40  # 指定预处理图片时使用的fold数量

# 编码 化学式
util.make_train(fold=Kfold)  # 读入train文件夹下的 train.csv文件然后将化学式编码 转变为一系列的数字
# 按照长度 对train进行排序 排序完成后划分为48份
util.make_test() #由于 test 不存在化学式标签 所以这里只是根据submitsample 来生成了一个空df_test.csv便于测试提交

# 随机抽取 48份中的 6个关键数字出 来组成训练数据
fold_idxs =  random.sample(range(0,Kfold-1),6) # 从0 到  39 中随机抽取 6个数字 构成pick
print(fold_idxs)

# 获取 fold 0 的数据
# 可以使用 kfold 模型融合技巧
for i,fold_id in enumerate(fold_idxs):
    print("Now Processing id :{}/{}".format(fold_id,Kfold))
    df_labels = pd.read_csv('../data/bms_dataset/df_train.csv')
    # 获取训练集
    df_train = df_labels[df_labels.fold!=fold_id]
    # 获取验证集
    df_valid = df_labels[df_labels.fold==fold_id]
    # 利用 eval 将 str -> list
    df_train['sequence'] = df_train.sequence.progress_apply(lambda x:eval(x))
    df_valid['sequence'] = df_valid.sequence.progress_apply(lambda x:eval(x))
    # 获取需要的列
    df_train = df_train[['image_id','InChI','formula','text','sequence','length']]
    df_valid = df_valid[['image_id','InChI','formula','text','sequence','length']]
    # 保存为 pkl 文件
    df_train.to_pickle('../data/df_train{}.pkl'.format(i+1))
    df_valid.to_pickle('../data/df_valid{}.pkl'.format(i+1))

    #一下为测试部分 测试 df_train中获取的信息是否正确  为了节省时间 只检测第一个fold的特征
    if(i == 0):
        # 加载分词器
        token = util.load_tokenizer()
        # 生成 torch 的 Dataset
        dataset_train = util.BmsDataset(df_train, token,augment= train_transforms())
        # 验证第 0 个的输出
        print(dataset_train[0])
        print('########################')
        print(dataset_train[0].keys())
        print('########################')
        print(len(dataset_train[0]['d']['text'].split(' ')) + 2 == len(dataset_train[0]['token']))

        # 检查训练集的 loader 和 collate_fn
        loader = DataLoader(
            dataset_train,
            sampler=RandomSampler(dataset_train),  # 抽样方法
            batch_size=8,
            drop_last=True,
            num_workers=0,
            pin_memory=True,  # 锁页内存
            collate_fn=util.collate_fn,  # 整理函数  在整理函数中 实现了照片的读入 如果需要用到增强 就在utils中改
        )
        for t, batch in enumerate(loader):
            if t > 2: break

            print(t, '-----------')
            print('index : ', batch['index'])
            print('image : ')
            print('\t', batch['image'].shape, batch['image'].is_contiguous())
            print('length  : ')
            print('\t', len(batch['length']))
            print('\t', batch['length'])
            print('token  : ')
            print('\t', batch['token'].shape, batch['token'].is_contiguous())
            print('\t', batch['token'])

            print('')

    # 删除这一个fold的数据信息
    del df_train,df_valid,df_labels
    gc.collect()

# 测试集做上面相同的处理
df_test = pd.read_csv('../data/bms_dataset/df_test.csv')
df_test['sequence'] = df_test.sequence.progress_apply(lambda x:eval(x))
df_test = df_test[['image_id','InChI','formula','text','sequence','length','orientation']]
# 保存为 pkl 文件
df_test.to_pickle('../data/df_test.pkl')

# 检查测试集数据集是否正确
token = util.load_tokenizer()
dataset_test = util.BmsDataset(df_test, token, mode = 'test', augment = util.rot_augment)
print(dataset_test[0])
print('########################')
print(dataset_test[0].keys())
print('########################')
# 检查测试集的 loader 和 collate_fn
loader = DataLoader(
        dataset_test,
        sampler = RandomSampler(dataset_test),
        batch_size  = 8,
        drop_last   = True,
        num_workers = 0,
        pin_memory  = True,
        collate_fn  = lambda batch: util.collate_fn(batch,False),
    )
for t,batch in enumerate(loader):
    if t>2: break

    print(t, '-----------')
    print('index : ', batch['index'])
    print('image : ')
    print('\t', batch['image'].shape, batch['image'].is_contiguous())
    print('length  : ')
    print('\t',len( batch['length']))
    print('\t', batch['length'])
    print('token  : ')
    print('\t', batch['token'].shape, batch['token'].is_contiguous())
    print('\t', batch['token'])

    print('')



