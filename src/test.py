# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         test
# Description:  此文件用于测试 并进行给定pth的裁剪
# Author:       Administrator
# Date:         2021/5/17
# -------------------------------------------------------------------------------
# import torch
# from collections import OrderedDict
# import os
# import torch.nn as nn
#
# initial_checkpoint = '../model/res_trans/1360000_model.pth'
#
# # for key in ["aux_classifier.0.weight", "aux_classifier.1.weight", "aux_classifier.1.bias",
# #             "aux_classifier.1.running_mean", "aux_classifier.1.running_var", "aux_classifier.1.num_batches_tracked",
# #             "aux_classifier.4.weight", "aux_classifier.4.bias"]:
# #     del pth[key]
#
# f = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
# start_iteration = f['iteration']
# start_epoch     = f['epoch']
# state_dict = f['state_dict']
# print(list(state_dict.keys()))
#
# for element in list(state_dict.keys()):
#     if (element[0:7] == "encoder"):
#         del state_dict[element]
# # if list(state_dict.keys())[0].startswith('module'):

## 测试efficientnet b3 的模型结构并进行测试优化
# https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py
# 根据原作者的指引 成功分离出最终的步长
# from torch.nn.utils.rnn import pack_padded_sequence
# from torch.nn import functional as F
# from utils.util import Swish,PositionEncode2D,PositionEncode1D
# from utils.network_model.transformer import *  # 注意 这里的nn是指 utils文件夹下的nn
# from torch import Tensor
# from typing import Optional
# import torch.nn as nn
# import numpy as np
# import pretrainedmodels
# from efficientnet_pytorch import EfficientNet
# import segmentation_models_pytorch as smp
# import torch
# import gc
#
# model = EfficientNet.from_pretrained('efficientnet-b5').cuda()
# # print(model)
# inputs = torch.randn(size=[1,3,224,224]).cuda()
# modules = []
# # stem
# modules.append(model._conv_stem)
# modules.append(model._bn0)
# modules.append(model._swish)
#
# # Blocks
# for idx, block in enumerate(model._blocks):
#     drop_connect_rate = model._global_params.drop_connect_rate
#     if drop_connect_rate:
#         drop_connect_rate *= float(idx) / len(model._blocks)  # scale drop connect_rate
#     modules.append(block)
# # head
# modules.append(model._conv_head)
# modules.append(model._bn1)
# modules.append(model._swish)
# p1 = nn.Sequential(*modules)
# del model
# gc.collect()
#
# print(p1)
# p1.eval()
# out1 = p1(inputs)
# print(out1.shape)

## 测试 学习率变化曲线
# import torch
# import matplotlib.pyplot as plt
#
# model = torch.nn.Linear(2, 1)
# optimizer = torch.optim.SGD(model.parameters(), lr=7e-5)
# lr_sched = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.2, div_factor=1e2, max_lr=2e-4,steps_per_epoch=10000, epochs=20)
#
#
# lrs = []
#
# for i in range(20 * 10000):
#     lr_sched.step()
#     lrs.append(
#         optimizer.param_groups[0]["lr"]
#     )
#
# plt.plot(lrs)
# plt.show()

## 生成每天的日志图
# from utils.plot_box.plot_logger import plot_logger
# source = r"F:\BMS_Molecular_Sub\total_figure_eb6"
#
# plot_logger(
#     source,
#     source + r"\log.png",
#     figure_title="train_val_lb")
##  测试新的swin _transformer
# import timm
# import torch
# model = timm.models.tnt_s_patch16_224(pretrained= True)
# input = torch.randn(size = [1,3,384,384])
# model.eval()
# output = model(input)
# print(output.shape)

## 测试softmax的作用维度
# import numpy as np
# import torch
# test= np.array([
#     [0,1,2],
#     [-20,1,20]])
# input = torch.tensor(test,dtype=torch.float)
#
# print(input.shape)
# softmax_test = torch.softmax(input,dim= -1)
# print(softmax_test)
#
# print(9.0031e-02 + 2.4473e-01 +6.6524e-01)
# Test OK

## 尝试将lnh分子式变成化学图
from rdkit import Chem
from rdkit import RDLogger

inchi = "InChI=1S/C10H14BrN5S/c1-6-10(11)9(16(3)14-6)4-7(12-2)8-5-13-17-15-8/h5,7,12H,4H2,1-3H3"
mol = Chem.MolFromInchi(inchi)
print(mol)

