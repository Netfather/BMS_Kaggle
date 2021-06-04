# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         DenoDeblur
# Description:  此文件尝试将推断过程中的图片进行去噪和去模糊
# 2021年5月27日 V1. 图像去模糊效果一般 不予采用 这里使用denosing 将所有的test_160W张照片都全部进行去噪
# Author:       Administrator
# Date:         2021/5/26
# -------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import cv2
import os
from sklearn import preprocessing
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from pathlib import Path
import urllib

import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from random import choice
from json import load
from io import BytesIO
import PIL.Image
import IPython.display
from joblib import load, dump

from math import trunc, ceil

import torch
import torch.nn as nn
import cv2

from tqdm import tqdm
## 去噪处理 使用opencv的fastNlMeansDenoising()  方法
# https://docs.opencv.org/master/d1/d79/group__photo__denoise.html#ga4c6b0031f56ea3f98f768881279ffe93

# 注意这个版本已经是针对服务器的版本了！！
csv_path = r"/storage/BMS_Molecular/data/bms_dataset/df_test.csv"
origin_name_pth = r"/storage/BMS_Molecular/data/bms_dataset"

#辅助函数  将读入的image_id装为可读取到的图像名字
def convert_to_path(split: str,image_name:str)-> str:
    return origin_name_pth + r"/{}/{}/{}/{}/{}.png".format(
        split, image_name[0], image_name[1], image_name[2], image_name
    )

def convert_to_output_path(split: str,image_name:str)-> str:
    return origin_name_pth + r"/{}/{}/{}/{}".format(
        split, image_name[0], image_name[1], image_name[2]
    )



# 读入文件名字
df_test = pd.read_csv(csv_path)
# 拿到image_ed列
images_id_list = df_test["image_id"]

total_num = len(df_test)
for id,image_name in enumerate(images_id_list):
    image_path = convert_to_path("test",image_name)
    output_path = convert_to_output_path("denoise_test", image_name)
    # 将指定为止创建出来
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    img = 255 - img

    conv = nn.Conv2d(1, 1, (3, 3), padding=1, bias=False)
    conv.weight = torch.nn.Parameter(torch.ones(3, 3)[None, None, :, :])
    conv.require_grad = False

    denoised = img.reshape(-1).copy()
    denoised[(conv(torch.Tensor(img[None, None, :, :])).detach().numpy() <= 255).squeeze().reshape(-1)] = 0
    img = denoised.reshape(img.shape)

    img = 255 - img


    cv2.imwrite(os.path.join(output_path,"{}.png".format(image_name)),img)

    print('\r %8d / %d ' % (id+1, total_num), end='',
          flush=True)

    # print(image_name)
    # print(id)



#
# img = cv.imread("./aabf6767af06.png", cv2.IMREAD_GRAYSCALE)
#
# img = 255 - img
#
# conv = nn.Conv2d(1, 1, (3, 3), padding=1, bias=False)
# conv.weight = torch.nn.Parameter(torch.ones(3, 3)[None, None, :, :])
# conv.require_grad = False
#
# denoised = img.reshape(-1).copy()
# denoised[(conv(torch.Tensor(img[None, None, :, :])).detach().numpy() <= 255).squeeze().reshape(-1)] = 0
# img = denoised.reshape(img.shape)
#
# img = 255 - img
#
# cv.imwrite("./denoised.png",img)


## 测试去模糊方法 数据集并没有非常模糊 但是椒盐噪声特别明显 确实需要进行去噪
from scipy.signal import convolve2d as conv2
# 效果一般 不予采用
# from skimage import color, data, restoration
# import cv2 as cv
# img = color.rgb2gray(cv.imread("./000e3230b02f.png"))
#
# psf = np.ones((5, 5)) / 25
#
# # Restore Image using Richardson-Lucy algorithm
# # deconvolved_RL = restoration.richardson_lucy(img, psf, iterations=30)
# deconvolved_RL = restoration.richardson_lucy(img, psf, iterations=30)
