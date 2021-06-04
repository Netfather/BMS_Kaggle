import Levenshtein
import pickle
import pandas as pd
import numpy as np
import torch
import pickle
import re
from tqdm import tqdm
from collections import defaultdict
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import random
import math
import cv2
from tqdm import tqdm
import gc
from pathlib import Path

# Only For model ensamble
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn import functional as F
from torch import Tensor
from typing import Optional
from utils.network_model.transformer import *  # 注意 这里的nn是指 utils文件夹下的nn
# from network_model.transformer import *

tqdm.pandas(desc='apply')


data_dir = '../data'

STOI = {
    '<sos>': 190,
    '<eos>': 191,
    '<pad>': 192,
}

# 如下为模型继承相关参数 只在模型继承中使用到
vocab_size = 193
max_length = 300

image_dim   = 1024
text_dim    = 1024
decoder_dim = 1024

num_layer = 2
num_head = 8
ff_dim = 1024
# featuremap 的 h 和 w
num_pixel=7*7


class YNakamaTokenizer(object):
    """token类
    """

    def __init__(self, is_load=True):
        """初始化方法
        生成 stoi 字典和 itos 字典
        stoi : {char:int}
        itos : {int:char}
        """
        self.stoi = {}
        self.itos = {}

        if is_load:
            #{char:int}
            with open(data_dir+'/tokenizer.stoi.pickle','rb') as f:
                self.stoi = pickle.load(f)
            #{int:char}
            self.itos = {k: v for v, k in self.stoi.items()}

    def __len__(self):
        return len(self.stoi)

    def build_vocab(self, text):
        """根据文本生成字典
        :param: list, text 为分词后形成的 list e.g ['C 13 H 20 O S','C 21 H 30 O 4',...]
        """
        vocab = set()
        for t in text:
            vocab.update(t.split(' '))
        vocab = sorted(vocab)
        vocab.append('<sos>')
        vocab.append('<eos>')
        vocab.append('<pad>')
        for i, s in enumerate(vocab):
            self.stoi[s] = i
        self.itos = {k: v for v, k in self.stoi.items()}

    def one_text_to_sequence(self, text):
        """将 str 转换成 int list
        """
        sequence = []
        sequence.append(self.stoi['<sos>'])
        for s in text.split(' '):
            sequence.append(self.stoi[s])
        sequence.append(self.stoi['<eos>'])
        return sequence

    def one_sequence_to_text(self, sequence):
        """将 intlist 转换成 str
        """
        return ''.join(list(map(lambda i: self.itos[i], sequence)))

    def one_predict_to_inchi(self, predict):
        """将预测结果 (intlist) 转换为字符 (str)，组装为标准 InChI 格式
        e.g [190, 178, 47, 182, 89, 185, 187, 6, 13, 4, 165, 0, 88, 1, 154, 4, 69, 4, 47, 4, 132, 4, 121, 4, 14, 0, 99, 1, 143, 4, 36, 0, 47, 1, 25, 0, 110, 1, 58, 7, 121, 4, 143, 3, 165, 3, 25, 3, 58, 182, 3, 154, 182, 88, 3, 13, 4, 110, 182, 99, 191]
            ->
            InChI=1S/C13H20OS/c1-9(2)8-15-13-6-5-10(3)7-12(13)11(4)14/h5-7,9,11,14H,8H2,1-4H3 
        """
        # 添加头部
        inchi = 'InChI=1S/'
        # 遍历预测
        for p in predict:
            # 遇到 <eos> 或 <pad> 中止
            if p == self.stoi['<eos>'] or p == self.stoi['<pad>']:
                break
            inchi += self.itos[p]
        return inchi

    def text_to_sequence(self, text):
        """将多个 str 转换成 intlist
        """
        sequence = [
            self.one_text_to_sequence(t)
            for t in text
        ]
        return sequence

    def sequence_to_text(self, sequence):
        """将多个 intlist 转换成str
        """
        text = [
            self.one_sequence_to_text(s)
            for s in sequence
        ]
        return text

    def predict_to_inchi(self, predict):
        """将多个预测结果 (intlist) 转换为字符 (str)，组装为标准 InChI 格
        """
        inchi = [
            self.one_predict_to_inchi(p)
            for p in predict
        ]
        return inchi


class FixNumSampler(Sampler):
    """验集和测试集采样
    生成固定长度的采样
    """
    def __init__(self, dataset, length=(0,50000), is_shuffle=False):
        # if length<=0:
        #     length=len(dataset)

        self.is_shuffle = is_shuffle
        self.length = length


    def __iter__(self):
        assert(len(self.length) == 2)
        index = np.arange(self.length[0],self.length[1])
        if self.is_shuffle: random.shuffle(index)
        return iter(index)

    def __len__(self):
        return (self.length[1] - self.length[0])



def compute_lb_score(predict, truth):
    """评估函数，编辑距离计算   
    """
    score = []
    for p, t in zip(predict, truth):
        s = Levenshtein.distance(p, t)
        score.append(s)
    score = np.array(score)
    return score


def pad_sequence_to_max_length(sequence, max_length, padding_value):
    """填充文本
    :param: ndarray, 待填充序列
    :param: int, 最大填充长度
    :param: int, 填充值
    :return: ndarray, 填充后序列
    """
    # sequence: batchsize, length
    batch_size =len(sequence)
    # 生成 [batchsize,max_length] 全为 padding_value 的张量
    pad_sequence = np.full((batch_size,max_length), padding_value, np.int32)
    # 赋值，起到 padding 作用
    for b, s in enumerate(sequence):
        L = len(s)
        pad_sequence[b, :L] = s
    return pad_sequence

def load_tokenizer():
    """加载tokenizer
    """
    tokenizer = YNakamaTokenizer(is_load=True)
    print('len(tokenizer) : vocab_size', len(tokenizer))
    return tokenizer

def split_form1(form):
    """化学式预处理 
    :param: str 化学式
    :return: str 分词，用空格分开 e.g C13H20OS -> C 13 H 20 O S
    """
    string = ''
    # 正则表达式获取以一个大写字母开头，任意多个小写字母和数字结尾的组合。 e.g C13 Br
    for i in re.findall(r"[A-Z][^A-Z]*", form):
        # 匹配其中的字母
        elem = re.match(r"\D+", i).group()
        # 得到其中的数字
        num = i.replace(elem, "")
        #用空格做连接
        if num == "":
            string += f"{elem} "
        else:
            string += f"{elem} {str(num)} "
    # 去除末尾空格
    return string.rstrip(' ')

def split_form2(form):
    """原子连接预处理
    :param: str 原子连接式
    :return: str 分词，用空格分开 e.g c1-9(2)8-15-13-6-5-10(3)7-12(13)11(4)14 -> /c 1 - 9 ( 2 ) 8 - 15 - 13 - 6 - 5 - 10 ( 3 ) 7 - 12 ( 13 ) 11 ( 4 ) 14
    """
    string = ''
    for i in re.findall(r"[a-z][^a-z]*", form):
        elem = i[0]
        num = i.replace(elem, "").replace('/', "")
        num_string = ''
        for j in re.findall(r"[0-9]+[^0-9]*", num):
            num_list = list(re.findall(r'\d+', j))
            assert len(num_list) == 1, f"len(num_list) != 1"
            _num = num_list[0]
            if j == _num:
                num_string += f"{_num} "
            else:
                extra = j.replace(_num, "")
                num_string += f"{_num} {' '.join(list(extra))} "
        string += f"/{elem} {num_string}"
    return string.rstrip(' ')

def split_form3(formlst):
    """原子连接处理
    :param list 多个原子连接式
    :return: str 分词，用空格分开 e.g [c1-9(2)8-15-13-6-5-10(3)7-12(13)11(4)14,h5-7,9,11,14H,8H2,1-4H3] -> /c 1 - 9 ( 2 ) 8 - 15 - 13 - 6 - 5 - 10 ( 3 ) 7 - 12 ( 13 ) 11 ( 4 ) 14 /h 5 - 7 , 9 , 11 , 14 H , 8 H 2 , 1 - 4 H 3
    """
    string = ''
    for form in formlst:
        string+= ' '+split_form2(form)
    return string.rstrip(' ')



def make_train(fold = 6, random_seed=666):
    """生成训练集
    :param: fold 生成的 fold 数
    :param: random_seed 随机种子

    :return: DataFrame  e.g 'image_id', 'InChI', 'formula', 'text', 'sequence', 'length'
    """
    # 加载标记器
    token = load_tokenizer()
    # 读取训练数据 InChI=1S/C13H20OS/c1-9(2)8-15-13-6-5-10(3)7-12(13)11(4)14/h5-7,9,11,14H,8H2,1-4H3
    df_label = pd.read_csv(data_dir+'/bms_dataset/train_labels.csv')
    # df_label = df_label.head(n = 30)
    # 加工 formula e.g C13H20OS
    df_label['formula'] = df_label.InChI.progress_apply(lambda x:x.split('/')[1])
    # 加工 text e.g  C 13 H 20 O S /c 1 - 9 ( 2 ) 8 - 15 - 13 - 6 - 5 - 10 ( 3 ) 7 - 12 ( 13 ) 11 ( 4 ) 14 /h 5 - 7 , 9 , 11 , 14 H , 8 H 2 , 1 - 4 H 3
    df_label['text'] = df_label.formula.progress_apply(lambda x:split_form1(x))+df_label.InChI.progress_apply(lambda x:split_form3(x.split('/')[2:]))
    # 将 str 转换成 index 包含 <sos> <eos> e.g  [190, 178, 47, 182, 89, 185, 187, 6, 13, 4, 165, 0, 88, 1, 154, 4, 69, 4, 47, 4, 132, 4, 121, 4, 14, 0, 99, 1, 143, 4, 36, 0, 47, 1, 25, 0, 110, 1, 58, 7, 121, 4, 143, 3, 165, 3, 25, 3, 58, 182, 3, 154, 182, 88, 3, 13, 4, 110, 182, 99, 191]
    df_label['sequence'] = df_label.text.progress_apply(lambda x:token.one_text_to_sequence(x))
    # 加工 token 长度 59 不包含 <sos> <eos>
    df_label['length'] = df_label.sequence.progress_apply(lambda x:len(x)-2)

    # V4 修正 根据 length 对数据进行排序
    df_label = (df_label.sort_values(["length"]))

    # 生成 fold 值
    fold_lst = (len(df_label)//fold + 1)*[i for i in range(fold)]
    # fold_lst = fold_lst[:len(df_label)]  # 从开始取到 数组长度
    # r=random.random
    # random.seed(random_seed)
    # random.shuffle(fold_lst,random=r)
    df_label['fold'] = fold_lst[:len(df_label)]
    # 保存为 CSV 数据
    print(df_label)  # 大致看一下结果
    df_label.to_csv(data_dir+'/bms_dataset/df_train.csv')


def make_test():
    """生成测试集
    :return: DataFrame  e.g 'image_id', 'InChI', 'formula', 'text', 'sequence', 'length'
    """
    # 加载测试集
    df = pd.read_csv(data_dir+'/bms_dataset/sample_submission.csv')
    # 对测试集做旋转
    df_orientation = pd.read_csv(data_dir+'/test_orientation.csv')  #读入一下 data文件夹下的  测试机旋转文件
    df = df.merge(df_orientation, on='image_id')

    # 和训练集的数据保持一致
    df.loc[:, 'path'] = 'test'
    df.loc[:, 'InChI'] = '0'
    df.loc[:, 'formula'] = '0'
    df.loc[:, 'text'] =  '0'
    df.loc[:, 'sequence'] = pd.Series([[0]] * len(df))
    df.loc[:, 'length'] = 1

    # 保存为 CSV 数据
    df_test = df
    df_test.to_csv(data_dir+'/bms_dataset/df_test.csv')


#####################################
# 数据加载
#####################################

def rot_augment(r,test_size = 320):
    """图像预处理函数
    以 90 度为标准单位对图像进行旋转
    """
    image = r['image']
    h, w = image.shape


    # 2021年5月29日 更新  必须加入 旋转文件判定旋转 否则会导致下降0.2个点左右
    l= r['d'].orientation
    if l == 1:
        image = np.rot90(image, -1)
    if l == 2:
        image = np.rot90(image, 1)
    if l == 3:
        image = np.rot90(image, 2)

    # 2021年5月26日 新增 对test_image在每次读入的时候进行去噪处理
    # 图像 resize
    image = cv2.resize(image, dsize=(test_size,test_size), interpolation=cv2.INTER_LINEAR)
    r['image'] = image

    #del conv,denoised,img
    #gc.collect()
    return r

def null_augment(r,test_size = 320):
    """图像预处理函数
    """
    image = r['image']
    image = cv2.resize(image, dsize=(test_size,test_size), interpolation=cv2.INTER_LINEAR)
    r['image'] = image
    return r


class BmsDataset(Dataset):
    """Dataset 类
    """
    def __init__(self, df, tokenizer, mode = 'train' ,augment=null_augment, test_resize_shape = 320):
        """初始化方法
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.df = df
        self.augment = augment
        self.mode = mode
        self.length = len(self.df)
        self.test_resize_shape = test_resize_shape

    def __str__(self):
        string  = ''
        string += '\tlen = %d\n'%len(self)
        string += '\tdf  = %s\n'%str(self.df.shape)
        return string

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        """索引方法
        """
        d = self.df.iloc[index]

        image_file = data_dir+'/bms_dataset/{}'.format(self.mode) +'/{}/{}/{}/{}.png'.format(d.image_id[0], d.image_id[1], d.image_id[2], d.image_id)
        # 加载图像，读取为灰度图像
        image = cv2.imread(image_file,cv2.IMREAD_GRAYSCALE)
        token = d.sequence
        r = {
            'index' : index,
            'image_id' : d.image_id,
            'InChI' : d.InChI,
            'formula' : d.formula,
            'd' : d,
            'image' : image,
            'token' : token,
        }
        # 图像预处理
        # if self.augment is not None: r = self.augment(r)
        # 2021年5月18日 加入新的数据增强手段
        if self.mode == 'test':
            r = self.augment(r,self.test_resize_shape)
        elif self.mode == 'denoise_test':
            r = self.augment(r,self.test_resize_shape)
        else:
            image = self.augment(image=image)['image']
            r['image'] = image
        return r


def collate_fn(batch, is_sort_decreasing_length=True):
    """配置函数
    :param: batch 数据
    :is_sort_decreasing_length: boolen 是否需要做降序排列
    """
    collate = defaultdict(list)

    if is_sort_decreasing_length: #按照 token 长度做降序排列
        sort  = np.argsort([-len(r['token']) for r in batch])
        batch = [batch[s] for s in sort]

    for r in batch:
        for k, v in r.items():
            collate[k].append(v)

    collate['length'] = [len(l) for l in collate['token']]

    # token标签的处理
    token  = [np.array(t,np.int32) for t in collate['token']]
    token  = pad_sequence_to_max_length(token, max_length=max_length, padding_value=STOI['<pad>'])
    collate['token'] = torch.from_numpy(token).long()

    # 图像数据的处理
    image = np.stack(collate['image'])
    image = image.astype(np.float32) / 255
    collate['image'] = torch.from_numpy(image).unsqueeze(1).repeat(1,3,1,1)

    return collate


def np_loss_cross_entropy(probability, truth):
    """np交叉熵损失函数
    :param: array probability [bs,dim]
    :param: array truth       [bs]
    """
    batch_size = len(probability)
    truth = truth.reshape(-1)
    p = probability[np.arange(batch_size),truth]
    loss = -np.log(np.clip(p,1e-6,1))
    loss = loss.mean()
    return loss

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
        lr +=[ param_group['lr'] ]

    assert(len(lr)>=1) #we support only one param_group
    lr = lr[0]

    return lr


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


# conda install -y -c rdkit rdkit

#############################################
class Swish(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


#https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py
class PositionEncode2D(torch.nn.Module):
    def __init__(self, dim, width, height):
        super().__init__()
        assert (dim % 4 == 0)
        self.width  = width
        self.height = height

        dim = dim//2
        d = torch.exp(torch.arange(0., dim, 2) * -(math.log(10000.0) / dim))
        position_w = torch.arange(0., width ).unsqueeze(1)
        position_h = torch.arange(0., height).unsqueeze(1)
        pos = torch.zeros(1, dim*2, height, width)

        pos[0,      0:dim:2, :, :] = torch.sin(position_w * d).transpose(0, 1).unsqueeze(1).repeat(1,1, height, 1)
        pos[0,      1:dim:2, :, :] = torch.cos(position_w * d).transpose(0, 1).unsqueeze(1).repeat(1,1, height, 1)
        pos[0,dim + 0:   :2, :, :] = torch.sin(position_h * d).transpose(0, 1).unsqueeze(2).repeat(1,1, 1, width)
        pos[0,dim + 1:   :2, :, :] = torch.cos(position_h * d).transpose(0, 1).unsqueeze(2).repeat(1,1, 1, width)
        self.register_buffer('pos', pos)

    def forward(self, x):
        batch_size,C,H,W = x.shape
        x = x + self.pos[:,:,:H,:W]
        return x

# https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
# https://stackoverflow.com/questions/46452020/sinusoidal-embedding-attention-is-all-you-need

class PositionEncode1D(torch.nn.Module):
    def __init__(self, dim, max_length):
        super().__init__()
        assert (dim % 2 == 0)
        self.max_length = max_length

        d = torch.exp(torch.arange(0., dim, 2)* (-math.log(10000.0) / dim))
        position = torch.arange(0., max_length).unsqueeze(1)
        pos = torch.zeros(1, max_length, dim)
        pos[0, :, 0::2] = torch.sin(position * d)
        pos[0, :, 1::2] = torch.cos(position * d)
        self.register_buffer('pos', pos)

    def forward(self, x):
        batch_size, T, dim = x.shape
        x = x + self.pos[:,:T]
        return x


@torch.jit.export
def ModelsEnsamble(models:list,image):
    '''
    用于模型集成的提交记录
    :param models: 输入的模型列表
    :param image: 输入的batch图片
    :return:
    '''
    device = image.device
    batch_size = len(image)

    # b*n 填充 <pad>  token为最终填写的答案大小
    token = torch.full((batch_size, max_length), STOI['<pad>'], dtype=torch.long).to(device)

    # Only For debug
    # token1 = torch.full((batch_size, max_length), STOI['<pad>'], dtype=torch.long).to(device)
    # token2 = torch.full((batch_size, max_length), STOI['<pad>'], dtype=torch.long).to(device)

    token[:, 0] = STOI['<sos>']  # 设定起步位置

    # Only For debug
    # token1[:, 0] = STOI['<sos>']  # 设定起步位置
    # token2[:, 0] = STOI['<sos>']  # 设定起步位置

    eos = STOI['<eos>']
    pad = STOI['<pad>']

    incremental_state = torch.jit.annotate(
        Dict[str, Dict[str, Optional[Tensor]]],
        torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {}),
    )

    ## 这部分是从图像中提取到特征 是固定的部分
    # 预处理图像特征
    # 拆开写
    # 模型1  encoder处理
    model_efb = models[0]
    model_efb.eval()
    model_efb_image_embed = model_efb.encoder(image)
    model_efb_image_embed = model_efb.image_embed(model_efb_image_embed)
    model_efb_image_embed = model_efb.image_pos(model_efb_image_embed)
    model_efb_image_embed = model_efb_image_embed.permute(2, 3, 0, 1).contiguous()
    model_efb_image_embed = model_efb_image_embed.reshape(num_pixel, batch_size, image_dim)
    model_efb_image_embed = model_efb.image_encode(model_efb_image_embed)  # 此时的shape 以 batchsize 30  锚定7*7 为例子    torch.Size([49, 30, 1024])
    model_efb_text_pos = model_efb.text_pos.pos

    # 模型2 encoder处理
    model_eb6 = models[1]
    model_eb6.eval()
    model_eb6_image_embed = model_eb6.encoder(image)
    model_eb6_image_embed = model_eb6.image_embed(model_eb6_image_embed)
    model_eb6_image_embed = model_eb6.image_pos(model_eb6_image_embed)
    model_eb6_image_embed = model_eb6_image_embed.permute(2, 3, 0, 1).contiguous()
    model_eb6_image_embed = model_eb6_image_embed.reshape(num_pixel, batch_size, image_dim)
    model_eb6_image_embed = model_eb6.image_encode(model_eb6_image_embed)
    model_eb6_text_pos = model_eb6.text_pos.pos

    for t in range(max_length - 1):

        last_token = token[:, t]  # 此时的last_token为batchsize 大小 一维  【batchsize】
        # ## Only for test
        # print("last_token's shape: {}".format(last_token.shape))

        ## model1 model_efb处理
        # 向量嵌入
        model_efb_text_embed = model_efb.token_embed(last_token)
        # 加上位置向量
        model_efb_text_embed = model_efb_text_embed + model_efb_text_pos[:, t]  #
        # b*text_dim -> 1*b*text_dim
        model_efb_text_embed = model_efb_text_embed.reshape(1, batch_size, text_dim)
        # 得到下一个向量 1*b*text_dim
        model_efb_x = model_efb.text_decode.forward_one(model_efb_text_embed, model_efb_image_embed, incremental_state)
        # b*text_dim
        model_efb_x = model_efb_x.reshape(batch_size, decoder_dim)
        # b*vocab_size
        model_efb_l = model_efb.logit(model_efb_x)

        # model_efb_k = torch.argmax(model_efb_l, -1)
        # token1[:, t + 1] = model_efb_k


        ## ONly for debug
        # model_efb_res  = torch.argmax(model_efb_l, -1)
        # token1[:, t + 1] = model_efb_res

        last_token = token[:, t]  # 此时的last_token为batchsize 大小 一维  【batchsize】
        ## model2 model_eb6处理
        # 向量嵌入
        model_eb6_text_embed = model_eb6.token_embed(last_token)
        # 加上位置向量
        model_eb6_text_embed = model_eb6_text_embed + model_eb6_text_pos[:, t]  #
        # b*text_dim -> 1*b*text_dim
        model_eb6_text_embed = model_eb6_text_embed.reshape(1, batch_size, text_dim)
        # 得到下一个向量 1*b*text_dim
        model_eb6_x = model_eb6.text_decode.forward_one(model_eb6_text_embed, model_eb6_image_embed, incremental_state)
        # b*text_dim
        model_eb6_x = model_eb6_x.reshape(batch_size, decoder_dim)
        # b*vocab_size
        model_eb6_l = model_eb6.logit(model_eb6_x)

        # model_eb6_k = torch.argmax(model_eb6_l, -1)
        # token2[:, t + 1] = model_eb6_k

        ## ONly for debug
        # model_eb6_res  = torch.argmax(model_eb6_l, -1)
        # token2[:, t + 1] = model_eb6_res


        # 将结果分别做softmax
        model_efb_l = torch.softmax(model_efb_l, dim= -1)
        model_eb6_l = torch.softmax(model_eb6_l, dim= -1)
        # 结果直接相加 然后做argmax
        l = torch.add(model_efb_l,model_eb6_l)

        # print("res's shape:{}".format(l.shape))

        k = torch.argmax(l, -1)
        token[:, t + 1] = k
        # 遇到 <eos> 和 <pad> 停止预测
        if ((k == eos) | (k == pad)).all():  break

        # model_result_logit[0] # 这是一个 [batchsize,voca_size]大小的变量
    predict = token[:, 1:]
    # predict1 = token1[:, 1:]
    # predict2 = token2[:, 1:]
    ## 释放内存和显存
    del model_efb_image_embed,model_efb_text_pos,model_eb6_image_embed,model_eb6_text_pos,model_efb_text_embed,model_efb_x,model_efb_l,model_eb6_text_embed,model_eb6_x,model_eb6_l,l,k,token,last_token
    gc.collect()
    torch.cuda.empty_cache()
    return predict
    # return predict,predict1,predict2





if __name__ == '__main__':
    pass