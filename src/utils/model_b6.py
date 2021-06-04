# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         model_b0
# Description:  2021年5月25日 修正 使用B6来进行模型探索
# Author:       Administrator
# Date:         2021/5/23
# -------------------------------------------------------------------------------
# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         model_ef
# Description:  此代码用于实现 efficientb3 + transform的模型构建
# Author:       Administrator
# Date:         2021/5/18
# -------------------------------------------------------------------------------
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn import functional as F
from utils.util import Swish,PositionEncode2D,PositionEncode1D
from utils.network_model.transformer import *  # 注意 这里的nn是指 utils文件夹下的nn
from torch import Tensor
from typing import Optional
import torch.nn as nn
import numpy as np
from efficientnet_pytorch import EfficientNet
import gc

image_dim   = 1024
text_dim    = 1024
decoder_dim = 1024

num_layer = 2
num_head = 8
ff_dim = 1024
# featuremap 的 h 和 w
num_pixel=7*7

# 如下参数并无修改作用 要修改必须去 utils中修改 这里只是展示目前使用的实际情况
image_size = 384
vocab_size = 193
max_length = 300

STOI = {
    '<sos>': 190,
    '<eos>': 191,
    '<pad>': 192,
}

pretrained_model_path = {
    "se_resnext101_32x4d":"./utils/pretrain_model/se_resnext101_32x4d-3b2fe3d8.pth",  #弃用 无法收敛
    "effecient_b5":"./utils/pretrain_model/efficientnet-b5-b6417697.pth"
}


class Encoder(nn.Module):
    """编码器类
    这里是通过 CNN 网络得到图像的 embed 向量
    可以使用 timm 库代替
    """
    def __init__(self,):
        super(Encoder, self).__init__()
        # 生成网络 并从硬盘加载权重
        model_name = 'effecient_b6'
        # print(pretrained_model_path[model_name])
        model = EfficientNet.from_pretrained('efficientnet-b6')

        # 分拆得到每个组件
        modules = []
        # stem
        modules.append(model._conv_stem)
        modules.append(model._bn0)
        modules.append(model._swish)

        # Blocks
        for idx, block in enumerate(model._blocks):
            drop_connect_rate = model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(model._blocks)  # scale drop connect_rate
            modules.append(block)
        # head
        modules.append(model._conv_head)
        modules.append(model._bn1)
        modules.append(model._swish)
        self.p1 = nn.Sequential(*modules)
        # V2. 2021年5月20日 修正 扩大图片的输入分辨率 扩大为384 需要在最后一层加入DownSample
        # self.downsample = nn.AdaptiveMaxPool2d(output_size=[7, 7])  # 这个尺寸是根据Transformer的大小决定的
        # V2. 2021年5月20日 修正 扩大图片的输入分辨率 扩大为384 需要在最后一层加入DownSample 现在用的是AdaptivePool 原来是maxpool
        self.downsample = nn.AdaptiveAvgPool2d(output_size=[7, 7])  # 这个尺寸是根据Transformer的大小决定的


        self.margin_cov = nn.Conv2d(2304,image_dim, kernel_size=1, bias=False)


        self.margin_bn = nn.BatchNorm2d(image_dim,affine= False)

        del model
        gc.collect()

    def forward(self, image):
        """前向传播
        """
        batch_size, C, H, W = image.shape
        # x = 2 * image - 1

        x = self.p1(image)
        x = self.margin_cov(x)
        c = self.margin_bn(x)
        x = self.downsample(x)
        # print(x.shape)
        return x

class Net(nn.Module):
    """编码器+解码器
    """
    def __init__(self,):
        super(Net, self).__init__()
        # CNN 编码器
        self.encoder = Encoder()
        # 针对 featuremap 的通道压缩 h*w*c
        self.image_embed = nn.Sequential(
            # nn.Conv2d(2048,image_dim, kernel_size=1, bias=None),  # 由于b1模型出来就是 1024的模型 因此这里直接删除掉这个部分
            nn.BatchNorm2d(image_dim),
            Swish()
        )
        # 针对 Input 的位置向量
        self.image_pos    = PositionEncode2D(image_dim,int(num_pixel**0.5)+1,int(num_pixel**0.5)+1)

        # transformer 的 encoder
        # 输入时间序列维度，前馈网络中间层维度，多头个数，层数
        self.image_encode = TransformerEncode(image_dim, ff_dim, num_head, num_layer)

        # 针对 Ouput 的位置向量
        self.text_pos    = PositionEncode1D(text_dim, max_length)
        # token 嵌入矩阵
        self.token_embed = nn.Embedding(vocab_size, text_dim)

        # transforer 的 decoder
        # 输出时间序列维度，前馈网络中间层维度，多头个数，层数
        self.text_decode = TransformerDecode(decoder_dim, ff_dim, num_head, num_layer)

        # 输出层
        self.logit  = nn.Linear(decoder_dim, vocab_size)
        self.dropout = nn.Dropout(p=0.5)

        # 初始化
        self.token_embed.weight.data.uniform_(-0.1, 0.1)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-0.1, 0.1)



    @torch.jit.unused
    def forward(self, image, token, length):
        """训练阶段：前向传播
        """
        device = image.device
        batch_size = len(image)

        # b*2048*h*w
        image_embed = self.encoder(image)

        # b*1024*h*w
        image_embed = self.image_embed(image_embed)

        # 给 imgae_embed 加上位置向量
        # b*1024*h*w
        image_embed = self.image_pos(image_embed)
        # h*w*b*1024
        image_embed = image_embed.permute(2,3,0,1).contiguous()
        # (h*w)*b*1024
        image_embed = image_embed.reshape(num_pixel, batch_size, image_dim)
        # 进入 transformer encoder (h*w)*b*1024
        image_embed = self.image_encode(image_embed)

        # b*n(v_n) -> b*n*text_dim
        text_embed = self.token_embed(token)
        # n*b*text_dim
        text_embed = self.text_pos(text_embed).permute(1,0,2).contiguous()

        # triu 上三角矩阵，对角线上移动 k=1
        text_mask = np.triu(np.ones((max_length, max_length)), k=1).astype(np.uint8)
        text_mask = torch.autograd.Variable(torch.from_numpy(text_mask)==1).to(device)

        # n*b*decoder_dim
        # 遮住一部分
        x = self.text_decode(text_embed, image_embed, text_mask)

        # b*n*decoder_dim
        x = x.permute(1,0,2).contiguous()

        # b*n*vocab_size
        logit = self.logit(x)
        return logit

    @torch.jit.export
    def forward_argmax_decode(self, image):
        """预测阶段：前向传播
        """
        device = image.device
        batch_size = len(image)

        #同上
        image_embed = self.encoder(image)
        image_embed = self.image_embed(image_embed)
        image_embed = self.image_pos(image_embed)
        image_embed = image_embed.permute(2,3,0,1).contiguous()
        image_embed = image_embed.reshape(num_pixel, batch_size, image_dim)
        image_embed = self.image_encode(image_embed)

        # b*n 填充 <pad>
        token = torch.full((batch_size, max_length), STOI['<pad>'],dtype=torch.long).to(device)
        # 获取输出向量的位置向量
        text_pos = self.text_pos.pos
        # 第一个设置为 <sos>
        token[:,0] = STOI['<sos>']
        eos = STOI['<eos>']
        pad = STOI['<pad>']
        # fast version
        # https://github.com/pytorch/fairseq/blob/21b8fb5cb1a773d0fdc09a28203fe328c4d2b94b/fairseq/sequence_generator.py#L245-L247
        if 1:
            incremental_state = torch.jit.annotate(
                Dict[str, Dict[str, Optional[Tensor]]],
                torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {}),
            )
            for t in range(max_length-1):
                # 目前 token 的最后一个值
                last_token = token[:, t]
                # 向量嵌入
                text_embed = self.token_embed(last_token)
                # 加上位置向量
                text_embed = text_embed + text_pos[:,t] #
                # b*text_dim -> 1*b*text_dim
                text_embed = text_embed.reshape(1,batch_size,text_dim)
                # 得到下一个向量 1*b*text_dim
                x = self.text_decode.forward_one(text_embed, image_embed, incremental_state)
                # b*text_dim
                x = x.reshape(batch_size,decoder_dim)
                # b*vocab_size
                l = self.logit(x)
                # 以最大的作为预测
                k = torch.argmax(l, -1)
                token[:, t+1] = k
                # 遇到 <eos> 和 <pad> 停止预测
                if ((k == eos) | (k == pad)).all():  break
        # 返回除了 <sos> 之外的序列
        predict = token[:, 1:]
        return predict

# loss #################################################################
def seq_cross_entropy_loss(logit, token, length):
    truth = token[:, 1:]
    L = [l - 1 for l in length]
    logit = pack_padded_sequence(logit, L, batch_first=True).data
    truth = pack_padded_sequence(truth, L, batch_first=True).data
    loss = F.cross_entropy(logit, truth, ignore_index=STOI['<pad>'])
    return loss


# focal_loss ###########################################################
def seq_anti_focal_cross_entropy_loss(logit,token,length):
    gamma = 0.5
    label_smooth = 0.90

    truth = token[:, 1:]
    L = [l - 1 for l in length]
    # 压缩序列
    logit = pack_padded_sequence(logit, L, batch_first=True).data
    # 压缩序列
    truth = pack_padded_sequence(truth, L, batch_first=True).data

    logp = F.log_softmax(logit,-1)
    logp = logp.gather(1,truth.reshape(-1,1)).reshape(-1)

    p = logp.exp()

    loss = -((1 + p) ** gamma) * logp

    loss = loss.mean()

    return loss



def seq_focal_cross_entropy_loss(logit,token,length):
    gamma = 1.0
    label_smooth = 0.90

    truth = token[:, 1:]
    L = [l - 1 for l in length]
    # 压缩序列
    logit = pack_padded_sequence(logit, L, batch_first=True).data
    # 压缩序列
    truth = pack_padded_sequence(truth, L, batch_first=True).data

    logp = F.log_softmax(logit,-1)
    logp = logp.gather(1,truth.reshape(-1,1)).reshape(-1)

    # p = logp.exp()
    p = logp.exp()

    loss = -((1.0 - p) ** gamma) * logp

    loss = loss.mean()

    return loss

# check #################################################################

def run_check_net():
    batch_size = 7
    C,H,W = 3, 224, 224
    image = torch.randn((batch_size,C,H,W))

    token  = np.full((batch_size, max_length), STOI['<pad>'], np.int64) #token
    length = np.random.randint(5,max_length-2, batch_size)
    length = np.sort(length)[::-1].copy()
    for b in range(batch_size):
        l = length[b]
        t = np.random.choice(vocab_size,l)
        t = np.insert(t,0,     STOI['<sos>'])
        t = np.insert(t,len(t),STOI['<eos>'])
        L = len(t)
        token[b,:L]=t

    token  = torch.from_numpy(token).long()
    net = Net()
    net.train()

    logit = net(image, token, length)
    print('vocab_size',vocab_size)
    print('max_length',max_length)
    print('')
    print(length)
    print(length.shape)
    print(token.shape)
    print(image.shape)
    print('---')

    print(logit.shape)
    print('---')



# main #################################################################
if __name__ == '__main__':
     run_check_net()