import torch
from torch import nn
import torch.nn.functional as F
import attention
import layerorm
import transformer
class PositionwiseFeed(nn.Module):   # ForwardFeed
    def __init__(self,d_model,hidden,dropout=0.1):
        super(PositionwiseFeed,self).__init__()
        self.fc1=nn.Linear(d_model,hidden)  # 全链接层
        self.fc2=nn.Linear(hidden,d_model)  # 经过一次映射后，重新映射回d_model维度
        self.dropout=nn.Dropout(dropout)  # 随机丢弃数据，防止过拟合


    def forward(self,x):
        x=self.fc1(x)  # ->(batch,hidden) 第一个全链接层
        x=F.relu(x)  # relu 激活函数，小于0的地方置为0，大于等于0的不变
        x=self.dropout(x)  # 防止过拟合
        x=self.fc2(x)  # ->(batch,d_model) 第二个全连接层
        return x

class EncoderLayer(nn.Module):  # Encoder
    def __init__(self,d_model,ffn_hidden,n_head,dropout=0.1):
        super(EncoderLayer,self).__init__()
        self.attention=attention.MutiHeadAttention(d_model,n_head)  # 交叉注意力层
        self.norm1=layerorm.LayerNorm(d_model)  # 第一个Norm层
        self.dropout1 = nn.Dropout(dropout)   # 第一个dropout层
        self.ffn=PositionwiseFeed(d_model,ffn_hidden,dropout)   # 前馈网络
        self.norm2 = layerorm.LayerNorm(d_model)  # 第二个Norm层
        self.dropout2 = nn.Dropout(dropout)   # 第二个dropout层

    def forward(self,x,mask=None):  # Encoder计算步骤
        _x=x
        x=self.attention(x,x,x,mask)
        x=self.dropout1(x)
        x=self.norm1(x+_x)  # x+_x残差进norm模块
        _x=x  # 更新残差计算
        x=self.ffn(x)
        x=self.dropout2(x)
        x=self.norm2(x+_x)

        return x

class Encoder(nn.Module):
    def __init__(self,device, enc_voc_size,max_len,d_model, ffn_hidden, n_head, n_layer, dropout=0.1 ):  # 词汇表大小、最大序列长度
        super(Encoder,self).__init__()
        self.embedding=transformer.TransformerEmbedding(enc_voc_size, d_model, dropout, device)
        self.layers=nn.ModuleList(   # 比nn.Sequential更简单的写法，但是中间层不能复用
            [
                EncoderLayer(d_model,ffn_hidden,n_head,device)
                for _ in range(n_layer)   # 创建n_layer个EncoderLayer存在列表中，之后可以遍历取出
            ]
        )

    def forward(self,x,s_mask=None):
        x=self.embedding(x)
        for layer in self.layers:
            x=layer(x,s_mask)   # 整个Encoder已经封装好了
        return x

