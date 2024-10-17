import torch
from torch import nn
import torch.nn.functional as F
import encoder
import attention
import layerorm
import embedding

class DecoderLayer(nn.Module):
    def __init__(self,d_model,ffn_hidden, n_head, drop_prob):  # ffn为前馈神经网络的隐藏层
        super(DecoderLayer,self).__init__()
        self.attention1=attention.MutiHeadAttention(d_model,n_head)
        self.norm1=layerorm.LayerNorm(d_model)
        self.dropout1=nn.Dropout(drop_prob)
        self.cross_attention=attention.MutiHeadAttention(d_model,n_head)
        self.norm2=layerorm.LayerNorm(d_model)
        self.dropout2=nn.Dropout(drop_prob)
        self.ffn=encoder.PositionwiseFeed(d_model,ffn_hidden, drop_prob)
        self.norm3=layerorm.LayerNorm(d_model)
        self.dropout3=nn.Dropout(drop_prob)

    def forward(self,dec,enc,t_mask,s_mask): # decoder,encoder t_mask为Decoder解码的注意力，s_mask为源mask
        _x=dec  # 复制备用残差
        x=self.attention1(dec,dec,dec,t_mask)  # 第一个结构，解码自注意力
        x=self.dropout1(x)
        x=self.norm1(x+_x)   # 第一个Add与Norm
        _x=x
        x=self.cross_attention(x,dec,enc,s_mask)  # 交叉注意力，将encoder和decoder同时放进去
        x=self.dropout2(x)
        x=self.norm2(x+_x)
        _x=x
        x=self.ffn(x)
        x=self.dropout3(x)
        x=self.norm3(x+_x)
        return x


class Decoder(nn.Module):
    def __init__(self,dec_voc_size,max_len,d_model,ffn_hidden,n_head,n_layer,drop_prob, device):  # decoder的词汇表，encoder的词汇表
        super(Decoder,self).__init__()
        self.embedding=embedding.TransformerEmbedding(dec_voc_size, d_model, max_len, drop_prob, device)
        self.layers = nn.ModuleList(  # 比nn.Sequential更简单的写法，但是中间层不能复用
            [
                DecoderLayer(d_model, ffn_hidden, n_head, drop_prob)
                for _ in range(n_layer)  # 创建n_layer个EncoderLayer存在列表中，之后可以遍历取出
            ]
        )
        self.fc=nn.Linear(d_model,dec_voc_size)  # 出decoder后的线性层

    def forward(self, dec,enc, t_mask,s_mask=None):

        dec = self.embedding(enc)
        for layer in self.layers:
            dec = layer(dec,enc,t_mask, s_mask)# 整个Decoder已经封装好了
        dec=self.fc(dec)

        return dec


