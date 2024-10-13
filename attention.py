import torch
from torch import nn
import math
import torch.nn.functional as F

x=torch.rand(128,32,512)  # 例子
d_model=512
n_head=8

class MutiHeadAttention(nn.Module):
    def __init__(self,d_model,n_head): # 数据维度 头的数量
        super(MutiHeadAttention,self).__init__()
        self.n_head=n_head
        self.d_model=d_model
        # Q、K、V
        self.w_q=nn.Linear(d_model,d_model)   # 全链接层
        self.w_k=nn.Linear(d_model,d_model)
        self.w_v=nn.Linear(d_model,d_model)
        self.w_combine=nn.Linear(d_model,d_model)  # 合并线性层
        self.softmax=nn.Softmax(dim=1)   # softmax激活，使dim维度上的数值相加等于1，做归一化？

    # 处理Q、K、V的关系
    def forward(self,q,k,v,mask=None):  # mask可选
        batch,time,dimen=q.shape  # q是输入
        n_d=int(self.d_model/self.n_head)  # 模型的总维度被平均分，每个头处理n_d个
        q=self.w_q(q)  # qkv重塑为多头形状
        k=self.w_k(k)
        v=self.w_k(v)

        q=q.view(batch,time,self.n_head,n_d).permute(0,2,1,3) #q重塑为4个维度，并交换维度次序
        k=k.view(batch,time,self.n_head,n_d).permute(0,2,1,3)
        v=v.view(batch,time,self.n_head,n_d).permute(0,2,1,3)
        score=q@k.transpose(2,3)/math.sqrt(n_d)  # softmax再乘个V就是Attention   点乘是@
        if mask is not None: #
            score=score.masked_fill(mask==0,-10000)  # 掩码为0的地方填充为0.0001，确保softmax
        score=self.softmax(score)@v  # Attention
        score=score.permute(0,2,1,3).contiguous().view(batch,time,dimen)  # 重塑一下形状
        out=self.w_combine(score)
        return out

attention=MutiHeadAttention(d_model,n_head)
out=attention(x,x,x)
print(out)