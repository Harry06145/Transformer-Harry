import torch
from torch import nn
import torch.nn.functional as F

x=torch.rand(128,32,512)  # 例子
d_model=512

class LayerNorm(nn.Module):
    def __init__(self,d_mode,eps=1e-12):  # eps为稳定性
        super(LayerNorm,self).__init__()
        self.gamma=nn.Parameter(torch.ones(d_mode))  # 创建一个可训练的torch参数  均值为0
        self.beta=nn.Parameter(torch.zeros(d_mode))  # 方差为1
        self.eps=eps

    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True) # x在最后一个取平均值，并且不压缩维度
        var=x.var(-1,unbiased=False,keepdim=True)  # 计算方差  unbiased=False指分母使用N而不是N-1
        out=(x-mean)/torch.sqrt(var+self.eps)
        out=self.gamma*out+self.beta

        return out

test=LayerNorm(d_model)
print(test(x))