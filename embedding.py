import torch
from torch import nn
import torch.nn.functional as F
import math
print(torch.cuda.is_available())

# token embedding 输入的词汇表索引->指定维度embedding
class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model): # 词汇表大小，维度
        super(TokenEmbedding,self).__init__(vocab_size,d_model, padding_idx=1) # 索引为1填充，embedding 初始为0

class PositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len, device): # 模型维度，最大长度，设备
        super(PositionEmbedding, self).__init__()
        self.encoding=torch.zeros(max_len,d_model,device=device)  # [max_len, d_model], 0
        self.encoding.requires_grad=False
        pos=torch.arange(0,max_len,device=device)  # 一维向量，start, end, step, device
        pos=pos.float().unsqueeze(dim=1) #(max_len)->(max_len,1)
        _2i=torch.arange(0,d_model,2,device=device).float()
        self.encoding[:,0::2]=torch.sin(pos/(10000**(_2i/d_model)))    # 偶数位置
        self.encoding[:,1::2]=torch.cos(pos/(10000**(_2i/d_model)))    # 奇数位置

    def forward(self, x):
            batch_size,seq_len=x.size() #  序列长度
            return self.encoding[:seq_len,:] # 返回前seq_len的encoding


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        super(TransformerEmbedding,self).__init__()
        self.tok_emb=TokenEmbedding(vocab_size,d_model)
        self.pos_emb=PositionEmbedding(d_model=d_model,max_len=max_len,device=device)
        self.drop_out=nn.Dropout(p=drop_prob) #  随机丢弃神经元，防止过拟合

    def forward(self,x):
        tok_emb=self.tok_emb(x)  # 初始化tok
        pos_emb=self.pos_emb(x)  # 初始化pos
        return self.drop_out(tok_emb+pos_emb)
