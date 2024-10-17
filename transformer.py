import torch
from torch import nn
import torch.functional as F
import math
import embedding
import attention
import decoder
import encoder
import layerorm

class Transformer(nn.Module):
    def __init__(self,src_pad_ix,  # 源索引
                 trg_pad_ix, # 目标索引
                 enc_voc_size,  # 词汇表大小
                 dec_voc_size,  # decoder词汇表大小
                 d_model,
                 n_head,
                 ffn_hidden,
                 max_len,
                 n_layer,
                 drop_prob,
                 device):

        super(Transformer,self).__init__()

        self.encoder=encoder.Encoder(device, enc_voc_size,max_len,d_model, ffn_hidden, n_head, n_layer, dropout=drop_prob)
        self.decoder=decoder.Decoder(dec_voc_size,max_len,d_model,ffn_hidden,n_head,n_layer,drop_prob, device)
        self.src_pad_ix=src_pad_ix
        self.trg_pad_ix=trg_pad_ix
        self.device=device

    def make_pad_mask(self,q,k,pad_idx_q,pad_idx_k):
        len_q,len_k=q.size(1),k.size(1)
        q=q.ne(pad_idx_q).unsqueeze(1).unsqueeze(3)
        q=q.repeat(1,1,1,len_k)
        k = k.ne(pad_idx_k).unsqueeze(1).unsqueeze(2)
        k = k.repeat(1, 1, len_q,1)
        mask=q&k
        return mask

    def make_casual_mask(self,q,k):
        mask=torch.trill(torch.ones(len(q),len(k))).type(torch.BoolTensor).to(self.device)
        return mask

    def forwad(self,src,trg):
        src_mask=self.make_pad_mask(src,src,self.src_pad_ix,self.trg_pad_ix)
        trg_mask=self.make_pad_mask(trg,trg,self.trg_pad_ix,self.trg_pad_ix)*self.make_casual_mask(trg,trg)
        enc=self.encoder(src,src_mask)
        out=self.decoder(trg,enc,trg_mask,src_mask)
        return out




