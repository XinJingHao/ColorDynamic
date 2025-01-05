import torch.nn as nn
import torch
import math

class PositionalEncoding_NTD(nn.Module):
    """Batch First Positional Encoding. Note that emb_size must be even numbers"""
    def __init__(self, maxlen: int, emb_size: int):
        super(PositionalEncoding_NTD, self).__init__()

        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0) # (T,D) -> (1,T,D) 用于匹配 (N,T,D)

        self.register_buffer('pos_embedding', pos_embedding) # 同时保存;在同一个dvc;不参与训练

    def forward(self, token_embedding: torch.tensor):
        '''token_embedding的维度必须严格为 (N,T,D)'''
        return token_embedding + self.pos_embedding # (N,T,D) + (1,T,D) -> (N,T,D)



# pe = BF_PositionalEncoding(maxlen=5, emb_size=6)
# pos_embedding.shape = (1,T,D)
# tensor([[[ 0.0000,  1.0000,  0.0000,  1.0000,  0.0000,  1.0000],
#          [ 0.8415,  0.5403,  0.0464,  0.9989,  0.0022,  1.0000],
#          [ 0.9093, -0.4161,  0.0927,  0.9957,  0.0043,  1.0000],
#          [ 0.1411, -0.9900,  0.1388,  0.9903,  0.0065,  1.0000],
#          [-0.7568, -0.6536,  0.1846,  0.9828,  0.0086,  1.0000]]])
