import torch
import torch.nn as nn
import torch.nn.functional as F
import gensim
from tqdm import tqdm
import numpy as np
import math
import sys
sys.path.insert(0, '../')
from utils import get_pretrain_embedding

# 0.4543
class Config(object):
    def __init__(self):
        self.target = 'age'
        self.train_path = './datas/train.csv'
        # self.train_path = './datas/train_sample.csv'
        self.batch_size = 128
        self.learning_rate = 5e-4
        self.pad_size = 80
        self.num_epochs = 5
        self.save_path = './model_dict/transformer_age_pe.ckpt'
        self.require_improvement = 2000
        self.class_list = [str(i) for i in range(10)]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ad_id_embedding = (579847, 256)
        self.product_id_embedding = (11108, 256)
        self.advertiser_id_embedding = (24366, 256)
        self.industry_embedding = (302, 16)
        self.position_embedding = (self.pad_size, 16)
        self.dim_model = 800
        self.encode_params = {
            'dim_model': self.dim_model,
            'num_head': 8,
            'hidden': 512,
            'att_dropout': 0.1,
            'ffn_dropout': 0.2
        }
        self.pe_dropout = 0.1


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        ad_id, product_id, advertiser_id = self.load_pretrain_embedding(config)
        self.ad_id_embedding = nn.Embedding.from_pretrained(ad_id, freeze=False)
        self.product_id_embedding = nn.Embedding.from_pretrained(product_id, freeze=False)
        self.advertiser_id_embedding = nn.Embedding.from_pretrained(advertiser_id, freeze=False)
        self.industry_embedding = nn.Embedding(*config.industry_embedding, padding_idx=0)
        self.ad_id_embedding.padding_idx = 0
        self.product_id_embedding.padding_idx = 0
        self.advertiser_id_embedding.padding_idx = 0
        self.position_embedding = nn.Embedding(*config.position_embedding)
        self.embeddings = [
            self.ad_id_embedding,
            self.product_id_embedding, 
            self.advertiser_id_embedding, 
            self.industry_embedding,
            self.position_embedding
            ]
        # self.position_encoding = PositionalEncoding(config.dim_model, config.pe_dropout, config.device, config.pad_size)
        self.encoder = Encoder(**config.encode_params)
        self.fc = nn.Sequential(
            nn.Linear(config.dim_model, 128),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, 10)
        )

    def load_pretrain_embedding(self, config):

        # model_1 = gensim.models.Word2Vec.load('./word2vec/c_id.model')
        model_2 = gensim.models.Word2Vec.load('./word2vec/ad_id.model')
        model_3 = gensim.models.Word2Vec.load('./word2vec/p_id.model')
        model_4 = gensim.models.Word2Vec.load('./word2vec/a_id.model')

        # creative_id = get_pretrain_embedding(model_1, *config.creative_id_embedding)
        ad_id = get_pretrain_embedding(model_2, *config.ad_id_embedding)
        product_id = get_pretrain_embedding(model_3, *config.product_id_embedding)
        advertiser_id = get_pretrain_embedding(model_4, *config.advertiser_id_embedding)
        return ad_id, product_id, advertiser_id

    def forward(self, x):
        # x: ad_id, product_id, advertiser_id, industry, pos, mask
        mask = x[-1]
        x = torch.cat([self.embeddings[i](x[i]) for i in range(5)], dim=2)
        # x = self.position_encoding(x)
        x = self.encoder(x, mask)
        x = torch.mean(x, dim=1)     # 0.4543
        # x = x.view(x.size(0), -1)  # 0.4545
        out = self.fc(x)
        return out


class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, att_dropout=0.1, ffn_dropout=0.2):
        # dim_model 模型embedding维度
        super(Encoder, self).__init__()
        self.attention = MultiHeadAttention(dim_model, num_head, att_dropout)
        # self.feed_forward = PositionWiseFeedForward(dim_model, hidden, ffn_dropout)

    def forward(self, x, mask):
        out = self.attention(x, mask)
        # out = self.feed_forward(out)
        return out

def attention(query, key, value, mask=None):
    "Compute 'Scaled Dot Product Attention'"
    """
    @query, key, value: [B, num_head, seq_len, dim_head]
    @mask: [B, 1, 1, seq_len]
    """
    dim_head = query.size(-1)
    # todo 矩阵乘法,  [B, num_head, seq_len, dim_head] 矩阵乘 [B, num_head, dim_head, seq_len] = [B, num_head, seq_len, seq_len]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(dim_head)   # 缩放点积
    if mask is not None:
        # 广播机制 https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_model, num_head, att_dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert dim_model % num_head == 0, "dim_model cannot be devided by num_head !!!"
        self.dim_head = dim_model // num_head
        self.num_head = num_head
        self.fc_Q = nn.Linear(dim_model, dim_model)
        self.fc_K = nn.Linear(dim_model, dim_model)
        self.fc_V = nn.Linear(dim_model, dim_model)
        self.fc = nn.Linear(dim_model, dim_model)
        self.dropout = nn.Dropout(att_dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x, mask=None):
        # x [B, seq_len, dim_model], mask:[B, seq_len]
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1) # [B, 1, 1, seq_len] 利用广播机制
        batch_size = x.size(0)
        query = self.fc_Q(x).view(batch_size, -1, self.num_head, self.dim_head).transpose(1, 2)
        key = self.fc_K(x).view(batch_size, -1, self.num_head, self.dim_head).transpose(1, 2)
        value = self.fc_V(x).view(batch_size, -1, self.num_head, self.dim_head).transpose(1, 2)
        att_out = attention(query, key, value, mask)
        att_out = att_out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_head * self.dim_head)
        out = self.fc(att_out)
        out = x + self.dropout(out)
        out = self.layer_norm(out)
        return out


class PositionWiseFeedForward(nn.Module):
    def __init__(self, dim_model, hidden, ffn_dropout=0.3):
        super(PositionWiseFeedForward, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim_model, hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden, dim_model),
            nn.Dropout(ffn_dropout)
        )
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc(x)
        out = self.layer_norm(x + out)
        return out


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, device, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))  # [1, d_model/2]
        pe[:, 0::2] = torch.sin(position * div_term)  # 广播
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).to(device)   # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [B, seq_len, d_model]
        一个batch中的x应该都已经paddle好了
        """
        x = x + self.pe[:, :x.size(1)]  # 广播
        # todo 为什么这里用dropout
        return self.dropout(x)


        

