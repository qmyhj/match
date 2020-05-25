import torch
import torch.nn as nn
import torch.nn.functional as F
import gensim
from tqdm import tqdm
import numpy as np
import sys
sys.path.insert(0, '../')
from utils import get_pretrain_embedding

# 加attention: 0.932
class Config(object):
    def __init__(self):
        self.target = 'gender'
        self.train_path = './datas/train.csv'
        # self.train_path = './datas/train_sample.csv'
        self.batch_size = 128
        self.learning_rate = 5e-4
        self.pad_size = 128
        self.num_epochs = 5
        self.save_path = './model_dict/deep_att_gender.ckpt'
        self.require_improvement = 2000
        self.class_list = ['0', '1']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ad_id_embedding = (579847, 256)
        self.product_id_embedding = (11108, 256)
        self.product_category_embedding = (20, 8)
        self.advertiser_id_embedding = (24366, 256)
        self.industry_embedding = (302, 32)


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        ad_id, product_id, advertiser_id = self.load_pretrain_embedding(config)
        self.ad_id_embedding = nn.Embedding.from_pretrained(ad_id, freeze=False)
        self.product_id_embedding = nn.Embedding.from_pretrained(product_id, freeze=False)
        self.product_category_embedding = nn.Embedding(*config.product_category_embedding, padding_idx=0)
        self.advertiser_id_embedding = nn.Embedding.from_pretrained(advertiser_id, freeze=False)
        self.industry_embedding = nn.Embedding(*config.industry_embedding, padding_idx=0)
        self.ad_id_embedding.padding_idx = 0
        self.product_id_embedding.padding_idx = 0
        self.advertiser_id_embedding.padding_idx = 0
        self.embeddings = [
            self.ad_id_embedding,
            self.product_id_embedding, 
            self.product_category_embedding, 
            self.advertiser_id_embedding, 
            self.industry_embedding
            ]
        self.weight_W = nn.Parameter(torch.Tensor(808, 256))
        self.weight_proj = nn.Parameter(torch.Tensor(256, 1))
        self.fc = nn.Sequential(
            nn.Linear(808, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)
        )
        self.init_weight()

    def init_weight(self):
        # 初始化
        nn.init.uniform_(self.weight_W, -1e-2, 1e-2)
        nn.init.uniform_(self.weight_proj, -1e-2, 1e-2)
        for name, w in self.fc.named_parameters():
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                nn.init.kaiming_normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)

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
        # x: ad_id, product_id, product_category, advertiser_id, industry, gender
        out = torch.cat([self.embeddings[i](x[i]) for i in range(5)], dim=2)
        # attention
        u = torch.tanh(torch.matmul(out, self.weight_W))
        att = torch.matmul(u, self.weight_proj)
        att_score = F.softmax(att, dim=1) # [B, seq_len, 1]
        out = torch.sum(out * att_score, dim=1)
        out = self.fc(out)
        return out