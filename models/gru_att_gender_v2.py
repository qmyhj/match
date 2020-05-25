import torch
import torch.nn as nn
import torch.nn.functional as F
import gensim
from tqdm import tqdm
import numpy as np
import sys
sys.path.insert(0, '../')
from utils import get_pretrain_embedding

# 0.9352
class Config(object):
    def __init__(self):
        self.target = 'gender'
        self.train_path = './datas/train.csv'
        # self.train_path = './datas/train_sample.csv'
        self.batch_size = 128
        self.learning_rate = 1e-3
        self.pad_size = 100
        self.num_epochs = 5
        self.hidden_size = 128
        self.dropout = 0.5
        self.num_layers = 1
        self.save_path = './model_dict/gru_att_gender_v2.ckpt'
        self.require_improvement = 1000
        self.class_list = ['0', '1']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.creative_id_embedding = (757880, 256)
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
        self.gru = nn.GRU(808, config.hidden_size, num_layers=config.num_layers, batch_first=True, bidirectional=True)
        self.weight_proj = nn.Parameter(torch.Tensor(config.hidden_size*2, 1))
        self.fc = nn.Sequential(
            nn.Linear(config.hidden_size*2, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),  # BN和relu的位置? 
            nn.Linear(128, 2)
        )
        self.init_weight()

    def init_weight(self):
        # 初始化
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
        # x: creative_id, ad_id, product_id, product_category, advertiser_id, industry, gender

        out = torch.cat([self.embeddings[i](x[i]) for i in range(5)], dim=2)
        out, _ = self.gru(out)
        # attention https://www.aclweb.org/anthology/P16-2034.pdf
        att = torch.matmul(torch.tanh(out), self.weight_proj)
        att_score = F.softmax(att, dim=1)
        out = torch.sum(out * att_score, dim=1)
        out = torch.tanh(out)
        out = self.fc(out)
        return out