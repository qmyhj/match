import torch
import torch.nn as nn
import torch.nn.functional as F
import gensim
from tqdm import tqdm
import numpy as np

# 0.9218
class Config(object):
    def __init__(self):
        self.target = 'gender'
        self.train_path = './datas/train.csv'
        # self.train_path = './datas/train_sample.csv'
        self.batch_size = 128
        self.learning_rate = 5e-4
        self.pad_size = 128
        self.num_epochs = 5
        self.dropout = 0.3
        self.save_path = './model_dict/fasttext_gender.ckpt'
        self.require_improvement = 2000
        self.class_list = ['0', '1']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ad_id_embedding = (579847, 256)
        self.product_id_embedding = (11108, 256)
        self.product_category_embedding = (20, 8)
        self.advertiser_id_embedding = (24366, 256)
        self.industry_embedding = (302, 32)
        self.fc_in = 808
        self.num_classes = 2

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.ad_id_embedding = nn.Embedding(*config.ad_id_embedding, padding_idx=0)
        self.product_id_embedding = nn.Embedding(*config.product_id_embedding, padding_idx=0)
        self.product_category_embedding = nn.Embedding(*config.product_category_embedding, padding_idx=0)
        self.advertiser_id_embedding = nn.Embedding(*config.advertiser_id_embedding, padding_idx=0)
        self.industry_embedding = nn.Embedding(*config.industry_embedding, padding_idx=0)
        self.embeddings = [
            self.ad_id_embedding,
            self.product_id_embedding, 
            self.product_category_embedding, 
            self.advertiser_id_embedding, 
            self.industry_embedding
            ]
        self.fc = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.fc_in, config.num_classes)
        )

    def forward(self, x):
        # x: ad_id, product_id, product_category, advertiser_id, industry, gender
        out = torch.cat([self.embeddings[i](x[i]) for i in range(5)], dim=2)
        out = torch.mean(out, dim=1)
        out = self.fc(out)
        return out