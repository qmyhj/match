import torch
import torch.nn as nn
import torch.nn.functional as F
import gensim
from tqdm import tqdm
import numpy as np
import sys
sys.path.insert(0, '../')
from utils import get_pretrain_embedding

# 0.4469
class Config(object):
    def __init__(self):
        self.target = 'age'
        self.train_path = './datas/train.csv'
        # self.train_path = './datas/train_sample.csv'
        self.batch_size = 128
        self.learning_rate = 1e-3
        self.pad_size = 100
        self.num_epochs = 5
        self.dropout = 0.3
        self.save_path = './model_dict/textCNN_age.ckpt'
        self.require_improvement = 2000
        self.class_list = [str(i) for i in range(10)]
        self.kernal_nums = 50
        self.kernal_sizes = [2, 3, 4, 5]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ad_id_embedding = (579847, 256)
        self.product_id_embedding = (11108, 256)
        self.product_category_embedding = (20, 8)
        self.advertiser_id_embedding = (24366, 256)
        self.industry_embedding = (302, 32)
        self.embed_dim = 808
        self.fc_in = self.kernal_nums * len(self.kernal_sizes)
        self.num_classes = 10

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
        self.convs = nn.ModuleList([nn.Conv2d(1, config.kernal_nums, (K, config.embed_dim)) for K in config.kernal_sizes])
        self.fc = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.fc_in, config.num_classes)
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
        
    def conv_and_pool(self, x, conv):
        # x [B, kernal_nums, kernal_size, embed_dim]
        x = F.relu(conv(x)).squeeze(3)
        # x [B, kernal_nums, steps]
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x # [B, kernal_nums]

    def forward(self, x):
        # x: ad_id, product_id, product_category, advertiser_id, industry, gender
        x = torch.cat([self.embeddings[i](x[i]) for i in range(5)], dim=2)
        x = x.unsqueeze(1) #[B, 1, seq_len, emb]
        x = torch.cat([self.conv_and_pool(x, conv) for conv in self.convs], dim=1)
        out = self.fc(x)
        return out