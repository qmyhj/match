import torch
import torch.nn as nn
import torch.nn.functional as F


class Config(object):
    def __init__(self):
        self.target = 'gender'
        self.train_path = './train_group.csv'
        self.dev_path = './val_group.csv'
        # self.dev_path = './val_sample.csv'
        self.test_path = './test_group.csv'
        self.batch_size = 128
        self.learning_rate = 1e-4
        self.pad_size = 128
        self.num_epochs = 10
        self.save_path = './model_dict/naive_deep.gender.ckpt'
        self.require_improvement = 1000
        self.class_list = ['0', '1']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.product_id_embedding = (44314, 128)
        self.product_category_embedding = (18, 8)
        self.advertiser_id_embedding = (62966, 128)
        self.industry_embedding = (336, 16)
        self.fc_in_1 = 280
        self.fc_in_2 = 50
        self.num_classes = 2


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.product_id_embedding = nn.Embedding(*config.product_id_embedding)
        self.product_category_embedding = nn.Embedding(*config.product_category_embedding)
        self.advertiser_id_embedding = nn.Embedding(*config.advertiser_id_embedding)
        self.industry_embedding = nn.Embedding(*config.industry_embedding)
        self.embeddings = [self.product_id_embedding, self.product_category_embedding, self.advertiser_id_embedding, self.industry_embedding]
        self.fc1 = nn.Linear(config.fc_in_1, config.fc_in_2)
        self.fc2 = nn.Linear(config.fc_in_2, config.num_classes)

    def sum_pooling(self, i, data, mask):
        # import pdb; pdb.set_trace()
        data_embedding = self.embeddings[i](data)
        out = torch.sum(data_embedding * mask.unsqueeze(2).repeat(1, 1, data_embedding.size(2)).float(), dim=1)
        return out

    def forward(self, x):
        # x: product_id, product_category, advertiser_id, industry, mask
        out = torch.cat([self.sum_pooling(i, x[i], x[-1]) for i in range(4)], dim=1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out