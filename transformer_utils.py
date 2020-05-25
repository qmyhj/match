import torch
from tqdm import tqdm
import time
from datetime import timedelta
import random
import numpy as np
from models import deep_gender_v4

def build_dataset(config):

    def load_dataset(path, pad_size):
        datas = []
        with open(path, 'r') as f:
            for line in tqdm(f):
                line = line.strip()
                if not line:
                    continue
                seqences = line.split()
                age = int(seqences[0].split('_')[1])
                gender = int(seqences[0].split('_')[2])
                arr = np.array([list(map(int, x.split('_'))) for x in seqences[1:]], dtype=int)
                seq_len = arr.shape[1]
                if  arr.shape[1] < pad_size:
                    height, width = arr.shape[0], (pad_size - arr.shape[1])
                    arr = np.concatenate([arr, np.zeros((height, width), dtype=int)], axis=1)
                else:
                    arr = arr[:, :pad_size]
                seq_len = min(seq_len, pad_size)
                datas.append((
                    arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], seq_len, gender, age
                    ))
        return datas
    datas = load_dataset(config.train_path, config.pad_size)
    random.seed(1)
    random.shuffle(datas)
    all_users = len(datas)
    train_size = int(all_users * 0.9)
    return datas[:train_size], datas[train_size:], datas[train_size:]


class DatasetIterater(object):

    def __init__(self, dataset, config):
        self.dataset = dataset
        self.batch_size = config.batch_size
        self.index = 0
        self.dataset_len = len(dataset)
        self.device = config.device
        self.pad_size = config.pad_size
        self.config = config

    def _to_tensor(self, batch_data, pad_size):
        
        # embedding 特征
        # creative_id = torch.LongTensor([x[0] for x in batch_data]).to(self.device)
        ad_id = torch.LongTensor([x[1] for x in batch_data]).to(self.device)
        product_id = torch.LongTensor([x[2] for x in batch_data]).to(self.device)
        advertiser_id = torch.LongTensor([x[4] for x in batch_data]).to(self.device)
        industry = torch.LongTensor([x[5] for x in batch_data]).to(self.device)
        pos = torch.LongTensor([list(range(pad_size)) for _ in range(len(batch_data))]).to(self.device)
        mask = torch.LongTensor([[1] * x[6] + [0] * (pad_size - x[6]) for x in batch_data]).to(self.device)
        
        
        config = self.config
        # 单独训练还是联合训练
        if config.target == 'gender':
            gender = torch.LongTensor([x[-2] for x in batch_data]).to(self.device)
            return (ad_id, product_id, advertiser_id, industry, pos, mask), gender
        elif config.target == 'age':
            age = torch.LongTensor([x[-1] for x in batch_data]).to(self.device)
            return (ad_id, product_id, advertiser_id, industry, pos, mask), age

    def __next__(self):
        if self.index < self.dataset_len:
            batch_data = self.dataset[self.index: self.index + self.batch_size]
            self.index += self.batch_size
            return self._to_tensor(batch_data, self.config.pad_size)
        else:
            self.index = 0
            # 每个epoch随机shuffle数据集
            random.shuffle(self.dataset)
            raise StopIteration

    def __iter__(self):
        return self

    def __len__(self):
        n_batches = len(self.dataset) // self.batch_size
        if len(self.dataset) % self.batch_size == 0:
            return n_batches
        else:
            return n_batches + 1


# def build_iterator(dataset, config):
#     iter = DatasetIterater(dataset, config.batch_size, config.device)
#     return iter

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def get_pretrain_embedding(model, word_nums, embedding_dim):
    # index 0 for pad, 随机初始化， 1 for unk
    pad = np.zeros(embedding_dim)
    unk = np.random.normal(loc=0.0, scale=1e-3, size=embedding_dim)
    embeddings = [pad, unk]
    for i in tqdm(range(2, word_nums)):  # 从 1 开始
        embed = model.wv[str(i)]
        embeddings.append(embed)
    embeddings = np.array(embeddings)
    return torch.FloatTensor(embeddings)


if __name__ == "__main__":
    config = deep_gender_v4.Config()
    build_dataset(config)