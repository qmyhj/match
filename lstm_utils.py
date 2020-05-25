import torch
from tqdm import tqdm
import time
from datetime import timedelta
import random
import numpy as np
from models.deep_age_v3 import Config

def build_dataset(config):

    def load_dataset(path, pad_size):
        datas = []
        with open(path, 'r') as f:
            for line in tqdm(f):
                line = line.strip()
                if not line:
                    continue
                 # 1_3_0	 1_1_1_1_1	2_0_2_2_2	4_0_2_4_0	10_5_1_10_8	0_0_0_0_0
                user = line.split('\t')[0].split('_')
                age = int(user[1])
                gender = int(user[2])
                seqences = line.split('\t')[1:]
                arr = np.array([list(map(int, seq.split('_'))) for seq in seqences])
                if arr.shape[0] < pad_size:
                    height, width = (pad_size - arr.shape[0]), arr.shape[1]
                    arr = np.concatenate([arr, np.zeros((height, width), dtype=int)], axis=0)
                else:
                    arr = arr[:pad_size, :]

                # history_len = len(seqences) # 用户实际点击序列长度
                # seq_len = min(history_len, pad_size)
    
                datas.append((arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4], gender, age))
        return datas
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)

    return train, dev, test
    # return test


class DatasetIterater(object):

    def __init__(self, dataset, config):
        self.dataset = dataset
        self.batch_size = config.batch_size
        self.index = 0
        self.dataset_len = len(dataset)
        self.device = config.device
        self.pad_size = config.pad_size
        self.config = config

    def _to_tensor(self, batch_data):

        # import pdb; pdb.set_trace()
        # embedding 特征
        creative_id = torch.LongTensor([x[0] for x in batch_data]).to(self.device)
        product_id = torch.LongTensor([x[1] for x in batch_data]).to(self.device)
        product_category = torch.LongTensor([x[2] for x in batch_data]).to(self.device)
        advertiser_id = torch.LongTensor([x[3] for x in batch_data]).to(self.device)
        industry = torch.LongTensor([x[4] for x in batch_data]).to(self.device)

        # pad_size = self.pad_size
        # mask = torch.LongTensor([[1]*(x[-3]) + [0]*(pad_size-x[-3]) for x in batch_data]).to(self.device)
        config = self.config
        # 单独训练还是联合训练
        if config.target == 'gender':
            gender = torch.LongTensor([x[-2] for x in batch_data]).to(self.device)
            return (creative_id, product_id, product_category, advertiser_id, industry), gender
        elif config.target == 'age':
            age = torch.LongTensor([x[-1] for x in batch_data]).to(self.device)
            return (creative_id, product_id, product_category, advertiser_id, industry), age
        else:
            gender = torch.LongTensor([x[-2] for x in batch_data]).to(self.device)
            age = torch.LongTensor([x[-1] for x in batch_data]).to(self.device)
            return (creative_id, product_id, product_category, advertiser_id, industry), gender, age

    def __next__(self):
        if self.index < self.dataset_len:
            batch_data = self.dataset[self.index: self.index + self.batch_size]
            self.index += self.batch_size
            return self._to_tensor(batch_data)
        else:
            self.index = 0
            # 每个epoch随机shuffle数据集
            # random.shuffle(self.dataset)
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

if __name__ == "__main__":
    config = Config()
    train, dev, test = build_dataset(config)
    train_iter = DatasetIterater(train, config)
    for train_data, train_label in train_iter:
        import pdb
        pdb.set_trace()
        pass
