import torch
from tqdm import tqdm
import time
from datetime import timedelta
import random
import numpy as np
from models.naive_deep import Config

def build_dataset(config):

    def load_dataset(path, pad_size):
        datas = []
        with open(path, 'r') as f:
            for line in tqdm(f):
                line = line.strip()
                if not line:
                    continue
                seqences = line.split('\t')[1:]
                arr = np.array([list(map(int, seq.split('_')[6:])) for seq in seqences])
                if arr.shape[0] < pad_size:
                    height, width = (pad_size - arr.shape[0]), arr.shape[1]
                    # import pdb; pdb.set_trace()
                    arr = np.concatenate([arr, np.zeros((height, width), dtype=int)], axis=0)
                else:
                    arr = arr[:pad_size, :]
                gender = int(seqences[0].split('_')[1])
                age = int(seqences[0].split('_')[0])
                seq_len = min(len(seqences), pad_size)
                datas.append((arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], seq_len, gender, age))

        return datas
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return train, dev, test
    # return dev, dev, dev


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
        product_id = torch.LongTensor([x[0] for x in batch_data]).to(self.device)
        product_category = torch.LongTensor([x[1] for x in batch_data]).to(self.device)
        advertiser_id = torch.LongTensor([x[2] for x in batch_data]).to(self.device)
        industry = torch.LongTensor([x[3] for x in batch_data]).to(self.device)

        pad_size = self.pad_size
        mask = torch.LongTensor([[1]*(x[4]) + [0]*(pad_size-x[4]) for x in batch_data]).to(self.device)
        config = self.config
        # 单独训练还是联合训练
        if config.target == 'gender':
            gender = torch.LongTensor([x[5] for x in batch_data]).to(self.device)
            return (product_id, product_category, advertiser_id, industry, mask), gender
        elif config.target == 'age':
            age = torch.LongTensor([x[6] for x in batch_data]).to(self.device)
            return (product_id, product_category, advertiser_id, industry, mask), age
        else:
            gender = torch.LongTensor([x[5] for x in batch_data]).to(self.device)
            age = torch.LongTensor([x[6] for x in batch_data]).to(self.device)
            return (product_id, product_category, advertiser_id, industry, mask), gender, age

    def __next__(self):
        if self.index < self.dataset_len:
            batch_data = self.dataset[self.index: self.index + self.batch_size]
            self.index += self.batch_size
            return self._to_tensor(batch_data)
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

if __name__ == "__main__":
    config = Config()
    dev = build_dataset(config)
    dev_iter = DatasetIterater(dev, config)
    for train_data, train_label in dev_iter:
        import pdb
        pdb.set_trace()
        pass
