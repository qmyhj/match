import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
# from utils import get_time_dif, build_dataset, DatasetIterater
from transformer_utils import get_time_dif, build_dataset, DatasetIterater
import importlib
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model")
args = parser.parse_args()

module = importlib.import_module('models.' + args.model)
config = module.Config()
model = module.Model(config)

import logging
file_name = 'logs/{}.log'.format(config.save_path.split('/')[-1])
logging.basicConfig(level=logging.INFO, filename=file_name, filemode='w', format='%(asctime)-15s %(message)s')

# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass

def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    model.train()
    for epoch in range(config.num_epochs):
        msg = 'Epoch [{}/{}]'.format(epoch + 1, config.num_epochs)
        logging.info(msg)
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predict = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predict)
                dev_acc, dev_auc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.3},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.3},  Val Acc: {4:>6.2%},  Val Auc: {5:>6.4},  Time: {6} {7}'
                msg = msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, dev_auc, time_dif, improve)
                logging.info(msg)
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                logging.info("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_auc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}, Test Auc: {2:>6.4}'
    logging.info(msg.format(test_loss, test_acc, test_auc))
    logging.info("Precision, Recall and F1-Score...")
    logging.info(test_report)
    logging.info("Confusion Matrix...")
    logging.info(test_confusion)
    time_dif = get_time_dif(start_time)
    logging.info("Time usage: %s", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    pred_prob_all = np.array([], dtype=float)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
            if config.target != 'gender':
                continue
            pred_prob = F.softmax(outputs, dim=1)[:, 1].data.cpu().numpy()
            pred_prob_all = np.append(pred_prob_all, pred_prob)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if config.target == 'gender':
        auc = metrics.roc_auc_score(labels_all, pred_prob_all)
    else:
        auc = 0.0
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, auc, loss_total / len(data_iter), report, confusion
    return acc, auc, loss_total / len(data_iter)


if __name__ == '__main__':
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    logging.info(model)
    init_network(model)
    model.to(config.device)
    start_time = time.time()
    logging.info('start loading data ...')
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = DatasetIterater(train_data, config)
    dev_iter = DatasetIterater(dev_data, config)
    test_iter = DatasetIterater(test_data, config)
    logging.info('load data success, Time usage: %s', get_time_dif(start_time))
    train(config, model, train_iter, dev_iter, test_iter)
    logging.info('ALL Time usage: %s', get_time_dif(start_time))