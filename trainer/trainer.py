#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/12/29 11:52
# @Author  : Chocolate
# @Site    : 
# @File    : trainer.py
# @Software: PyCharm

from tqdm.auto import tqdm  # 显示它的进度条，会更好看点
import torch
from torch import nn
import time

# 参数解释
# dataloader ： 批量数据的loader
# model : 定义的模型
# loss_fn ： 定义的损失函数
# optimizer ：优化器
# lr_scheduler ： 学习率根据步数会下降，动态变化的。如果用一个固定的学习率，其实是没有这种随着迭代次数下降的效果好的
# epoch ：训练的轮次
# total_loss ：整体loss的情况

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
def train_loop(dataloader, model, loss_fn, optimizer, lr_scheduler, epoch, total_loss):
    start_time = time.time()
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_batch_num = (epoch - 1) * len(dataloader)

    # 获取训练集文本数据量
    size = len(dataloader.dataset)
    # 统计预测正确的个数
    correct = 0

    model.train()
    for batch, data in enumerate(dataloader, start=1):
        labels = data.labels.to(device)
        input_ids = data.input_ids.to(device)
        token_type_ids = data.token_type_ids.to(device)
        attention_mask = data.attention_mask.to(device)
        pred = model(input_ids, token_type_ids, attention_mask)
        loss = loss_fn(pred, labels)

        # 统计准确率
        correct += (pred.argmax(1) == labels).type(torch.float).sum().item()

        loss.backward()  # 向后传播
        optimizer.step()  # 算完梯度下降之后更改参数
        lr_scheduler.step()  # 对学习率进行调整
        optimizer.zero_grad()  # 把之前的梯度都清掉

        total_loss += loss.item()  # 统计一下整体的loss
        progress_bar.set_description(f'loss: {total_loss / (finish_batch_num + batch):>7f}')
        progress_bar.update(1)

    # 统计训练一轮花费的时间
    spend_time = time.time() - start_time
    correct /= size
    # total_loss/(finish_batch_num + batch) 统计每一轮的损失率
    return total_loss,total_loss/(finish_batch_num + batch),spend_time,correct

criterion = nn.CrossEntropyLoss() # 损失函数，交叉熵
def test_loop(dataloader, model, mode='Test'):
    start_time = time.time()
    assert mode in ['Valid', 'Test']
    size = len(dataloader.dataset)
    correct = 0
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for batch, data in enumerate(dataloader, start=1):
            labels = data.labels.to(device)
            input_ids = data.input_ids.to(device)
            token_type_ids = data.token_type_ids.to(device)
            attention_mask = data.attention_mask.to(device)
            pred = model(input_ids, token_type_ids, attention_mask)
            loss = criterion(pred,labels)
            test_loss += loss.item()
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()

        test_loss /= batch
        correct /= size
        spend_time = time.time() - start_time
        print(f"Average loss:{test_loss},{mode} Accuracy: {(100 * correct):>0.2f}%,spend time:{spend_time}\n")
        return test_loss,correct,spend_time