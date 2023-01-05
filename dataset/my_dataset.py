#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/12/29 10:22
# @Author  : Chocolate
# @Site    : 
# @File    : my_dataset.py
# @Software: PyCharm

from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from transformers.data.data_collator import DataCollatorWithPadding

model = "schen/longformer-chinese-base-4096"
tokenizer = AutoTokenizer.from_pretrained(model)
# 加载数据集
def load_data(file_path):
    return load_dataset('csv', data_files=file_path, split='train')

# 数据预处理函数
def preprocess_function(data):
    return tokenizer(data['text'], padding='max_length', max_length=1500, truncation=True)

# Huggingface推荐德处理方法
def preprocess (dataset):
    return dataset.map(function=preprocess_function,batched=True,remove_columns=['text']).rename_column("label", "labels")

# 数据集加载器
def dataloader(dataset,batch_size):
    return torch.utils.data.DataLoader(dataset = dataset,batch_size=batch_size,collate_fn=DataCollatorWithPadding(tokenizer),shuffle=True,drop_last=True)

# 把数据集组成字典
def dataloader_dict():
    train_batch_size = 2
    valid_batch_size = 1
    test_batch_size = 1
    train_file_path = './data/MyData/train_dataset.csv'
    valid_file_path = './data/MyData/valid_dataset.csv'
    test_file_path = './data/MyData/test_dataset.csv'
    train_dataset = load_data(train_file_path)
    valid_dataset = load_data(valid_file_path)
    test_dataset = load_data(test_file_path)
    encoded_train_dataset = preprocess(train_dataset)
    encoded_valid_dataset = preprocess(valid_dataset)
    encoded_test_dataset = preprocess(test_dataset)
    train_dataloader = dataloader(encoded_train_dataset,train_batch_size)
    valid_dataloader = dataloader(encoded_valid_dataset,valid_batch_size)
    test_dataloader = dataloader(encoded_test_dataset,test_batch_size)
    dataloaders = {
        "train":train_dataloader,
        "valid":valid_dataloader,
        "test":test_dataloader
    }
    return dataloaders