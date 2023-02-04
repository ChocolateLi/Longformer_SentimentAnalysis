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
from datasets import load_from_disk

# model = "bert-base-chinese"
model = "schen/longformer-chinese-base-4096"
tokenizer = AutoTokenizer.from_pretrained(model)
# 加载数据集
def load_data1(file_path):
    return load_from_disk(file_path)

# 数据预处理函数
def preprocess_function1(data):
    return tokenizer(data['text'], padding='max_length', max_length=512, truncation=True)

# Huggingface推荐德处理方法
def preprocess1 (dataset):
    return dataset.map(function=preprocess_function1,batched=True,remove_columns=['text']).rename_column("label", "labels")

# 数据集加载器
def dataloader1(dataset,batch_size):
    return torch.utils.data.DataLoader(dataset = dataset,batch_size=batch_size,collate_fn=DataCollatorWithPadding(tokenizer),shuffle=True,drop_last=True)

# 把数据集组成字典
def dataloader_dict1():
    train_batch_size = 2
    valid_batch_size = 1
    test_batch_size = 1
    file_path = './data/ChnSentiCorp'
    datasets = load_data1(file_path)
    train_dataset = datasets['train']
    valid_dataset = datasets['validation']
    test_dataset = datasets['test']
    encoded_train_dataset = preprocess1(train_dataset)
    encoded_valid_dataset = preprocess1(valid_dataset)
    encoded_test_dataset = preprocess1(test_dataset)
    train_dataloader = dataloader1(encoded_train_dataset,train_batch_size)
    valid_dataloader = dataloader1(encoded_valid_dataset,valid_batch_size)
    test_dataloader = dataloader1(encoded_test_dataset,test_batch_size)
    dataloaders = {
        "train":train_dataloader,
        "valid":valid_dataloader,
        "test":test_dataloader
    }
    return dataloaders