#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/1/29 14:20
# @Author  : Chocolate
# @Site    : 
# @File    : BERT_ShortText.py
# @Software: PyCharm

import torch
from torch import nn
from transformers import AutoModel

model = "schen/longformer-chinese-base-4096"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # 这个可以拿到预训练模型最后一层的结果
        self.longformer = AutoModel.from_pretrained(model)
        # 这里可以接分类层，输入768维，最后分为2个类别
        # 这里可以添加其他网络模型，提升效果
        # self.linear = nn.Linear(768, 2)
        self.linear = nn.Sequential(
            nn.Linear(768, 1024),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            nn.Linear(1024, 2)
        )

    def forward(self,input_ids,token_type_ids,attention_mask):
        # 取最后一层的第一个，因为我们希望拿到的是整句话的一个语义
        output = self.longformer(input_ids,token_type_ids,attention_mask).last_hidden_state[:,0] # 维度 [batch,seq,hidden_size]
        # 然后输送给分类的linear层
        output = self.linear(output)
        return output
