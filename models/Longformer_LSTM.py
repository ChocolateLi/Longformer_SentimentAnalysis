#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/12/31 14:29
# @Author  : Chocolate
# @Site    : 
# @File    : Longformer_LSTM.py
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
        # 接一个BiGRU
        self.lstm = nn.LSTM(input_size=768,hidden_size=512,batch_first=True,dropout=0.5)
        # 这里可以接分类层，输入768维，最后分为2个类别
        # 这里可以添加其他网络模型，提升效果
        # 加多一个线性层
        self.linear = self.linear = nn.Sequential(
            nn.Linear(512,512),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            nn.Linear(512,2)
        )

    def forward(self,input_ids,token_type_ids,attention_mask):
        # 取最后一层的第一个，因为我们希望拿到的是整句话的一个语义
        output = self.longformer(input_ids,token_type_ids,attention_mask).last_hidden_state # 维度 [batch,seq,hidden_size]
        output,h_n = self.lstm(output)
        output = output[:,-1,:] # [batch,1024]
        # 然后输送给分类的linear层
        output = self.linear(output)
        return output
