#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/12/29 11:19
# @Author  : Chocolate
# @Site    : 
# @File    : Longformer_BiGRU.py
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
        # self.gru = nn.GRU(input_size=768, hidden_size=512, batch_first=True, bidirectional=True, dropout=0.5)
        # UserWarning: dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1, but got dropout=0.5 and num_layers=1
        # 如果设置dropout不为0，那么层数要大于1.这里默认设置为1层，就没必要加dropout=0.5
        self.gru = nn.GRU(input_size=768, hidden_size=512, batch_first=True, bidirectional=True)
        # 这里可以接分类层，输入768维，最后分为2个类别
        # 这里可以添加其他网络模型，提升效果
        # 加多一个线性层
        """
        做分类的时候，全连接层加dropout层，防止过拟合，提升模型泛化能力。
        RNN一般在不同层循环结构体之间使用dropout, 而不在同一层的循环结构之间使用。
        dropout的直接作用是减少中间特征的数量，从而减少冗余，经过交叉验证，隐含节点dropout率等于0.5的时候效果最好， 
        训练时对于神经网络单元，按照一定的概率将其暂时从网络中丢弃
        （强迫一个神经单元，和随机挑选出来的其他神经单元共同工作，达到好的效果。消除减弱了神经元节点间的联合适应性，增强了泛化能力）。
        """
        self.linear = nn.Sequential(
            nn.Linear(1024, 512),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            nn.Linear(512, 2)
        )
        # self.linear = nn.Sequential(
        #     nn.Linear(1536, 512),
        #     torch.nn.Dropout(0.5),
        #     torch.nn.ReLU(),
        #     nn.Linear(512, 2)
        # )

    def forward(self, input_ids, token_type_ids, attention_mask):
        # 取最后一层的第一个，因为我们希望拿到的是整句话的一个语义
        output = self.longformer(input_ids, token_type_ids,
                                 attention_mask).last_hidden_state  # 维度 [batch,seq,hidden_size]
        output, h_n = self.gru(output)
        output = output[:, -1, :]  # [batch,1024]
        # 然后输送给分类的linear层
        output = self.linear(output)
        return output
