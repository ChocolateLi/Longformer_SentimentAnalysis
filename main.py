#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/12/26 21:42
# @Author  : Chocolate
# @Site    : 
# @File    : main.py
# @Software: PyCharm


import yaml
import torch
import pandas as pd
import dataset.my_dataset,dataset.shorttext_dataset
import models.Longformer_BiGRU
from trainer import trainer
from transformers import AdamW, get_scheduler
from torch import nn
from models import BERT_ShortText,Longformer_ShortText,Longformer,Longformer_BiGRU,Longformer_GRU,Longformer_BiLSTM,Longformer_LSTM,Longformer_RNN

"""
修改四处：
1.device (main.py，trainer.py)
2.model
3.csv
4.模型保存位置
"""

# 加载配置文件函数
def load_config(file_path):
    with open(file_path,'r',encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def main():
    # 训练参数
    # train_config = load_config(args.model_config_path)
    # learning_rate = train_config['learning rate']
    # 模型参数
    # model_config = ...

    # dataset 自定义输入模型的数据

    # model = get from model-config

    # dataloader:将model和dataset联系起来

    # trainer

    # 1.加载数据
    # 这里加载的是自己的数据集

    dataloaders = dataset.my_dataset.dataloader_dict()
    train_dataloader = dataloaders['train']
    valid_dataloader = dataloaders['valid']
    test_dataloader = dataloaders['test']


    # 这里加载短文本数据集
    """
    dataloaders = dataset.shorttext_dataset.dataloader_dict1()
    train_dataloader = dataloaders['train']
    valid_dataloader = dataloaders['valid']
    test_dataloader = dataloaders['test']
    """


    # 2.加载模型
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # model = models.BERT_ShortText.NeuralNetwork().to(device)
    # model = models.Longformer_ShortText.NeuralNetwork().to(device)
    model = models.Longformer.NeuralNetwork().to(device)
    # model = models.Longformer_BiGRU.NeuralNetwork().to(device)
    # model = models.Longformer_GRU.NeuralNetwork().to(device)
    # model = models.Longformer_BiLSTM.NeuralNetwork().to(device)
    # model = models.Longformer_LSTM.NeuralNetwork().to(device)
    # model = models.Longformer_RNN.NeuralNetwork().to(device)
    # 3.加载trainer

    # 4.加载训练参数
    learning_rate = 1e-5  # 定义学习率
    epoch_num = 10  # 轮次定义
    loss_fn = nn.CrossEntropyLoss()  # 损失函数，交叉熵
    optimizer = AdamW(model.parameters(), lr=learning_rate)  # Adamw一个常用的优化器
    lr_scheduler = get_scheduler(
        "linear",  # 使用线性的方式，慢慢往下降
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=epoch_num * len(train_dataloader),
    )

    # 5.开始训练并记录实验数据
    # df = pd.DataFrame(columns=['epoch','total_loss','mean_loss','test_loss','valid_acc','spend_time'])
    # csv_file_path = "./output/BERT_ShortText.csv"
    # csv_file_path = "./output/Longformer_ShortText.csv"
    csv_file_path = "./output/Longformer.csv"
    # csv_file_path = "./output/Longformer_BiGRU.csv"
    # csv_file_path = "./output/Longformer_GRU.csv"
    # csv_file_path = "./output/Longformer_BiLSTM.csv"
    # csv_file_path = "./output/Longformer_LSTM.csv"
    # csv_file_path = "./output/Longformer_RNN.csv"
    total_loss = 0.
    best_acc = 0.
    for t in range(epoch_num):
        print(f"Epoch {t + 1}/{epoch_num}\n-------------------------------")
        total_loss,mean_loss,train_time,train_correct = trainer.train_loop(train_dataloader, model, loss_fn, optimizer, lr_scheduler, t + 1, total_loss)
        valid_loss,valid_acc,valid_time = trainer.test_loop(valid_dataloader, model, mode='Valid')
        list = [total_loss,mean_loss,train_time,train_correct,valid_loss,valid_acc,valid_time]
        data = pd.DataFrame([list])
        data.to_csv(csv_file_path,mode='a',header=False,index=False)
        if valid_acc > best_acc:
            best_acc = valid_acc
            print('saving new weights...\n')
            # 保存模型
            # torch.save(model.state_dict(), f'./output/epoch_{t + 1}_valid_acc_{(100 * valid_acc):0.1f}_model_weights.bin')
            torch.save(model.state_dict(), f'./output1/epoch_{t + 1}_valid_acc_{(100 * valid_acc):0.1f}_model_weights.bin')
    print("Done!")

"""
运行方式：
python main.py --加参数。不加参数就是取默认值
"""

if __name__ == '__main__':
    # 1.定义解析器，解析参数等
    # parse = argparse.ArgumentParser()
    # 命令行输入的，或者默认的，添加参数
    # parse.add_argument('--model_config_path',type=str,default='./config/train_config.yaml')
    # ....

    # 解析参数，从命令行中取出来
    # args = parse.parse_args()
    # main(args)

    # 我觉得没必要搞那么复杂，用最简单的方式
    main()