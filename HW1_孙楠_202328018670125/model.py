# -*- coding: utf-8 -*-

import sys
print(sys.executable)
import os 
import gzip
import struct 
import math
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader


# 保存数据的目录
def cat_data_dir(data_dir = './MNIST_data'):
    os.makedirs(data_dir, exist_ok=True)
    print("\nFiles in directory './MNIST_data':")
    for file_name in os.listdir(data_dir):
        print(file_name)


def main():
    # 数据集参数
    cat_data_dir()
    # input_dim = 10   # 输入维度
    # output_dim = 2   # 输出维度（类别数）
    # batch_size = 32  # 批处理大小

    # 加载数据
    # train_dataset = create_dataset(1000, input_dim, output_dim)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型、损失函数和优化器
    # model = SimpleNN(input_dim, output_dim).to(device)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型


    # 评估模型（使用训练数据作为示例，通常会使用单独的测试数据）
 

# 确保脚本是直接运行而不是作为模块导入时执行main函数
if __name__ == "__main__":
    main()