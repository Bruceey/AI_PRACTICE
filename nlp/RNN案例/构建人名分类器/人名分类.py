"""
数据下载地址： https://download.pytorch.org/tutorial/data.zip
实现步骤：分为五步
（1）导入必备的工具包
（2）对data文件中的数据进行处理，满足训练要求
（3）构建RNN模型（包括传统RNN，LSTM以及GRU）
（4）构建训练函数并进行训练
（5）构建评估函数并进行预测
"""
from io import open
# 帮助使用正则表达式进行子目录的查询
import glob
import os
from pathlib import Path
# 用于获得常见字母及字符规范化
import string
import unicodedata

import random
import time
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 获取常用的字符数量
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)  # 26*2 + 5


# print("n_letter: ", n_letters)

def unicodeToAscii(s):
    """
    去除一些语言的重音标记
    :param s:
    :return:
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


data_path = Path(r'/Users/wangrui/Desktop/人工智能教程/阶段5-自然语言处理与NLP/05_深度学习与NLP/01_深度学习与NLPday08/02-代码/code_and_data/names/')


def readLines(filename):
    """从文件中读取每一行到内存中形成列表"""
    lines = []
    with open(filename, encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if line:
                line = unicodeToAscii(line)
                lines.append(line)
    return lines


# filename = data_path + "Chinese.txt"
# lines = readLines(filename)
# print(lines)

# 构建一个人名类别与具体人名对应关系的字典
category_lines = {}

# 构建所有类别的列表
all_categories = []

for file in data_path.glob('*.txt'):
    category = file.stem
    all_categories.append(category)
    lines = readLines(file)
    category_lines[category] = lines

n_categories = len(all_categories)


def lineToTensor(line):
    # 首先初始化一个全0的张量, 这个张量的形状是(len(line), 1, n_letters)
    # 代表人名中的每一个字母都用一个(1 * n_letters)张量来表示
    tensor = torch.zeros(len(line), 1, n_letters)
    # 遍历每个人名中的每个字符, 并搜索其对应的索引, 将该索引位置置1
    for li, letter in enumerate(line):
        tensor[li][0][all_letters.find(letter)] = 1

    return tensor

line = "Bai"
line_tensor = lineToTensor(line)
# print("line_tensor:", line_tensor)

# 三、构建RNN模型
# (1) 使用nn.RNN构建完成传统RNN使用类
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        """初始化函数中有4个參数，分别代表RNN输入最后一维尺寸，RNN的隐层最后一维尺寸，RNN层数"""
        super().__init__()
        # 将hidden_size与num_ layers传入其中
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 实例化预定义的nn.RNN，它的三个参数分别是input_size, hidden_size, num- layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        # 实例化nn.Linear，这个线性层用于将nn.RNN的输出维度转化为指定的输出维度
        self.linear = nn.Linear(hidden_size, output_size)
        # 实例化nn中预定的Softmax层，用于从输出层获得类别结果
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):
        """完成传统RNN中的主要逻辑，输入参数input代表输入张量，它的形状是1 x n_letters
        hidden代表RNN的隐层张量，它的形状是self .num_ layers x 1x self.hidden_size"""
        # 因为预定义的nn.RNN要求输入维度一定是三维张量，因此在这里使用unsqueeze (0)扩展一个维度
        input = input.unsqueeze(0)
        # 将input和hidden输入到传统RNN的实例化对象中，如果num_ layers=1， rr恒等于hn
        rr, hn = self.rnn(input, hidden)
        # 将从RNN中获得的结果通过线性 变换和softmax返回，同时返回hn作为后续RNN的输入
        return self.softmax(self.linear(rr)), hn

    def initHidden(self):
        """初始化隐层张量"""
        return torch.zeros(self.num_layers, 1, self.hidden_size)


# (2) 构建LSTM模型
# 使用nn.LSTM构建完成LSTM使用类
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        """初始化函数的參数与传统RNN相同"""
        super(LSTM, self).__init__()
        # 将hidden_size与num_layers传入其中
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 实例化预定义的nn.LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        # 实例化nn. Linear，这个线性层用于将nn.RNN的输出维度转化为指定的输出维度
        self.linear = nn.Linear(hidden_size, output_size)
        # 实例化nn中预定的Softmax层，用于从输出层获得类别结果
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden, c):
        """在主要逻辑函数中多出一个参数C，也就是LSTM中的细胞状态张量"""
        # 使用unsqueeze(0)扩展一个维度
        input = input.unsqueeze(0)
        # 将input, hidden以及初始化的c传入1stm中
        rr, (hn, c) = self.lstm(input, (hidden, c))
        # 最后返回处理后的rr， hn, c
        return self.softmax(self.linear(rr)), hn, c

    def initHiddenAndC(self):
        """初始化函数不仅初始化hidden还要初始化细胞状态c，它们形状相同"""
        c = hidden = torch.zeros(self.num_layers, 1, self.hidden_size)
        return hidden, c


# (3) 使用nn.GRU枸建完成传统RNN使用类
# GRU与传统RNN的外部形式相同，都是只传递隐层张量，因此只需要更改预定义层的名字
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 实例化预定义的nn.GRU，它的三个参数分别是input_size, hidden_size, num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):
        input = input.unsqueeze(0)
        rr, hn = self.gru(input, hidden)
        return self.softmax(self.linear(rr)), hn

    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size)


#参数
input_size = n_letters
n_hidden = 12
output_size = n_categories



# 从输出结果中获得指定类别函数：
def categoryFromoutput(output):
    """从输出结果中获得指定类别，参数为输出张量output"""
    # 从输出张量中返回最大的值和索引对象，我们这里主要需要这个索引
    top_n, top_i = output.topk(1)
    # top_i对象中取出素引的值
    category_i = top_i[0].item()
    # 根据素引值获得对应语言类别，返回语言类别和索引值
    return all_categories[category_i], category_i