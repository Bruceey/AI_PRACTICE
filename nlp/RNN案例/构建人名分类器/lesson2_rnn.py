from io import open
import glob
import os
import string
import unicodedata
import random
import time
import math
import torch
import torch.nn as nn       
import matplotlib.pyplot as plt

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)
# print("n_letters:", n_letters)

# 函数的作用是去掉一些语言中的重音标记
def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn' 
                   and c in all_letters)

s = "Ślusàrski"
a = unicodeToAscii(s)
# print(a)

data_path = "./data/names/"

def readLines(filename):
    # 打开指定的文件并读取所有的内容, 使用strip()去除掉两侧的空白符, 然后以'\n'为换行符进行切分
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

filename = data_path + "Chinese.txt"
result = readLines(filename)
# print(result[:20])


# 构建一个人名类别与具体人名对应关系的字典
category_lines = {}

# 构建所有类别的列表
all_categories = []

# 遍历所有的文件, 使用glob.glob中可以利用正则表达式的便利
for filename in glob.glob(data_path + "*.txt"):
    # 获取每个文件的文件名, 起始就是得到名字的类别
    category = os.path.splitext(os.path.basename(filename))[0]
    # 逐一将其装入所有类别的列表中
    all_categories.append(category)
    # 然后读取每个文件的内容, 形成名字的列表
    lines = readLines(filename)
    # 按照对应的类别, 将名字列表写入到category_lines字典中
    category_lines[category] = lines

n_categories = len(all_categories)
# print("n_categories:", n_categories)

# print(category_lines['Italian'][:10])

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


# x = torch.tensor([1, 2, 3, 4])
# print(x.shape)
# y = torch.unsqueeze(x, 0)
# print(y.shape)
# z = torch.unsqueeze(x, 1)
# print(z)
# print(z.shape)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        # input_size: 代表RNN输入的最后一个维度
        # hidden_size: 代表RNN隐藏层的最后一个维度
        # output_size: 代表RNN网络最后线性层的输出维度
        # num_layers: 代表RNN网络的层数
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # 实例化预定义的RNN，三个参数分别是input_size, hidden_size, num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        # 实例化全连接线性层, 作用是将RNN的输出维度转换成指定的输出维度
        self.linear = nn.Linear(hidden_size, output_size)
        # 实例化nn中预定义的Softmax层, 用于从输出层中获得类别的结果
        self.softmax = nn.LogSoftmax(dim=-1)
        
    def forward(self, input1, hidden):
        # input1: 代表人名分类器中的输入张量, 形状是1 * n_letters
        # hidden: 代表RNN的隐藏层张量, 形状是 self.num_layers * 1 * self.hidden_size
        # 注意一点: 输入到RNN中的张量要求是三维张量, 所以需要用unsqueeze()函数扩充维度
        input1 = input1.unsqueeze(0)
        # 将input1和hidden输入到RNN的实例化对象中, 如果num_layers=1, rr恒等于hn
        rr, hn = self.rnn(input1, hidden)
        # 将从RNN中获得的结果通过线性层的变换和softmax层的处理, 最终返回
        return self.softmax(self.linear(rr)), hn
    
    def initHidden(self):
        # 本函数的作用是用来初始化一个全零的隐藏层张量, 维度是3
        return torch.zeros(self.num_layers, 1, self.hidden_size)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        # input_size: 代表输入张量x中最后一个维度
        # hidden_size: 代表隐藏层张量的最后一个维度
        # output_size: 代表线性层最后的输出维度
        # num_layers: 代表LSTM网络的层数
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # 实例化LSTM对象
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        # 实例化线性层, 作用是将LSTM网络的输出维度转换成指定的输出维度
        self.linear = nn.Linear(hidden_size, output_size)
        # 实例化nn中预定义的Softmax层, 作用从输出层的张量中得到类别结果
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, input1, hidden, c):
        # 注意: LSTM网络的输入有3个张量,尤其不要忘记细胞状态c
        input1 = input1.unsqueeze(0)
        # 将3个参数输入到LSTM对象中
        rr, (hn, cn) = self.lstm(input1, (hidden, c))
        # 最后将3个张量结果全部返回, 同时rr要经过线性层和softmax的处理
        return self.softmax(self.linear(rr)), hn, cn
    
    def initHiddenAndC(self):
        # 注意: 对于LSTM来说, 初始化的时候要同时初始化hidden和细胞状态c
        # hidden和c的形状保持一致
        c = hidden = torch.zeros(self.num_layers, 1, self.hidden_size)
        return hidden, c


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        # input_size: 代表输入张量x最后一个维度
        # hidden_size: 代表隐藏层最后一个维度
        # output_size: 代表指定的线性层输出的维度
        # num_layers: 代表GRU网络的层数
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # 实例化GRU对象
        self.gru = nn.GRU(input_size, hidden_size, num_layers)
        # 实例化线性层的对象
        self.linear = nn.Linear(hidden_size, output_size)
        # 定义softmax对象, 作用是从输出张量中得到类别分类
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, input1, hidden):
        input1 = input1.unsqueeze(0)
        rr, hn = self.gru(input1, hidden)
        return self.softmax(self.linear(rr)), hn
    
    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size)


# 参数
input_size = n_letters

n_hidden = 128

output_size = n_categories

input1 = lineToTensor('B').squeeze(0)

hidden = c = torch.zeros(1, 1, n_hidden)

rnn = RNN(input_size, n_hidden, output_size)
lstm = LSTM(input_size, n_hidden, output_size)
gru = GRU(input_size, n_hidden, output_size)

rnn_output, next_hidden = rnn(input1, hidden)
# print('rnn:', rnn_output)
# print('rnn_shape:', rnn_output.shape)
# print('***********')

lstm_output, next_hidden1, c = lstm(input1, hidden, c)
# print('lstm:', lstm_output)
# print('lstm_shape:', lstm_output.shape)
# print('***********')

gru_output, next_hidden2 = gru(input1, hidden)
# print('gru:', gru_output)
# print('gru_shape:', gru_output.shape)


def categoryFromOutput(output):
    # output: 从输出结果中得到指定的类别
    # 需要调用topk()函数, 得到最大的值和索引, 作为我们的类别信息
    top_n, top_i = output.topk(1)
    # 从top_i中取出索引的值
    category_i = top_i[0].item()
    # 从前面已经构造号的all_categories中得到对应语言的类别,返回类别和索引
    return all_categories[category_i], category_i

# x = torch.arange(1, 6)
# print(x)
# res = torch.topk(x, 3)
# print(res)

# category, category_i = categoryFromOutput(gru_output)
# print('category:', category)
# print('category_i:', category_i)


def randomTrainingExample():
    # 该函数的作用用于随机产生训练数据
    # 第一步使用random.choice()方法从all_categories中随机选择一个类别
    category = random.choice(all_categories)
    # 第二步通过category_lines字典取出category类别对应的名字列表
    line = random.choice(category_lines[category])
    # 第三步将类别封装成tensor
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    # 将随机取到的名字通过函数lineToTensor()转换成onehot张量
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor


# for i in range(10):
#     category, line, category_tensor, line_tensor = randomTrainingExample()
#     print('category = ', category, ' / line = ', line, ' / category_tensor = ', category_tensor)
# print('line_tensor = ', line_tensor)


# a = torch.randn(4)
# print(a)
# b = torch.randn(4, 1)
# print(b)
# print('---------------')
# c = torch.add(a, b)
# print(c)
# d = torch.add(a, b, alpha=10)
# print(d)

# 定义损失函数, nn.NLLLoss()函数, 因为和RNN最后一层的nn.LogSoftmax()逻辑匹配
criterion = nn.NLLLoss()

# 设置学习率为0.005
learning_rate = 0.005

def trainRNN(category_tensor, line_tensor):
    # category_tensor: 代表训练数据的标签
    # line_tensor: 代表训练数据的特征
    # 第一步要初始化一个RNN隐藏层的张量
    hidden = rnn.initHidden()
    
    # 关键的一步: 将模型结构中的梯度归零
    rnn.zero_grad()
    
    # 循环遍历训练数据line_tensor中的每一个字符, 传入RNN中, 并且迭代更新hidden
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
        
    # 因为RNN的输出是三维张量, 为了满足category_tensor, 需要进行降维操作
    loss = criterion(output.squeeze(0), category_tensor)
    
    # 进行反向传播
    loss.backward()
    
    # 显示的更新模型中所有的参数
    for p in rnn.parameters():
        # 将参数的张量标识与参数的梯度进行乘法运算并乘以学习率, 结果加到参数上, 并进行覆盖更新
        p.data.add_(-learning_rate, p.grad.data)
    
    # 返回RNN最终的输出结果output, 和模型的损失loss
    return output, loss.item()


def trainLSTM(category_tensor, line_tensor):
    # 初始化隐藏层张量, 以及初始化细胞状态
    hidden, c = lstm.initHiddenAndC()
    # 先要将LSTM网络的梯度归零
    lstm.zero_grad()
    # 遍历所有的输入时间步的xi
    for i in range(line_tensor.size()[0]):
        # 注意LSTM每次输入包含3个张量
        output, hidden, c = lstm(line_tensor[i], hidden, c)
    
    # 将预测张量, 和目标标签张量输入损失函数中
    loss = criterion(output.squeeze(0), category_tensor)
    # 进行反向传播
    loss.backward()
    # 进行参数的显示更新
    for p in lstm.parameters():
        p.data.add_(-learning_rate, p.grad.data)
    return output, loss.item()


def trainGRU(category_tensor, line_tensor):
    # 注意GRU网络初始化的时候只需要初始化一个隐藏层的张量
    hidden = gru.initHidden()
    # 首先将GRU网络的梯度进行清零
    gru.zero_grad()
    # 遍历所有的输入时间步的xi
    for i in range(line_tensor.size()[0]):
        output, hidden = gru(line_tensor[i], hidden)
    
    # 将预测的张量值和真实的张量标签传入损失函数中
    loss = criterion(output.squeeze(0), category_tensor)
    # 进行反向传播
    loss.backward()
    # 进行参数的显示更新
    for p in gru.parameters():
        p.data.add_(-learning_rate, p.grad.data)
    return output, loss.item()


def timeSince(since):
    # 本函数的作用是打印每次训练的耗时, since是训练开始的时间
    # 第一步获取当前的时间
    now = time.time()
    # 第二步得到时间差
    s = now - since
    # 第三步计算得到分钟数
    m = math.floor(s / 60)
    # 第四步得到秒数
    s -= m * 60
    # 返回指定格式的耗时
    return '%dm %ds' % (m, s)

# since = time.time() - 10 * 60
# period = timeSince(since)
# print(period)


# 设置训练的迭代次数
n_iters = 1000
# 设置结果的打印间隔
print_every = 50
# 设置绘制损失曲线上的制图间隔
plot_every = 10

def train(train_type_fn):
    # train_type_fn代表选择哪种模型来训练函数, 比如选择trainRNN
    # 初始化存储每个制图间隔损失的列表
    all_losses = []
    # 获取训练开始的时间
    start = time.time()
    # 设置初始间隔的损失值等于0
    current_loss = 0
    # 迭代训练
    for iter in range(1, n_iters + 1):
        # 通过randomTrainingExample()函数随机获取一组训练数据和标签
        category, line, category_tensor, line_tensor = randomTrainingExample()
        # 将训练特征和标签张量传入训练函数中, 进行模型的训练
        output, loss = train_type_fn(category_tensor, line_tensor)
        # 累加损失值
        current_loss += loss
        
        # 如果到了迭代次数的打印间隔
        if iter % print_every == 0:
            # 取该迭代步的output通过函数categoryFromOutput()获取对应的类别和索引
            guess, guess_i = categoryFromOutput(output)
            # 判断和真实的类别标签进行比较, 如果相同则为True,如果不同则为False
            correct = 'True' if guess == category else 'False (%s)' % category
            # 打印若干信息
            print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter/n_iters*100, timeSince(start), loss, line, guess, correct))
        
        # 如果到了迭代次数的制图间隔
        if iter % plot_every == 0:
            # 将过去若干轮的平均损失值添加到all_losses列表中
            all_losses.append(current_loss / plot_every)
            # 将间隔损失值重置为0
            current_loss = 0
    
    # 返回对应的总损失列表, 并返回训练的耗时
    return all_losses, int(time.time() - start)


# 调用train函数, 分别传入RNN, LSTM, GRU的训练函数
# 返回的损失列表, 以及训练时间
# all_losses1, period1 = train(trainRNN)
# all_losses2, period2 = train(trainLSTM)
# all_losses3, period3 = train(trainGRU)

# 绘制损失对比曲线
# plt.figure(0)
# plt.plot(all_losses1, label="RNN")
# plt.plot(all_losses2, color="red", label="LSTM")
# plt.plot(all_losses3, color="orange", label="GRU")
# plt.legend(loc="upper left")

# 绘制训练耗时的柱状图
# plt.figure(1)
# x_data = ["RNN", "LSTM", "GRU"]
# y_data = [period1, period2, period3]
# plt.bar(range(len(x_data)), y_data, tick_label=x_data)


def evaluateRNN(line_tensor):
    # 评估函数, 仅有一个参数, line_tensor代表名字的张量标识
    # 初始化一个隐藏层的张量
    hidden = rnn.initHidden()
    # 将评估数据line_tensor中的每一个字符逐个传入RNN中
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    # 返回整个RNN的输出output
    return output.squeeze(0)


def evaluateLSTM(line_tensor):
    # 评估函数, 针对LSTM模型, 仅有一个参数, line_tensor代表名字的张量表示
    # 初始化一个隐藏层的张量, 同时再初始化一个细胞状态
    hidden, c = lstm.initHiddenAndC()
    # 将评估数据line_tensor中的每一个字符逐个传入LSTM中
    for i in range(line_tensor.size()[0]):
        output, hidden, c = lstm(line_tensor[i], hidden, c)
    # 返回整个LSTM的输出output, 同时完成降维的操作
    return output.squeeze(0)


def evaluateGRU(line_tensor):
    # 评估函数, 针对GRU模型, 仅有一个参数, line_tensor代表名字的张量表示
    # 初始化一个隐藏层的张量
    hidden = gru.initHidden()
    # 将评估数据line_tensor中的每一个字符逐个传入GRU中
    for i in range(line_tensor.size()[0]):
        output, hidden = gru(line_tensor[i], hidden)
    # 返回整个GRU的输出output, 同时完成降维的操作
    return output.squeeze(0)


# line = "Bai"
# line_tensor = lineToTensor(line)

# rnn_output = evaluateRNN(line_tensor)
# lstm_output = evaluateLSTM(line_tensor)
# gru_output = evaluateGRU(line_tensor)
# print('rnn_output:', rnn_output)
# print('lstm_output:', lstm_output)
# print('gru_output:', gru_output)


def predict(input_line, evaluate_fn, n_predictions=3):
    # input_line: 代表输入的字符串名字
    # evaluate_fn: 代表评估的模型函数, RNN, LSTM, GRU
    # n_predictions: 代表需要取得最有可能的n_predictions个结果
    # 首先将输入的名字打印出来
    print('\n> %s' % input_line)
    
    # 注意: 所有的预测函数都不能改变模型的参数
    with torch.no_grad():
        # 使用输入的人名转换成张量, 然后调用评估模型函数得到预测的结果
        output = evaluate_fn(lineToTensor(input_line))
        
        # 从预测的结果中取出top3个最大值及其索引
        topv, topi = output.topk(n_predictions, 1, True)
        # 初始化结果的列表
        predictions = []
        # 遍历3个最可能的结果
        for i in range(n_predictions):
            # 首先从topv中取出概率值
            value = topv[0][i].item()
            # 然后从topi中取出索引值
            category_index = topi[0][i].item()
            # 打印概率值及其对应的真实国家名称
            print('(%.2f) %s' % (value, all_categories[category_index]))
            # 将结果封装成列表格式, 添加到最终的结果列表中
            predictions.append([value, all_categories[category_index]])
        
        return predictions


# for evaluate_fn in [evaluateRNN, evaluateLSTM, evaluateGRU]:
#     print('-'*20)
#     predict('Dovesky', evaluate_fn)
#     predict('Jackson', evaluate_fn)
#     predict('Satoshi', evaluate_fn)















