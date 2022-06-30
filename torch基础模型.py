import torch
from torch import nn
from torch import optim
import numpy as np
from matplotlib import pyplot as plt

# 1. 定义数据
x = torch.rand([50, 1])
y = x * 3 + 0.8 + (torch.rand(x.shape) - 0.5)  #添加的噪点


# 2 .定义模型
class Lr(nn.Module):
    def __init__(self):
        super(Lr, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        out = self.linear(x)
        return out


# 2. 实例化模型，loss，和优化器
model = Lr()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)
# 3. 训练模型
for i in range(30000):
    out = model(x)  # 3.1 获取预测值
    loss = criterion(y, out)  # 3.2 计算损失
    optimizer.zero_grad()  # 3.3 梯度归零
    loss.backward()  # 3.4 计算梯度
    optimizer.step()  # 3.5 更新梯度
    if (i + 1) % 20 == 0:
        print('Epoch[{}/{}], loss: {:.6f}'.format(i, 30000, loss.data))

# print(model.parameters())
# 4. 模型评估
model.eval()  # 设置模型为评估模式，即预测模式
predict = model(x)
predict = predict.data.numpy()
plt.scatter(x.data.numpy(), y.data.numpy(), c="r")
plt.plot(x.data.numpy(), predict)
plt.show()
