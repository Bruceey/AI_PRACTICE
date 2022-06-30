import torch

rnn = torch.nn.RNN(5, 6, 1)

input = torch.randn(1,3,5)

h0 = torch.randn(1,3,6)

output, hn = rnn(input, h0)