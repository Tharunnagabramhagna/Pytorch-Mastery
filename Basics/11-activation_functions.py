import torch
import torch.nn as nn
import torch.nn.functional as F

# Types of activation functions #

# sigmoid => torch.sigmoid
# relu => torch.relu
# tanH => torch.tanh
# leaky relu => F.leaky_relu [not available in torch]

Tensor = torch.tensor([-1.0,1.0,2.0,3.0])

# 1)
output = torch.sigmoid(Tensor)
print("Sigmoid (torch version) :",output)
s = nn.Sigmoid()
print("Sigmoid (nn version) :",s(Tensor))

# 2)
output = torch.relu(Tensor)
print("Relu (torch version) :",output)
ru = nn.ReLU()
print("Relu (nn version) :",ru(Tensor))

# 3)
output = torch.tanh(Tensor)
print("tanH (torch version) :",output)
tan = nn.Tanh()
print("tanH (nn version) :",tan(Tensor))

# 4)
output = F.leaky_relu(Tensor)
print("leaky relu (nn functional version) :",output)
lr = nn.LeakyReLU()
print("Leaky relu (nn version) :",lr(Tensor))

# Activation functions implementation in Neural-net #

# option-1 => (Create a custom neural-net class with initialized in init)

class NeuralNet1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet1,self).__init__()
        self.lin1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(hidden_size,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        out = self.lin1(x)
        out = self.relu(out)
        out = self.lin2(out)
        out = self.sigmoid(out)
        return out

# option-2 => (Create a custom neural-net class with initialized in forward pass)

class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet2,self).__init__()
        self.lin1 = nn.Linear(input_size,hidden_size)
        self.lin2 = nn.Linear(hidden_size,1)

    def forward(self,x):
        out = self.lin1(x)
        out = torch.relu(out)
        out = self.lin2(out)
        out = torch.sigmoid(out)
        return out