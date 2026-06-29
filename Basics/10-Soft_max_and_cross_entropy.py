import torch
import torch.nn as nn
import numpy as np

# SOFTMAX CONCEPT #

# formula => e^yi / sum(all e^yi)
# SoftMax function => converts data to probabilities 
# Sum of those data is always eqaul to '1'
print("\n--- SOFTMAX ---\n")
def softMax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0) # axis == 0 => 1D

# Numpy Softmax
x = np.array([1.0,2.0,0.33])
print("Softmax of x (Numpy) : ",softMax(x))

# Tensor Softmax
y = torch.tensor([1.0,2.0,0.33])
output = torch.softmax(y,dim=0) # torch.softmax is a function in torch
print("SoftMax of y (Tensor) : ",output)

# CROSS-ENTROPY #
print("\n--- CROSS-ENTROPY ---\n")
# Numpy cross-entropy
# formula :- 1/N * (-yi * log(yi))
def cross_entropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss # / float(predicted.shape[0]) => neglitable

# Y must be one hot coded
# if class-0 : [1,0,0]
# if class-1 : [0,1,0]
# if class-2 : [0,0,1]
Y = np.array([1,0,0])

y_pred_good = np.array([0.7,0.2,0.1]) # good for class-0
y_pred_bad = np.array([0.1,0.3,0.6]) # good for class-2
l1 = cross_entropy(Y,y_pred_good)
l2 = cross_entropy(Y,y_pred_bad)
print(f'Good cross-entropy (Numpy): {l1:.4f}')
print(f'Bad cross-entropy (Numpy) : {l2:.4f}')

# Tensor Cross-entropy (actual shouldn't be hot coded)

print("\nExample-1")
loss = nn.CrossEntropyLoss()
# n_samples = 1
X = torch.tensor([0])
# n_samples x n_classes = 1x3
x_pred_good = torch.tensor([[2.2,1.3,0.1]])
x_pred_bad = torch.tensor([[0.2,1.3,1.5]])

l1 = loss(x_pred_good, X)
l2 = loss(x_pred_bad, X)

print(f'Loss in l1 (Tensor) : {l1:.4f}')
print(f'Loss in l2 (Tensor) : {l2:.4f}')

_, prediction1 = torch.max(x_pred_good,1) # torch.max => find max element index
_, prediction2 = torch.max(x_pred_bad,1)

print("Index of Entropy in l1 :",prediction1)
print("Index of Entropy in l2 :",prediction2)

print("\nExample-2")
# n_samples = 3
Z = torch.tensor([2,0,1])
# n_samples x n_classes = 3x3
z_pred_good = torch.tensor([[0.2,0.5,1.2],[3.0,1.0,2.3],[1.0,5.0,2.3]])
z_pred_bad = torch.tensor([[2.2,0.5,1.2],[0.4,1.0,2.3],[1.0,3.0,2.2]])

l1 = loss(z_pred_good,Z)
l2 = loss(z_pred_bad,Z)

print(f'Loss in l1 (Tensor) : {l1.item():4f}')
print(f'Loss in l2 (Tensor) : {l2.item():4f}')

_, predictions1 = torch.max(z_pred_good,1)
_, predictions2 = torch.max(z_pred_bad,1)
print(f'Entropy array order in l1 : {predictions1}')
print(f'Entropy array order in l2 : {predictions2}')

# NEUTRAL NETS #

# Binary Classification (Sigmoid) [demo]
# E.g :- is it a dog?
# step-1 : Create a custom class of neural-net

class NeuralNet1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet1,self).__init__()
        self.lin1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU() # activation function => f(x) = max(0,x)
        self.lin2 = nn.Linear(hidden_size, 1) # no. of classes (or) output_size == 1

    def forward(self,x):
        out = self.lin1(x)
        out = self.relu(out)
        out = self.lin2(out)
        out = torch.sigmoid(out)
        return out

# Step-2 : Initialize model and criterion

model = NeuralNet1(input_size=28*28, hidden_size=4)
loss = nn.BCELoss()

# Multi-class Problem (softmax) [demo]
# E.g :- which animal?
print("\n-- Neutal-Net with softmax --\n")

# step-1 : Create a custom netural-net class

class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet2,self).__init__()
        self.linear1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU() # activation function
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # no softmax is applied at the end
        return out

# step-2 : Initialize model and criterion

model = NeuralNet2(input_size=28*28, hidden_size=3, num_classes=2)
criterion = nn.CrossEntropyLoss()