import torch
import torch.nn as nn
import numpy as np

# SOFTMAX CONCEPT #

# formula => yi / sum(all yi)
# SoftMax function => converts data to probabilities 
# Sum of those data is always eqaul to '1'
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0) # axis == 0 => 1D

# Numpy Softmax
x = np.array([1,2,3,4])
print("Softmax of x : ",softmax(x))

# Tensor Softmax
y = torch.tensor([1.0,2.0,0.33])
output = torch.softmax(y,dim=0)
print("SoftMax of y : ",output)

# CROSS-ENTROPY #

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

y_pred_good = np.array([0.7,0.2,0.1])
y_pred_bad = np.array([0.1,0.3,0.6])
l1 = cross_entropy(Y,y_pred_good)
l2 = cross_entropy(Y,y_pred_bad)
print(f'Good cross-entropy : {l1:.4f}')
print(f'Bad cross-entropy : ",{l2:.4f}')