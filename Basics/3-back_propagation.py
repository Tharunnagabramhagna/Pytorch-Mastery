import torch

# Back propagation is done in three steps :
# 1) Forward Pass : Compute Loss
# 2) Compute Local gradients 
# 3) Backward Pass : Compute d Loss/ d Weights using Chain Rule

# Back propagation with an Example
# Let, x = 1, y = 2, w = 1
# first operation is x*w (we know that , y_hat = wx)
# second operation is y_hat - y --> s
# third operation is s^2 --> Loss

# Code
x = torch.tensor(1.0)
y = torch.tensor(2.0)
w = torch.tensor(1.0, requires_grad=True)

# Step-1 : Forward pass
y_hat = x*w
Loss = (y_hat - y) ** 2
print("Value of Loss :",Loss) # 1

# Step-2 : Compute gradients and step-3 : Backward Pass
Loss.backward()
print("Gradient of Loss with respect to weights : ",w.grad) # -2