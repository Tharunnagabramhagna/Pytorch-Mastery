import torch

# Linear regression => 4 Types
# type-2 => pytorch linear regression
# type-2 details :
# 1) Predicition: Manually
# 2) Gradients Computation : Autograd
# 3) Loss Computation : Manually
# 4) Parameter Updates : Manually

# step-1 : creating data samples

# f = w * x => f = 2 * x (w = 2)
# Y == f
X = torch.tensor([1,2,3,4], dtype = torch.float32)
Y = torch.tensor([2,4,6,8], dtype = torch.float32)
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# Step-2 : define user defined functions

# model predicition
def forward(x): # y_pred = w*x (forward pass)
    return w*x

# loss (here, loss = = Mean Square Error(MSE))
def loss(y,y_pred):
    return ((y_pred - y)**2).mean()

print(f'Prediction before training : f(5) : {forward(5):.2f}')

# Step-3 : training

learning_rate = 0.01
no_iters = 50 # (or) 47

for ephons in range(no_iters):
    # predicition : forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y,y_pred)

    # gradient : backward pass
    l.backward()

    # update weights
    with torch.no_grad():
        w -= learning_rate * w.grad

    # to manage correct data => make w zero everytime
    w.grad.zero_()

    if(ephons % 5 == 0):
        print(f'ephons {ephons+1}: w = {w:.2f}, loss = {l:.8f}')

print(f'Prediction after training : f(5) : {forward(5):.2f}')
