import torch
import torch.nn as nn # nn => neutral networks

# Linear regression => 4 Types
# type-3 => pytorch training pipelines 
# type-3 details :
# 1) Prediction: Manually
# 2) Gradients Computation : Autograd
# 3) Loss Computation : Pytorch Loss
# 4) Parameter Updates : Pytorch Optimizer

# steps in pipeline:
# 1) Design model (input, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
#    - forward pass : compute prediction
#    - backward pass : gradients
#    - update weights

# step-1 : Design model

# f = 4 * x
# Y = w * x (here, w = 4)

X = torch.tensor([1,2,3,4], dtype=torch.float32)
Y = torch.tensor([5,10,15,20], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# model prediction

def forward(x):
    return w*x

print(f'Prediction before training f(4) : {forward(4):.3f}')

# step-2 : construct loss and optimizer
learning_rate = 0.01
no_iters = 90

loss = nn.MSELoss() # mean square error loss function
optimizer = torch.optim.SGD([w],lr=learning_rate)

# step-3 : training model

for epoch in range(no_iters):
    # prediction : forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y,y_pred)

    # gradient : backward pass
    l.backward()

    # update weights
    optimizer.step()

    # zero gradients for each iteration
    optimizer.zero_grad()

    if epoch % 10 == 0:
        print(f'epoch {epoch+1} : w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training f(4) : {forward(4):.3f}')