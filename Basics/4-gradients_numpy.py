import numpy as np

# Linear regression => 4 Types
# type-1 => manual linear regression
# type-1 details :
# 1) Predicition: Manually
# 2) Gradients Computation : Manually
# 3) Loss Computation : Manually
# 4) Parameter Updates : Manually

# step-1 : creating data samples

# f = w * x => f = 3 * x (w = 3)
# Y == f
X = np.array([1,2,3,4], dtype = np.float32)
Y = np.array([3,6,9,12], dtype = np.float32)
w = 0.00

# Step-2 : define user defined functions

# model predicition
def forward(x): # y_pred = w*x (forward pass)
    return w*x

# loss (here, loss = = Mean Square Error(MSE))
def loss(y,y_pred):
    return ((y_pred - y)**2).mean()

# gradient
# MSE = 1/N ((w*x - y)^2) [here , 1/N is mean]
# let, J = MSE
# dJ/dw = 1/N * (2) * (w*x - y) * d/dw(w*x - y)
# dJ/dw = 1/N * (2) * (w*x - y) * (x)
# dJ/dw = 1/N *(2x)*(w*x - y)
def gradient(x,y,y_pred):
    return np.dot(2*x,(y_pred - y)).mean()

print(f'Prediction before training : f(5) : {forward(5):.2f}')

# Step-3 : training

learning_rate = 0.01
no_iters = 10 # (or) 9

for epoch in range(no_iters):
    # predicition : forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y,y_pred)

    # gradient
    dw = gradient(X,Y,y_pred)

    # update weights
    w -= learning_rate * dw

    if(epoch % 1 == 0):
        print(f'ephons {epoch+1}: w = {dw:.2f}, loss = {l:.8f}')

print(f'Prediction after training : f(5) : {forward(5):.2f}')