import torch
import torch.nn as nn

# Linear regression => 4 Types
# type-4 => pytorch training pipelines
# type-4 details :
# 1) Predicition: pytorch model
# 2) Gradients Computation : Autograd
# 3) Loss Computation : Pytorch Loss
# 4) Parameter Updates : Pytorch Optimizer

# step-1 : Design model

# f = 2 * x
# Y = w * x (here, w = 2)

X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)  # test tensor

# dimensions of x are taken as paramters for the model
n_samples, n_features = X.shape
print(n_samples, n_features)

input_size = output_size = n_features
model = nn.Linear(input_size, output_size)

# custom model

# class LinearRegression(nn.Module):
#     def __init__(self,input_dim, output_dim):
#         super(LinearRegression, self).__init__()
#         self.lin = nn.Linear(input_dim, output_dim)

#     def forward(self, x):
#         return self.lin(x)

# model = LinearRegression(input_size,output_size)


print(f'Prediction before training f(5) : {model(X_test).item():.3f}')

# step-2 : Compute loss and optimizer
learning_rate = 0.1
no_iters = 250

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# step-3 : training model

for epoch in range(no_iters):
    # model predicition : forward pass
    y_pred = model(X)

    # loss
    l = loss(Y, y_pred)

    # gradient : backward pass
    l.backward()

    # update gradients
    optimizer.step()

    # zero gradients for each iteration
    optimizer.zero_grad()

    if epoch % 50 == 0:
        # b => optional bias
        [w, b] = model.parameters()
        print(f'epoch {epoch+1} : w = {w[0][0].item():.3f}, loss = {l:.8f}')

print(f'Prediction after training : {model(X_test).item():.3f}')
