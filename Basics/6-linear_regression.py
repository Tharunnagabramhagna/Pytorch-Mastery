import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets  # to get ready-made sample data
import matplotlib.pyplot as plt  # to plot the graph in output

# steps in pipeline:
# 1) Design model (input, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
#    - forward pass : compute prediction
#    - backward pass : gradients
#    - update weights

# step-1 : prepare data

X_numpy, Y_numpy = datasets.make_regression(
    n_samples=100, n_features=1, noise=20, random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(Y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)  # reshape to get good graph

# step-2 : pytorch model

n_samples, n_features = X.shape
input_size = n_features
output_size = 1  # This is always 1
model = nn.Linear(input_size, output_size)

# step-3 : Compute loss and optimizer

learning_rate = 0.01
no_epochs = 200

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# step-4 : Training the model

for epoch in range(no_epochs):
    #  model prediction : forward pass
    y_pred = model(X)

    # compute loss
    loss = criterion(y_pred, y)

    # gradient : backward pass
    loss.backward()

    # update weights
    optimizer.step()

    # zero gradient for each iteration
    optimizer.zero_grad()

    if (epoch + 1) % 100 == 0:
        print(f'epoch {epoch+1}, loss : {loss.item():.4f}')

# step-5 : plot graph

predicted = model(X).detach().numpy()  # prevent the gradient history
plt.plot(X_numpy, Y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()