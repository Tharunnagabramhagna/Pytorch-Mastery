import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets # to get sample data
from sklearn.preprocessing import StandardScaler # to scale the data
from sklearn.model_selection import train_test_split # to split the train and test data

# steps in pipeline:
# 1) Design model (input, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
#    - forward pass : compute prediction
#    - backward pass : gradients
#    - update weights

# Breast cancer testing has two types
# 1) benign - 1 (non-cancerous)
# 2) malignant - 0 (cancerous)

# NOTE :- As we have a basic model the accuracy is around ~ 93-99%

# Step-1 : Prepare and Scale data

bc = datasets.load_breast_cancer() # bc => breast cancer variable
X , y = bc.data, bc.target
# bc.data => the measurements taken from cell nuclei in breast cancer biopsies
# bc.target => it measures whether the tumor is malignant or benign

n_samples, n_features = X.shape # 569, 30
# 569 => no. of patients
# 30 => features (measurements of patients)

# training data = 80% ; testing data = 20% => 0.2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=1234)

# random state could be 1 (or) 1234 it doesn't matter

# Scale
sc = StandardScaler()
X_train = sc.fit_transform(X_train) # fit_transform => learn scaling rules + apply them.
X_test = sc.transform(X_test) # transform => apply the same rules, no new fitting.

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1) # (569 x 1)
y_test = y_test.view(y_test.shape[0], 1)

# Step-2 : Create Model

# Linear model y = wx + b , sigmoid at the end
class Model(nn.Module):
    def __init__(self, input_n_features):
        super(Model, self).__init__()
        self.lin = nn.Linear(input_n_features, 1) # output_size is always 1 

    def forward(self, x):
        return torch.sigmoid(self.lin(x)) 
    # sigmoid => squashes/reduce output to probability type of data - 0 (or) 1

model = Model(n_features)

# Step-3 : Compute loss and optimizer

learning_rate = 0.1
no_epochs = 100
criterion = nn.BCELoss() # BCE => binary cross-entropy
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Step-4 : Training loop

for epoch in range(no_epochs):
    # model prediction : forward pass
    y_pred = model(X_train)

    # compute loss
    loss = criterion(y_pred,y_train) 

    # gradients : backward pass
    loss.backward()

    # update gradients
    optimizer.step()

    # zero gradients for each iteration
    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        print(f'epochs {epoch+1} : loss {loss.item():.4f}')


# Step-5 : Measure the accuracy between train and test data

with torch.no_grad():
    y_predicted = model(X_test)
    y_pred_cls = y_predicted.round() # round() => round off probablities
    # eq() => compares predictions with actual labels.
    # sum() => act as total
    acc = y_pred_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'Accuracy : {acc.item():4f}')