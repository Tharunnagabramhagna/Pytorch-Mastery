import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# General steps :-
# 1) Use MNIST dataset
# 2) Define Multilayer Neural Net, activation function in a custom class
# 3) Compute Loss and optimizer
# 4) Training Loop (batch training)
# 5) Model evaluation
# 6) GPU Support (Store in GPU)

# Topic :- Neural Network on the MNIST handwritten digit dataset

# step-1 : create a device variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# step-2 : Initialize hyper-parmeters

learning_rate = 0.001
num_epochs = 2
input_size = 784 # 1 batch => 28x28
hidden_size = 500
batch_size = 100
num_classes = 10

# step-3 : Use MNIST dataset

# Dataset
train_dataset = torchvision.datasets.MNIST(root='./data',download=True,transform=transforms.ToTensor(),train=True)
test_dataset = torchvision.datasets.MNIST(root='./data',train=False,transform=transforms.ToTensor())

print(len(train_dataset)) # 60k
print(len(test_dataset)) # 10k

# Data-Loader
train_loader = DataLoader(dataset=train_dataset,shuffle=True,batch_size=batch_size)
test_loader = DataLoader(dataset=test_dataset,shuffle=False,batch_size=batch_size)

example = iter(train_loader)
example_data, example_labels = next(example)
print("Shape of data and labels :",example_data.shape, example_labels.shape)

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(example_data[i][0], cmap='grey')
plt.show()

# step-4 : Create a custom NeuralNet class

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet,self).__init__()
        self.lin1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(hidden_size,num_classes)

    def forward(self,x):
        out = self.lin1(x)
        out = self.relu(out)
        out = self.lin2(out)
        # no sigmoid is applied at the end
        return out

model = NeuralNet(input_size,hidden_size,num_classes).to(device)

# step-4 : Compute loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

# step-5 : Training Loop

no_iters = len(train_loader)

for epochs in range(num_epochs):
    for i, (images,labels) in enumerate(train_loader):
        # [100, 1 , 28, 28]  => [100, 784]
        images = images.reshape(-1,28*28).to(device)
        labels = labels.to(device)

        # forward pass
        output = model(images)

        # gradients
        loss = criterion(output, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'epoch : {epochs+1}/{num_epochs}, iterations : {i+1}/{no_iters}, loss : {loss.item():.4f}')

# step-6 : Train the model

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images,labels in test_loader:
        images = images.reshape(-1,28*28).to(device)
        labels = labels.to(device)
        output = model(images)
        # max returns (value, index)
        _, predicted = torch.max(output.data, 1)
        # Updating Batch size in n_samples
        # E.g :- label.shape = [100] ; label.size(0) = 100
        n_samples += labels.size(0)
        # Right predictions
        n_correct += (predicted == labels).sum().item()
    # accuracy formula => 100 x correct_predictions / total_predictions
    acc = 100 * n_correct / n_samples
    print(f'\nAcccuracy of the network on the 10k test images : {acc:.3f}%')

# test accuracy => How well does the model perform on new images it has never seen before? 
# Train Accuracy → Performance on seen data
# Test Accuracy  → Performance on unseen data   