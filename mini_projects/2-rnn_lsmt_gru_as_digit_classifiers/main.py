import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Topic : MNIST digits Classfication using RNN,LSTM and GRU

# step-1 : create a device variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# step-2 : Initialize hyper-parmeters

learning_rate = 0.001
num_epochs = 2
# input_size = 784 # 1 batch => (28x28)
hidden_size = 128
batch_size = 100
num_classes = 10

# input is divided as sequences here
input_size = 28
sequence_length = 28
num_layers = 2

# step-3 : Use MNIST dataset

# Dataset
train_dataset = torchvision.datasets.MNIST(
    root='./data', download=True, transform=transforms.ToTensor(), train=True)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, transform=transforms.ToTensor())

# Data-Loader
train_loader = DataLoader(dataset=train_dataset,
                          shuffle=True, batch_size=batch_size)
test_loader = DataLoader(dataset=test_dataset,
                         shuffle=False, batch_size=batch_size)

example = iter(train_loader)
example_data, example_labels = next(example)
# print("Shape of data and labels :", example_data.shape, example_labels.shape)

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(example_data[i][0], cmap='grey')
plt.show()

# step-4 : Create a custom RNN class


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers ,num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size,hidden_size,num_layers, batch_first=True)
        # (or)
        # self.gru = nn.GRU(input_size,hidden_size,num_layers, batch_first=True)
        # self.lstm = nn.LSTM(input_size,hidden_size,num_layers, batch_first=True)
        # Refer to https://docs.pytorch.org/docs/2.12/generated/torch.nn.RNN.html
        # from batch_first = True -> x needs to be: (batch_size, seq, input_size)
        self.fc = nn.Linear(hidden_size, num_classes)


    def forward(self, x):
        # Set initial hidden states (and cell states for LSTM)
        h0 = torch.zeros(self.num_layers, x.size(0) ,self.hidden_size).to(device)
        # c0 = torch.zeros(self.num_layers, x.size(0) ,self.hidden_size).to(device)
        # x : (n,28,28) ; h0 : (2,n,128)

        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # out : (n, 28, 128)
        out,_ = self.rnn(x,h0)
        # (or)
        # out,_ = self.gru(x,h0)
        # out,_ = self.lstm(x,(h0,c0))

        # out : (n, 128)
        out = out[:,-1,:]

        #  out : (n, 10)
        out = self.fc(out)

        return out


model = RNN(input_size, hidden_size, num_layers ,num_classes).to(device)

# step-4 : Compute loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# step-5 : Training Loop

no_iters = len(train_loader)

for epochs in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # [100, 1 , 28, 28]  => [100, 28, 28]
        images = images.reshape(-1, sequence_length, input_size).to(device)
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
            print(
                f'epoch : {epochs+1}/{num_epochs}, iterations : {i+1}/{no_iters}, loss : {loss.item():.4f}')

# step-6 : Train the model

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        output = model(images)
        _, predicted = torch.max(output.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
    acc = 100 * n_correct / n_samples
    print(f'\nAcccuracy of the network on the 10k test images : {acc:.3f}%')