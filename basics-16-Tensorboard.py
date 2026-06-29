import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

############## TENSORBOARD ########################
# imported to write something into tensorboard
from torch.utils.tensorboard import SummaryWriter
import sys  # Here we use this to exit an any line
import torch.nn.functional as F

# TensorBoard Setup :-
# 1) Open command prompt on windows / terminal on mac
# 2) type "pip install tensorboard"
# 3) then type "tensorboard --logdir=runs"
# 4) It shows a localhost link, click and open it in your browsers

# Writer Setup
writer = SummaryWriter("runs/mnist1")  # mention the path
###################################################

# step-1 : create a device variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# step-2 : Initialize hyper-parmeters

learning_rate = 0.001
num_epochs = 2
input_size = 784  # 1 batch => 28x28
hidden_size = 500
batch_size = 64  # image becomes 8x8 grid
num_classes = 10

# step-3 : Use MNIST dataset

# Dataset
train_dataset = torchvision.datasets.MNIST(
    root='./data', download=True, transform=transforms.ToTensor(), train=True)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, transform=transforms.ToTensor())

print(len(train_dataset))  # 60k
print(len(test_dataset))  # 10k

# Data-Loader
train_loader = DataLoader(dataset=train_dataset,
                          shuffle=True, batch_size=batch_size)
test_loader = DataLoader(dataset=test_dataset,
                         shuffle=False, batch_size=batch_size)

example = iter(train_loader)
example_data, example_labels = next(example)
print("Shape of data and labels :", example_data.shape, example_labels.shape)

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(example_data[i][0], cmap='grey')

############## TENSORBOARD ########################
# plt.show()
img_grid = torchvision.utils.make_grid(example_data)
writer.add_image('MNIST-images', img_grid)  # (title, parameter)
writer.close()
# sys.exit()
###################################################

# step-4 : Create a custom NeuralNet class


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.lin1(x)
        out = self.relu(out)
        out = self.lin2(out)
        # no sigmoid is applied at the end
        return out


model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# step-4 : Compute loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

############## TENSORBOARD ########################
writer.add_graph(model, example_data.reshape(-1, 28*28))
writer.close()
# sys.exit()
###################################################

# step-5 : Training Loop

no_iters = len(train_loader)
running_loss = 0.0
running_corrects = 0
for epochs in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        output = model(images)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ############## TENSORBOARD ########################
        running_loss += loss.item()
        _, preds = torch.max(output.data, 1)
        running_corrects += (preds == labels).sum().item()
        ###################################################

        if (i+1) % 100 == 0:
            print(
                f'epoch : {epochs+1}/{num_epochs}, iterations : {i+1}/{no_iters}, loss : {loss.item():.4f}')
            ############## TENSORBOARD ########################
            # structure of parameters => (label,actual_data,global_step)
            writer.add_scalar('training loss', running_loss /
                              100, epochs * no_iters + i)
            writer.add_scalar('accuracy',
                              running_corrects / 100, epochs * no_iters + i)
            # reset the values for every 100 iterations
            running_loss = 0.0
            running_corrects = 0
            ###################################################


# step-6 : Train the model

''' " Precision recall curve definition " :- it is an evaluation tool used to visualize
the trade-off between a model's exactness (precision) and completeness (recall) 
across various decision thresholds'''

label = []
pred = []

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        output = model(images)
        _, predicted = torch.max(output.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        ############## TENSORBOARD ########################
        class_pred = [F.softmax(out, dim=0) for out in output]
        pred.append(class_pred)
        label.append(predicted)

    pred = torch.cat([torch.stack(batch) for batch in pred])  # 2D Tensor
    label = torch.cat(label)  # 1D Tensor
    # print(pred.shape,label.shape) # (10000x1, 10000x10)
    ###################################################

    acc = 100 * n_correct / n_samples
    print(f'\nAccuracy of the network on the 10k test images : {acc:.3f}%')

    ############## TENSORBOARD ########################
    for i in range(10):
        label_i = label == i
        pred_i = pred[:, i]
        # tack => str(i) [title of each curve]
        writer.add_pr_curve(str(i), label_i, pred_i, global_step=0)
        writer.close()
    ###################################################
