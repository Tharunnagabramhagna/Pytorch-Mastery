import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Topic :- CNN Image Classification using CIFAR10

# CNN => Convolutional Netural Network
# CIFAR10 is used as a dataset for CNN 
# CIFAR10 images are RGB images (RGB => Red Green Blue)
# Shape of image in CIFAR10 is [3,32,32]

# Formula to calulate image size transformation => ((W-K+2P)/S) + 1
# Here, W => width
#       K => kernel
#       P => padding
#       S => stride (How many pixels the filter jumps each time it moves.)

# E.g :- if Ouptut channels is 6 ==> 6 Feature maps
#     Filter 1 → vertical edges
#     Filter 2 → horizontal edges
#     Filter 3 → corners
#     Filter 4 → curves
#     Filter 5 → textures
#     Filter 6 → patterns

# Input Channels → fixed by data
# Output Channels → your choice
# Kernel Size → your choice (kerenel = 5 -> classic cnn styl )
# Stride → your choice
# Padding → your choice

# step-1 : create a device 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# step-2 : Initialize datset, data-loader and hyper parameters

batch_size = 4
num_epochs = 5
learning_rate = 0.01

# PILImages => tensor ; range = [0,1] => [-1,1]
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

train_dataset = torchvision.datasets.CIFAR10(root='./data',transform=transform,train=True,download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data',transform=transform,train=False,download=True)

train_loader = DataLoader(dataset=train_dataset,shuffle=True,batch_size=batch_size)
test_loader = DataLoader(dataset=test_dataset,shuffle=False,batch_size=batch_size)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5 # un-normailze
    npimg = img.numpy()
    # PyTorch     : [C,H,W]
    # Matplotlib  : [H,W,C]
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

example = iter(train_loader)
images, labels = next(example)
print(images, labels)

# show images
imshow(torchvision.utils.make_grid(images))

# step-3 : Create a custom ConvNet class

# (3, 6, 5) on (3,32,32) => ((32 - 5 + 0) / 1) + 1 = 27 + 1 = 28
# (2,2) on (6,28,28) => (6,14,14)
# (6, 16, 5) on (6, 14, 14) => ((14 - 5 + 0) / 1) + 1 = 9 + 1 = 10
# (2,2) on (16, 10, 10) => (16,5,5)
# fc => fully connected layer (A linear function that connects everything)
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # [input_size, output_size, kernel]
        self.pool = nn.MaxPool2d(2,2) # [2x2 matrix]
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5 , 120) # (400, 120) from (16,5,5 => 400)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10) # 10 => num_classes

    def forward(self, x):
        # n,3,32,32
        x = self.pool(F.relu(self.conv1(x))) # n,6,28,28 => n,6,14,14
        x = self.pool(F.relu(self.conv2(x))) # n,16,10,10 => n,16,5,5
        x = x.view(-1, 16*5*5) # n,400
        x = F.relu(self.fc1(x)) # n,120
        x = F.relu(self.fc2(x)) # n,84
        x = self.fc3(x) # n, 10
        return x

model = ConvNet().to(device)

# step-4 : Compute loss and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

# step-5 : Training Loop
no_iters = len(train_loader)
for epoch in range(num_epochs):
    for i, (images,labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # forward pass
        output = model(images)

        # gradient
        loss = criterion(output,labels)

        # backward pass and updation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'epochs : [{epoch + 1}/{num_epochs}], step : [{i+1}/{no_iters}], loss : {loss.item():.4f}')

print("Training Finished")

# Step-6 : Test the model

PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)] # [0,0,0,....10 times]
    n_class_samples = [0 for i in range(10)]
    for (images,labels) in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        # max returns (value, index)
        _ , predicted = torch.max(output.data, 1)
        n_samples += labels.size(0)
        n_correct += (labels == predicted).sum().item()

        for i in range(labels.size(0)): # (or) range(batch_size)
            label = labels[i].item()
            pred = predicted[i].item()
            if(label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100 * n_correct / n_samples
    print(f'Accuracy of the network : {acc} %')

    for i in range(10):
        acc = 100 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]} : {acc} %')
