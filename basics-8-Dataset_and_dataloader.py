import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader, Dataset
import math

# gradient computation etc. not efficient for whole data set
# -> divide dataset into small batches

'''
# training loop
for epoch in range(num_epochs):
    # loop over all batches
    for i in range(total_batches):
        batch_x, batch_y = ...
'''

'''
epoch = 1 forward and backward pass of ALL training samples

batch_size = number of training samples in one forward and backward pass

number of iterations = number of passes, each pass using [batch_size] number of samples

e.g. 100 samples, batch_size = 20 --> 100/20 = 5 iterations for 1 epoch

'''

# --> DataLoader can do the batch computation for us

# Implement a custom Dataset:
# inherit Dataset
# implement __init__ , __getitem__ , and __len__

# step-1 : Define custom dataset


class WineDataSet(Dataset):
    def __init__(self):
        xy = np.loadtxt('./data/wine/wine.csv', delimiter=',',
                        dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])  # [n_samples, n_features]
        self.y = torch.from_numpy(xy[:, [0]])  # [n_samples, 1]
        # shape => (samples, labels) => shape[0] == total_samples
        self.no_samples = xy.shape[0]

    def __getitem__(self, index):
        # dataset[0]
        return self.x[index], self.y[index]

    def __len__(self):
        return self.no_samples  # len(dataset)


dataset = WineDataSet()

# get first sample and unpack
features, labels = dataset[0]
# (or)
# first_data = dataset[0]
# features, labels = first_data
print(features, labels)

# step-2 : Initialize data-loader
# shuffle => train the data in better way
# num_workers => works with multiple sub-processors
# batch_size => it can be any number that's your choice
data_loader = DataLoader(dataset=dataset, batch_size=4,
                         shuffle=True, num_workers=0)
dataiter = iter(data_loader)
data = next(dataiter)  # move the batch pointer to it's next batch
# E.g :- Batch-1 (next) ---> Batch-2 => Batch-1 ---> Batch-2 (next)
features, labels = data
# (or) 
# features,labels = next(dataiter)
print(features, labels)  # size == 4

# step-3 : Training loop
num_epochs = 2
total_samples = len(dataset)
no_iters = math.ceil(total_samples/4)  # samples / batch_size
print(total_samples, no_iters)

# here: 178 samples, batch_size = 4, n_iters=178/4=44.5 -> 45 iterations
for epochs in range(num_epochs):
    for i, (input, labels) in enumerate(data_loader):
        if (i+1) % 5 == 0:
            print(
                f'epochs {epochs+1}/{num_epochs} , itertion : {i+1}/{no_iters} | Input : {input.shape} | Labels : {labels.shape}')

# some famous datasets are available in torchvision.datasets
# e.g. MNIST, Fashion-MNIST, CIFAR10, COCO

# step-4 : use MNIST to convert batch of ndarrays to binary data

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, 
                                           transform=torchvision.transforms.ToTensor(), download=True)

# batch_size can be anything your choice
train_dataloader = DataLoader(dataset=train_dataset, batch_size= 3, shuffle= True)

# print first sample
dataiter = iter(train_dataloader)
inputs, target = next(dataiter)
print(inputs,target)