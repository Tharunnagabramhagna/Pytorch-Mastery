import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets,models,transforms
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy

# step-1 : Do model transformation

# Same values of resnet18 model are placed here
mean = np.array([0.5,0.5,0.5])
std = np.array([0.25,0.25,0.25]) 

data_transforms = {'train':
                   transforms.Compose([
                       transforms.RandomResizedCrop(244), # 244 => industry-standard
                       transforms.RandomHorizontalFlip(), # shows data by fliping one by one
                       transforms.ToTensor(), # Convert data to tensor
                       transforms.Normalize(mean, std)
                   ]),
                   'val':
                   transforms.Compose([
                       transforms.Resize(256), # test with higher input (Final exam for model)
                       transforms.RandomResizedCrop(244),
                       transforms.ToTensor(),
                       transforms.Normalize(mean, std)
                   ])
}



# step-2 : Create dataset and dataloader 

data_dir = './data/hymenoptera_data'

image_dataset = {x: datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x])
                                        for x in ['train','val']}

data_loader = {x: DataLoader(image_dataset[x],batch_size=4,shuffle=True) for x in ['train','val']}
                   
dataset_sizes = {x : len(image_dataset[x]) for x in ['train','val']}

# names of the images
class_names = image_dataset['train'].classes
print("Class names : ",class_names)

# batch of training data
inputs,classes = next(iter(data_loader['train']))
# print("Sample inputs and classes :\n",inputs,classes)

# step-3 : Create device and define imshow function

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def imshow(inp, title):
    inp = inp.numpy().transpose((1,2,0)) # Tensor --> Numpy (Here, inp => input)
    inp = std * inp + mean # Un-Normalize
    inp = np.clip(inp, 0, 1) # inputs lies in range of [0,1]
    plt.imshow(inp)
    plt.title(title)
    plt.show()

out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])

# step-4 : Training model function

def train_model(model,optimizer,criterion,scheduler,num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch,num_epochs-1))
        print('-'*10)

        # Each epoch has training and vaildation phase