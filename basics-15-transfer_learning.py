import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy

# Creating a well trained model on image recognition #

# step-1 : Do data transformations

# Same values of resnet18 model are placed here
mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])

data_transforms = {'train':
                   transforms.Compose([
                       # 244 => industry-standard
                       transforms.RandomResizedCrop(244),
                       transforms.RandomHorizontalFlip(),  # shows data by fliping one by one
                       transforms.ToTensor(),  # Convert data to tensor
                       transforms.Normalize(mean, std)
                   ]),
                   'val':
                   transforms.Compose([
                       # test with higher input (Final exam for model)
                       transforms.Resize(256),
                       transforms.CenterCrop(244), # CenterCrop the image
                       transforms.ToTensor(),
                       transforms.Normalize(mean, std)
                   ])
                   }


# step-2 : Create dataset and dataloader

data_dir = './data/hymenoptera_data'

image_dataset = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                 for x in ['train', 'val']}

data_loader = {x: DataLoader(
    image_dataset[x], batch_size=4, shuffle=True) for x in ['train', 'val']}

dataset_sizes = {x: len(image_dataset[x]) for x in ['train', 'val']}

# names of the images
class_names = image_dataset['train'].classes
print("Class names : ", class_names)

# batch of training data
inputs, classes = next(iter(data_loader['train']))
# print("Sample inputs and classes :\n",inputs,classes)

# step-3 : Create device and define imshow function

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def imshow(inp, title):
    inp = inp.numpy().transpose((1, 2, 0))  # Tensor --> Numpy (Here, inp => input)
    inp = std * inp + mean  # Un-Normalize
    inp = np.clip(inp, 0, 1)  # inputs lies in range of [0,1]
    plt.imshow(inp)
    plt.title(title)
    plt.show()


out = torchvision.utils.make_grid(inputs)

imshow(out, [class_names[x] for x in classes])

# step-4 : Training model function


def train_model(model, optimizer, criterion, scheduler, num_epochs=25):
    # to count the models we call each time
    count = 1
    print(f'\nTransfer Learning model-{count}\n')
    count += 1

    # start a stopwatch to see the training time
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict()) # store the best weights of the model
    best_acc = 0 # best accuracy so far

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-'*10)

        # Each epoch has training and vaildation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in data_loader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward pass (we track grad in only training phase)
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize (only if in training phase)
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0) 
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model and change the best_acc if we have a better one
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print() # new line for each epoch

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# There are two types in transfer learning 

# Tyep-1 : Fine-tuning the ConvNet with Fully connected Layer

# step-1 : Create a well trained pre-exisiting model

model = models.resnet18(pretrained=True)
num_features = model.fc.in_features # number of features

# step-2 : Initialize the fully connected model

model.fc = nn.Linear(num_features,2) # 2 (or) len(claases_names)
model = model.to(device)

# step-3 : Compute loss and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.001)

# step-4 : Create scheduler and call the train_model function

# StepLR Decays the learning rate of each parameter group by gamma every step_size epochs
# Decay LR by a factor of 0.1 every 7 epochs
# Learning rate scheduling should be applied after optimizer’s update
# e.g., you should write your code this way:
# for epoch in range(100):
#     train(...)
#     validate(...)
#     scheduler.step()

step_lr_scheduler = lr_scheduler.StepLR(optimizer,step_size=7,gamma=0.1)

model = train_model(model,optimizer,criterion,step_lr_scheduler)

# Type-2 : ConvNet as a fixed feature Extracter (only the last layer is present)
# Remaining Layers are freezed

# step-1 : Create a well trained pre-exisiting model


model_conv = models.resnet18(pretrained=True)

for param in model_conv.parameters():
    param.requires_grad = False

# step-2 :  Initialize the fully connected model_conv
no_features = model_conv.fc.in_features
model_conv.fc = nn.Linear(no_features,2)
model_conv = model_conv.to(device)

# step-3 : Compute Loss,Optimizer and scheduler

criterion = nn.CrossEntropyLoss()

# momentum is initilazed as we are dealing with the last layer
optimizer = torch.optim.SGD(model_conv.parameters(),lr=0.01,momentum=0.9)
step_lr_scheduler = lr_scheduler.StepLR(optimizer,step_size=7,gamma=0.1)

# step-4 : Call the train_model function

model_conv = train_model(model_conv,optimizer,criterion,step_lr_scheduler)