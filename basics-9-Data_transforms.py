'''
Transforms can be applied to PIL images, tensors, ndarrays, or custom data
during creation of the DataSet

complete list of built-in transforms: 
https://pytorch.org/docs/stable/torchvision/transforms.html

On Images
---------
CenterCrop, Grayscale, Pad, RandomAffine
RandomCrop, RandomHorizontalFlip, RandomRotation
Resize, Scale

On Tensors
----------
LinearTransformation, Normalize, RandomErasing

Conversion
----------
ToPILImage: from tensor or ndrarray
ToTensor : from numpy.ndarray or PILImage

Generic
-------
Use Lambda 

Custom
------
Write own class

Compose multiple Transforms
---------------------------
composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)])
'''

import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np

# step-1 : Create a custom dataset

class WineDataSet(Dataset):
    def __init__(self,transform=None):
        xy = np.loadtxt('./data/wine/wine.csv',delimiter=',',dtype=np.float32,skiprows=1)
        self.x = xy[:,1:]
        self.y = xy[:,[0]]
        self.n_samples = xy.shape[0]
        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.n_samples

# step-2 : create a custom transform tensor class

class ToTensor:
   # convert ndarrays to Tensors
   def __call__(self, sample):
    inputs, target = sample
    return torch.from_numpy(inputs), torch.from_numpy(target)

class MultiTransform:
    def  __init__(self,factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, target = sample
        target *= self.factor
        return inputs,target


# step-3 : print the transform details

dataset = WineDataSet()
print("Without transform")
features,labels = dataset[0]
print("Data type :",type(features), type(labels))
print("Values : ",features,labels)


print("\nWith Tensor transform")
dataset = WineDataSet(transform=ToTensor())
features,labels = dataset[0]
print("Data type :",type(features), type(labels))
print("Values : ",features,labels)      


print("\nWith Tensor and multiplication transform")
composed = torchvision.transforms.Compose([ToTensor(), MultiTransform(4)])
dataset = WineDataSet(transform=composed)
features,labels = dataset[0]
print("Data type :",type(features), type(labels))
print("Values : ",features,labels)      