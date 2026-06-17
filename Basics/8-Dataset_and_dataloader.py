import torch
import torchvision
from torch.utils.data import dataset,dataloader
import numpy as np
import math

'''
epoch = 1 forward and backward pass of ALL training samples

batch_size = number of training samples in one forward and backward pass

number of iterations = number of passes, each pass using [batch_size] number of samples

e.g. 100 samples, batch_size = 20 --> 100/20 = 5 iterations for 1 epoch

'''

