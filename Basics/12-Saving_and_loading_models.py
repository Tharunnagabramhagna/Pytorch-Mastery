import torch
import torch.nn as nn

''' 3 DIFFERENT METHODS TO REMEMBER:
 - torch.save(arg, PATH) # can be model, tensor, or dictionary
 - torch.load(PATH)
 - torch.load_state_dict(arg)
'''

''' 2 DIFFERENT WAYS OF SAVING
# 1) lazy way: save whole model
torch.save(model, PATH)

# model class must be defined somewhere
model = torch.load(PATH)
model.eval()

# 2) recommended way: save only the state_dict
torch.save(model.state_dict(), PATH)

# model must be created again with parameters
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()
'''

class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model,self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


# Tyep-1 : Saving and loading the Entire model
print("Type-1\n")
model = Model(n_input_features=5)
print("Parameters before evaluation : ")
for param in model.parameters():
    print(param) # 5 weights + 1 bias
FILE = "model.pth" # pth => .pth => shortform for pytorch
torch.save(model,FILE)

model = torch.load(FILE, weights_only=False)
model.eval()
print("\nParameters after evaluation : ")
for param in model.parameters():
    print(param)


# Type-2 : Saving and loading using state_dict
print("\nType-2\n")
model = Model(n_input_features=7)
FILE1 = "Diff_model.pth"
torch.save(model.state_dict(),FILE1) # save the state_dict of model
print("Model parameters before evaluation : ")
for param in model.parameters():
    print(param)

# re-initiliaze the model
loaded_model = Model(n_input_features=7)
loaded_model.load_state_dict(torch.load(FILE1, weights_only=False))
loaded_model.eval()
print("\nModel parameters after evaluation : ")
for param in loaded_model.parameters():
    print(param)


# Example for the Usage of Type-2 : Load the checkpoint
print("\nExample for Type-2\n")
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

checkpoint =  {
    'num_epochs' : 90,
    'model' : model.state_dict(),
    'optimizer' : optimizer.state_dict()
}
FILE2 = "checkpoint.pth"
torch.save(checkpoint, FILE2)

print("Optimizer before evaluation : ")
print(optimizer.state_dict()) # optimizer paramters

checkpoint = torch.load(FILE2, weights_only=False)
model = Model(n_input_features=7)
optimizer = torch.optim.SGD(model.parameters(), lr=0)
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
ecophs = checkpoint['num_epochs']

model.eval()

print("Optimizer after evaluation : ")
print(optimizer.state_dict())

# Remember that you must call model.eval() to set dropout and batch normalization layers 
# to evaluation mode before running inference. Failing to do this will yield 
# inconsistent inference results. If you wish to resuming training, 
# call model.train() to ensure these layers are in training mode.

""" SAVING ON GPU/CPU 

# 1) Save on GPU, Load on CPU
device = torch.device("cuda")
model.to(device)
torch.save(model.state_dict(), PATH)

device = torch.device('cpu')
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location=device))

# 2) Save on GPU, Load on GPU
device = torch.device("cuda")
model.to(device)
torch.save(model.state_dict(), PATH)

model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.to(device)

# Note: Be sure to use the .to(torch.device('cuda')) function 
# on all model inputs, too!

# 3) Save on CPU, Load on GPU
torch.save(model.state_dict(), PATH)

device = torch.device("cuda")
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location="cuda:0"))  # Choose whatever GPU device
number you want model.to(device)

# This loads the model to a given GPU device. 
# Next, be sure to call model.to(torch.device('cuda')) to convert the model's parameter 
tensors to CUDA tensors.
"""