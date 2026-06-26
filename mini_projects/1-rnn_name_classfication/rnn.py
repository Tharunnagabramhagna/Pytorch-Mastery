import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import ALL_LETTERS,N_LETTERS
from utils import load_data,line_to_tensor,letter_to_tensor,random_training_example

# step-1 : Create a RNN class (from scratch)

class RNN(nn.Module):
    # implement RNN from scratch rather than using nn.RNN
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        # i2h => input to hidden
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        # i2o => input to output
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        # This is a pair for NLLLoss() function
        self.softmax = nn.LogSoftmax(dim=1) # batch_size == 1 (E.g : [1,57] in utils)

    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor + hidden_tensor), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output) 
        return output, hidden

    def __init_hidden(self):
        return torch.zeros(1, self.hidden_size)

# step-2 : Create a rnn model

category_lines, all_categories = load_data()
n_categories = len(all_categories)

n_hidden = 128 # Custom input
rnn = RNN(N_LETTERS,n_hidden,n_categories)

# step-3 : Perform a DEMO

# step-4 : Compute Loss and Optimizer

# step-5 : Define training function

# step-6 : Run the Model

# step-7 : Test the Model with custom input