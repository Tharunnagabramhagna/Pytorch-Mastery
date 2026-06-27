import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import ALL_LETTERS, N_LETTERS
from utils import load_data, line_to_tensor, letter_to_tensor, random_training_example

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
        # batch_size == 1 (E.g : [1,57] in utils)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor, hidden_tensor), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

# step-2 : Create a rnn model


category_lines, all_categories = load_data()
n_categories = len(all_categories)

n_hidden = 128  # Custom input
rnn = RNN(N_LETTERS, n_hidden, n_categories)

# step-3 : Perform a DEMO

# one letter
input_tensor = letter_to_tensor('A')
hidden_tensor = rnn.init_hidden()

output, next_hidden = rnn(input_tensor, hidden_tensor)
print("Output tensor of one letter :\n", output)
print("hidden tensor of one letter :\n", next_hidden)
print(output.size())
print(next_hidden.size())

# whole sequence/word
input_tensor = line_to_tensor("Albert")
hidden_tensor = rnn.init_hidden()

output, next_hidden = rnn(input_tensor[0], hidden_tensor)
print(output.size())
print(next_hidden.size())


def category_from_output(output):
    # idx of max output for required letters
    category_idx = torch.argmax(output).item()
    return all_categories[category_idx]

# step-4 : Compute Loss and Optimizer


criterion = nn.NLLLoss()
learning_rate = 0.005
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

# step-5 : Define training function


def train(line_tensor, category_tensor):
    hidden = rnn.init_hidden()

    # forward pass
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    # gradients
    loss = criterion(output, category_tensor)

    # Backward pass + Update optimizer
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return output, loss.item()

# step-6 : Run the Model


curr_loss = 0
all_losses = []
plot_steps, print_steps = 1000, 5000
n_iters = 100000

for i in range(n_iters):
    category, line, category_tensor, line_tensor = random_training_example(
        category_lines, all_categories)

    output, loss = train(line_tensor, category_tensor)
    curr_loss += loss

    if (i+1) % plot_steps == 0:
        all_losses.append(curr_loss / plot_steps)
        curr_loss = 0

    if (i+1) % print_steps == 0:
        guess = category_from_output(output)
        correct = "CORRECT" if guess == category else f"WRONG ({category})"
        print(
            f'Steps: {i+1}, Percentage: {(i+1)/n_iters*100:.2f}, Loss : {loss:.4f} {line}, {guess} {correct}')

# Plotting the Graph

plt.figure()
plt.plot(all_losses)
plt.show()

# step-7 : Test the Model with custom input


def predict(input_line):
    print(f"\n> {input_line}")

    line_tensor = line_to_tensor(input_line) # Convert input to tensor

    hidden = rnn.init_hidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    guess = category_from_output(output)

    print("Guess by RNN model :", guess)


print("\n-- Welcome to the RNN name classification --\n")
while True:
    sentence = input("Enter your word : ")
    if sentence == "quit" or sentence == "no" or sentence == "NO":
        break
    predict(sentence)
