import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

# E.g-1 : LambdaLR

lr = 0.1
model = nn.Linear(10, 1)  # (input_size, output_size)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
def lambda1(epoch): return epoch / 10


scheduler = lr_scheduler.LambdaLR(
    optimizer, lambda1)  # => lr * lambda1 (everytime)

print("Example-1")
print(optimizer.state_dict())

for epoch in range(7):
    # loss.backward() => forward pass (part of structure)
    optimizer.step()
    # optimizer.zero_grad() => validation step (part of structure)
    scheduler.step()
    print(
        f"Learning Rate in each epoch : {(optimizer.state_dict()['param_groups'][0]['lr']):.2f}")

# E.g-2 : MultiplicativeLR

lr = 0.1
model = nn.Linear(20, 1)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
def lambda2(epoch): return 0.25


scheduler = lr_scheduler.MultiplicativeLR(optimizer, lambda2)

print("\nExample-2")
print(optimizer.state_dict())

for epoch in range(5):
    optimizer.step()
    scheduler.step()
    print(
        f"Learning rate for each iter : {(optimizer.state_dict()['param_groups'][0]['lr']):.4f}")

# E.g-3 : StepLR

lr = 0.05
model = nn.Linear(10, 1)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

print("\nExample-3")
print(optimizer.state_dict())

for epoch in range(100):
    optimizer.step()
    scheduler.step()
    # lr decreases for every 30 steps
    if (epoch + 1) % 30 == 0:
        print(
            f"Learning rate : {(optimizer.state_dict()['param_groups'][0]['lr']):2f}")

# E.g-4 : MultiStepLR

lr = 0.06
model = nn.Linear(30, 1)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)

print("\nExample-4")
print(optimizer.state_dict())

for epoch in range(100):
    optimizer.step()
    scheduler.step()
    # Learning rate changes for a required range here that is [30, 80]
    if (epoch + 1) % 30 == 0:
        print(
            f"Learning rate in {epoch + 1} epochs : {(optimizer.state_dict()['param_groups'][0]['lr']):4f}")

# E.g-5 : ExponentialLR

lr = 0.1
model = nn.Linear(40, 1)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1)

print("\nExample-5")
print(optimizer.state_dict())

for epoch in range(100):
    optimizer.step()
    scheduler.step()
    if (epoch + 1) % 20 == 0:
        print(
            f"Learning rate in {epoch + 1} epochs : {(optimizer.state_dict()['param_groups'][0]['lr']):.7f}")
