import torch

# randn() => use for implementation of neutal networks for better training data
# range of randn() is very close to 0 so it gives better scope for well trained data

# gradient operation (Scalar) #
print("\n-- Gradient operation on Scalars --\n")
x = torch.randn(3, requires_grad=True) # gradient tensor
print("Value of x :",x) # Let x = [x1,x2,x3]
y = x + 2  # y = [x1+2,x2+2,x3+2]
print("Value of y :",y) 
z = y*y*2 # z = [2*(x1+2)^2 , 2*(x2+2)^2 , 2*(x3+2)^2]
print("Value of z :",z)
z = z.mean() # convert to scalar => 1/3 (2*(x1+2)^2 + 2*(x2+2)^2 + 2*(x3+2)^2)
print("Mean of z :",z)
z.backward() # dz/dx => # 1/3(2*2(xi+2)) = 4/3*(xi+2) [here xi is a general term]
print("Gradient of z with respect to x :",x.grad) # [4/3*(x1+2),4/3*(x2+2),4/3*(x2+2)]
print("Value of z :",z) # remains unchanges

# Gradient operation (Vector) #
print("\n-- Gradient operation on Vectors --\n")
a = torch.randn(4, requires_grad=True) # let, a = [a1,a2,a3,a4]
c = a*2 # c = [2*a1,2*a2,2*a3,2*a4]
print("Value of c :",c)
v = torch.tensor([0.1, 1.0, 0.001, 0.001], dtype= torch.float32)
c.backward(v) # d(c.v) / da => d / da ([2*ai*v]) => 2*ai [here, ai is a general term]
# This is a vector jucobian product
print("Gradient of vector c with respect to a :",a.grad) # [0.2, 2.0, 0.002, 0.002]

# Preventing gradient history #
print("\n-- Preventing gradient history --\n")

# Method-1) set requires_grad to False
print("Method-1")
m1 = torch.randn(3, requires_grad=True)
print("Value of m1 (with grad) :",m1)
m1.requires_grad_(False)
print("Value of m1 (without grad) :",m1)

# Method-2) use detach function
print("\nMethod-2")
m2 = torch.randn(3, requires_grad=True)
print("Value of m2 :",m2)
n = m2.detach()
print("Value of n :",n)

# Method-3) use with torch.no_grad():
print("\nMethod-3")
m3 = torch.randn(3, requires_grad=True)
print("Value of m3 : ",m3)
with torch.no_grad():
    y = m3 + 2
    print("Value of y :",y)

# Training Example #
print("\n-- Training Example --\n")
weights = torch.ones(5, requires_grad=True)
for ephons in range(3):
    model_output = (weights*4).sum()
    model_output.backward()
    print("Gradient of model_output with resepect to weights :\n",weights.grad)
    weights.grad.zero_() # make zero for every iteration to prevent from wrong data

# Optimization of data
# SGD => Stochastic Gradient Descent
# This is a sample program this does same work as the about code 
# optimizer = torch.optim.SGD(weights, lr=0.001)
# optimizer.step()
# optimizer.zero_grad()