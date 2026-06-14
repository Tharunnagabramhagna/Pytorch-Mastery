# PyTorch is an open-source Python library used to build, train, and deploy AI and 
# deep learning models

# PyTorch stores data in the form of tensors.

import torch as th # or use: import torch
import numpy as np

# Empty tensor
# Structure: torch.empty(size)
print("-- Empty tensor --\n")
x = th.empty(1) # 1D
print("Value of x :",x)

y = th.empty(2)
print("Value of y :",y)

a = th.empty(2,2) # 2D => 2 rows and 2 columns
print("Value of a :\n",a)

b = th.empty(2,3,2) # 3D => 2 layers,3 rows and 2 columns
print("Value of b :\n",b)

c = th.empty(3,3,2,2) # 4D => (blocks, layers, rows, columns)
print("Value of c : ",c)

# Tensor with random values
print("\n-- Random value tensor --\n")
p = th.rand(2) # 1D
print("Value of p : ",p)

q = th.rand(2,3) # 2D
print("Value of q :\n",q)

r = th.rand(2,2,2) # 3D
print("Value of r :\n",r)

# Zero and one tensors
print("\n-- Zero and one tensors --\n")
zero = th.zeros(2,2)
print("Value of zero :\n",zero)

one = th.ones(3,1)
print("\nValue of one :\n",one)
print("Data type of one :",one.dtype) # dtype => returns the data type

val1 = th.ones(2,2, dtype = th.int)
print("\nValue of val1 :\n",val1)
print("Data type of val1 :",val1.dtype)

val2 = th.zeros(3,2, dtype = th.double)
print("\nValue of val2 :\n",val2)
print("Data type of val2 :",val2.dtype)

val3 = th.zeros(2,2, dtype = th.float16)
print("\nValue of val3 :\n",val3)
print("Data type of val3 :",val3.dtype)
print("Size of val3 : ",val3.size()) # size() => checks the tensor size

# Direct tensor creation using a variable
print("\n-- Direct tensor creation --\n")
val = th.tensor([1.2,2.5,3.6])
print("Value of val : ",val)
print("Size of val : ",val.size())
print("Data type of val : ",val.dtype)

# Tensor operations using rand
print("\n-- tensor operations using rand --\n")

x = th.rand(2,2)
y = th.rand(2,2)
sum = x + y # Addition
# (or)
sum = th.add(x,y)
print("Operation-1")
print("\nValue of x :\n",x)
print("Value of y :\n",y)
print("Value of sum :\n",sum)

m = th.rand(2,4)
n = th.rand(2,4)
print("\nOperation-2")
print("Value of m :\n",m)
print("\nValue of n before the function :\n",n)
n.add_(m) # Addition using a function (n == n + m)
print("Value of n after the function :\n",n)

a = th.rand(3,3)
b = th.rand(3,3)
print("\nOperation-3") # Subtraction
print("\nValue of a before the function :\n",a) 
print("Value of b :\n",b)
sub = th.sub(a,b)
print("Value of sub (Using function) :\n",sub)
# (or)
sub = a - b
print("Value of sub (Normally) :\n",sub)
# (or)
a.sub_(b) # a == a - b (value changed)
print("Value of a after the function :\n",a)

d = th.rand(2,2)
k = th.rand(2,2)
print("\nOperation-4") # Multiplication
print("\nValue of d before the function :\n",d) 
print("Value of k :\n",k)
multi = d * k
print("Value of multi (Normally) :\n",multi)
# (or)
multi = th.mul(d,k)
print("Value of multi (Using function) :\n",multi)
# (or)
d.mul_(k) # d == d * k (value changed)
print("Value of d after the function :\n",d)

p = th.rand(2,2)
q = th.rand(2,2)
print("\nOperation-5") # Division
print("\nValue of p before the function :\n",p) 
print("Value of q :\n",q)
div = p / q
print("Value of div (Normally) :\n",div)
# (or)
div = th.div(p,q)
print("Value of div (Using function) :\n",div)
# (or)
p.div_(q) # p == p / q (value changed)
print("Value of p after the function :\n",p)

# Slicing
print("\n-- Slicing on Tensors --\n")

x = th.rand(4,3)
print("Value of x :\n",x)
print("Value of x[1,1] (with tensor) : ",x[1,1]) # with tensor
print("Value of x[1,1] (without tensor) : ",x[1,1].item()) # without the tensor wrapper (value only)
print("Value of x[:3] :\n",x[:3]) # Row slicing
print("Value of x[:2:2] :\n",x[:2:2]) # Step slicing
print("Value of x[:,1:2] :\n",x[:, 1:2]) # Column slicing
print("Value of x[1,:] :\n",x[1, :]) 
print("Value of x[:,0] :\n",x[:,0])

# Reshape tensors
print("\n-- Resize tensors --\n")
# 1)
v = th.rand(4,4)
print("Value of v :\n",v) # 2D array
y = v.view(16) # 2D --> 1D
print("Value of y :\n",y) # 1D array

# 2)
w = th.rand(4,4)
print("Value of w :\n",w)
u = w.view(-1,8) # -1 => 2
# PyTorch infers the missing dimension automatically.
# Here, -1 means "infer this dimension."
# 16 => total elements in the tensor
# inferred dimension * 8 = 16
# inferred dimension = 16 / 8 = 2
print("Size of u :",u.size())
print("Value of u :\n",u)

# Conversion from tensor to NumPy
print("\n-- Conversion from tensor to NumPy --\n")
a = th.ones(5)
print("Value of a :",a)
b = a.numpy() # now, a is converted to NumPy
print("Value of b :",b)
print("type of b :",type(b))
# Drawback: If PyTorch is running on the CPU instead of the GPU,
#           then `a` and `b` share the same memory location.
a.add_(1) # a += 1
print("Value of a (After Add function) : ",a)
print("Value of b (After Add function) : ",b)

# Conversion from NumPy to PyTorch
print("\n-- Conversion from NumPy to PyTorch --\n")
d = np.ones(5)
print("Value of d :",d)
e = th.from_numpy(d)
print("Value of e :",e)
print("type of e :",type(e))
# The same drawback applies here too.
d += 3
print("Value of d (After Add function) :",d)
print("Value of e (After Add function) :",e)

# This tells PyTorch that gradients need to be calculated later in the code.
f = th.ones(5, requires_grad=True)
print("Value of f :",f)

# Only for GPU PyTorch users
if th.cuda.is_available():
    device = th.device("cuda") # PyTorch object representing the GPU
    add1 = th.ones(5, device=device) # stored on the GPU
    add2 = th.ones(5) # created on CPU
    add2 = add2.to(device) # CPU --> GPU
    sum = add1 + add2 # created on GPU
    sum = sum.to("cpu") # GPU --> CPU
    print(sum) # tensor([2., 2., 2., 2., 2.])
