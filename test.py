import torch
import numpy as np
import torch


x = torch.ones(2,2, requires_grad = True)
y = torch.tensor([[2,3],[2,3]], dtype = torch.float,requires_grad = True)

z = x*x + y + 1
print(z)

k = z*z
print(k)

tmp_1 = torch.tensor([[1,2],[3,4]], dtype = torch.float)

z.retain_grad()         # non-leaf Tenor need retain_grad() and locat before .backward() if require its grad

k.backward(tmp_1)
print("z:", z.grad)
print(z.requires_grad)
print("y:", y.grad)

print("x:", x.grad)
