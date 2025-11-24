import torch

x = torch.randn(3, 3)
g = x.cuda()

print("x device:", x.device)
print("g device:", g.device)
print("Are they equal?", x.device == g.device)
