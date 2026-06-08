# Issue: https://github.com/pytorch/pytorch/issues/179784
# Status: confirmed
# State: gradient tracking
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.sens = nn.BatchNorm1d(8)
        self.fc2 = nn.Linear(8, 8)

    def forward(self, x):
        h = self.fc1(x)
        h = F.hardswish(h)
        h = self.sens(h)
        h = self.fc2(h)
        h = torch.log1p(h)
        return torch.xlogy(h, h)

model = Model()
x = torch.randn(4, 8)
x.requires_grad_(True)
out = model(x)
out.sum().backward()

print('Gradient:', x.grad)
print('Reference gradient:', 'finite gradient')
