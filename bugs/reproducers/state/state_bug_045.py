# Issue: https://github.com/pytorch/pytorch/issues/114087
# Status: confirmed
# State: gradient tracking
import torch
import torch.nn as nn

assert torch.cuda.is_available(), 'CUDA is required'
torch.manual_seed(0)

class Model(nn.Module):
    def forward(self, x):
        h = torch.mm(x, x, out=x)
        return h

model = Model()
x = torch.randn(32, 32)
with torch.no_grad():
    cpu = model(x.clone())
    gpu = model(x.clone().cuda()).cpu()

print('CPU:', cpu.flatten()[:8])
print('GPU:', gpu.flatten()[:8])
