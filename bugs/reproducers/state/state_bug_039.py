# Issue: https://github.com/pytorch/pytorch/issues/181806
# Status: fixed
# State: gradient tracking
import torch
import torch.nn as nn
import numpy as np

assert torch.cuda.is_available(), 'CUDA is required'

class Model(nn.Module):
    def forward(self, x):
        h = torch.signbit(x)
        return h

model = Model()
x = torch.from_numpy(np.array([0xFE00, 0xFE00, 0xFE00], dtype=np.uint16).view(np.float16))
with torch.no_grad():
    cpu = model(x)
    gpu = model(x.cuda()).cpu()

print('CPU:', cpu.tolist())
print('GPU:', gpu.tolist())
