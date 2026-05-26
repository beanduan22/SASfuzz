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
    y_cpu = model(x)
    y_gpu = model(x.cuda()).cpu()

print('CPU:', y_cpu.tolist())
print('CUDA:', y_gpu.tolist())
