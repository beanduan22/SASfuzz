import torch
import torch.nn as nn
import numpy as np

assert torch.cuda.is_available(), 'CUDA is required'
torch.manual_seed(0)

class Model(nn.Module):
    def forward(self, x):
        h = torch.cumprod(x, dim=0)
        return h

model = Model()
x = 1.0 + 0.0001 * torch.randn(1000000, dtype=torch.float32)
with torch.no_grad():
    y_cpu = model(x).numpy()
    y_gpu = model(x.cuda()).cpu().numpy()
expected = np.cumprod(x.numpy().astype(np.float64))

print('CPU error:', float(np.max(np.abs(y_cpu - expected))))
print('CUDA error:', float(np.max(np.abs(y_gpu - expected))))
