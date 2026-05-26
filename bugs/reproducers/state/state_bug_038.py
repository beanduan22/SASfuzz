import torch
import torch.nn as nn

assert torch.cuda.is_available(), 'CUDA is required'

class Model(nn.Module):
    def forward(self, x):
        h = torch.clamp(x, min=0, max=1)
        return h

model = Model()
x = torch.tensor([-0.0])
with torch.no_grad():
    y_cpu = model(x)
    y_gpu = model(x.cuda()).cpu()

print('CPU:', y_cpu, torch.signbit(y_cpu))
print('CUDA:', y_gpu, torch.signbit(y_gpu))
