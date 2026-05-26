import torch
import torch.nn as nn

assert torch.cuda.is_available(), 'CUDA is required'

class Model(nn.Module):
    def forward(self, x):
        h = torch.linspace(3.7, -3, 10, dtype=torch.int64, device=x.device)
        return h

model = Model()
x_cpu = torch.empty((), device='cpu')
x_gpu = torch.empty((), device='cuda')
with torch.no_grad():
    y_cpu = model(x_cpu)
    y_gpu = model(x_gpu).cpu()

print('CPU:', y_cpu.tolist())
print('CUDA:', y_gpu.tolist())
