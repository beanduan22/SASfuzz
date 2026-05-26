import torch
import torch.nn as nn

assert torch.cuda.is_available(), 'CUDA is required'

class Model(nn.Module):
    def forward(self, x):
        h = torch.linspace(3.7, -3, 10, dtype=torch.int64, device=x.device)
        return h

x_cpu = torch.empty((), device='cpu')
x_gpu = torch.empty((), device='cuda')
cpu_model = torch.jit.trace(Model(), x_cpu)
gpu_model = torch.jit.trace(Model().cuda(), x_gpu)
cpu = cpu_model(x_cpu)
gpu = gpu_model(x_gpu).cpu()

print('CPU:', cpu.tolist())
print('GPU:', gpu.tolist())
