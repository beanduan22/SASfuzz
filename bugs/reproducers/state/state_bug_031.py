import torch
import torch.nn as nn

assert torch.cuda.is_available(), 'CUDA is required'
torch.manual_seed(0)

class Model(nn.Module):
    def forward(self, x):
        h = torch.std(x)
        return h

model = Model()
x_cpu = (torch.randn(1000, dtype=torch.float32) * 1e19 + 1e20).requires_grad_(True)
x_gpu = x_cpu.detach().cuda().requires_grad_(True)
out_cpu = model(x_cpu)
out_gpu = model(x_gpu)
out_cpu.backward()
out_gpu.backward()

print('CPU grad:', x_cpu.grad[:8])
print('GPU grad:', x_gpu.grad.cpu()[:8])
