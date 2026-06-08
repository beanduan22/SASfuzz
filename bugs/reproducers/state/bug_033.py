import torch
import torch.nn as nn

assert torch.cuda.is_available(), 'CUDA is required'
torch.manual_seed(0)

class Model(nn.Module):
    def forward(self, x):
        h = torch.cumprod(x, dim=0)
        return h

model = Model()
x_cpu = (1.0 + 0.0001 * torch.randn(10000, dtype=torch.float32)).requires_grad_(True)
x_gpu = x_cpu.detach().cuda().requires_grad_(True)
out_cpu = model(x_cpu).sum()
out_gpu = model(x_gpu).sum()
out_cpu.backward()
out_gpu.backward()

print('CPU grad:', x_cpu.grad[-8:])
print('GPU grad:', x_gpu.grad.cpu()[-8:])
