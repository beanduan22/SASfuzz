# Issue: https://github.com/pytorch/pytorch/issues/114080
# Status: confirmed
# State: execution mode
import torch
import torch.nn as nn

assert torch.cuda.is_available(), 'CUDA is required'

class Model(nn.Module):
    def forward(self, x):
        h = x.view(x.size(0), -1)
        return torch.matrix_exp(h)

model = Model()
x_cpu = torch.tensor([[1.0, 1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 10.0, 200.0]])
x_gpu = x_cpu.cuda()
cpu_model = torch.jit.trace(model, x_cpu)
gpu_model = torch.jit.trace(Model().cuda(), x_gpu)
cpu = cpu_model(x_cpu)
gpu = gpu_model(x_gpu).cpu()

print('CPU:', cpu)
print('GPU:', gpu)
