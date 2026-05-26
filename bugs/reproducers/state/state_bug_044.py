import torch
import torch.nn as nn

assert torch.cuda.is_available(), 'CUDA is required'

class Model(nn.Module):
    def forward(self, x):
        vals, vecs = torch.lobpcg(x)
        return vals, vecs

model = Model()
x = torch.tensor([[0.0100, 0.0000, 0.0000, 0.0000, 0.1000], [0.0000, 0.0100, 0.0000, 0.1000, 0.0000], [0.0000, 0.0000, 0.0100, 0.0000, 0.0000], [0.0000, 0.1000, 0.0000, 0.0100, 0.0000], [0.1000, 0.0000, 0.0000, 0.0000, 0.0100]])
with torch.no_grad():
    vals_cpu, vecs_cpu = model(x)
    vals_gpu, vecs_gpu = model(x.cuda())

print('CPU eigenvalues:', vals_cpu)
print('CUDA eigenvalues:', vals_gpu.cpu())
print('Vector abs diff:', float((vecs_cpu.abs() - vecs_gpu.cpu().abs()).abs().max().item()))
