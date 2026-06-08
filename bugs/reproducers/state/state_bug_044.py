# Issue: https://github.com/pytorch/pytorch/issues/114081
# Status: confirmed
# State: gradient tracking
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
    cpu_vals, cpu_vecs = model(x)
    gpu_vals, gpu_vecs = model(x.cuda())

print('CPU:', cpu_vals, cpu_vecs.abs().flatten()[:8])
print('GPU:', gpu_vals.cpu(), gpu_vecs.cpu().abs().flatten()[:8])
