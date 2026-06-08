import torch
import torch.nn as nn

assert torch.cuda.is_available(), 'CUDA is required'

class Model(nn.Module):
    def forward(self, x):
        base, index, src = x
        h = torch.Tensor.scatter(base, 0, index, src)
        return h

model = Model()
x_cpu = (torch.tensor([1, 2, 3, 4, 5, 6, 7]), torch.tensor([0, 1, 1, 2]), torch.tensor([12, 14, 16, 18, 20]))
x_gpu = tuple(v.cuda() for v in x_cpu)
with torch.no_grad():
    cpu = model(x_cpu)
    gpu = model(x_gpu).cpu()

print('CPU:', cpu.tolist())
print('GPU:', gpu.tolist())
