import torch
import torch.nn as nn

assert torch.cuda.is_available(), 'CUDA is required'
torch.manual_seed(0)

class Model(nn.Module):
    def forward(self, x):
        h = torch.std(x)
        return h

model = Model()
x = torch.randn(1000, dtype=torch.float32) * 1e19 + 1e20
with torch.no_grad():
    y_cpu = model(x)
    y_gpu = model(x.cuda()).cpu()
    expected = torch.std(x.double())

print('CPU:', y_cpu.item())
print('CUDA:', y_gpu.item())
print('Expected:', expected.item())
