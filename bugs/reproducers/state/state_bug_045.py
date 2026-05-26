import torch
import torch.nn as nn

assert torch.cuda.is_available(), 'CUDA is required'
torch.manual_seed(0)

class Model(nn.Module):
    def forward(self, x):
        h = torch.mm(x, x, out=x)
        return h

model = Model()
x = torch.randn(32, 32)
with torch.no_grad():
    y_cpu = model(x.clone())
    y_gpu = model(x.clone().cuda()).cpu()

print('Allclose:', torch.allclose(y_cpu, y_gpu, atol=1e-5, rtol=1e-5, equal_nan=True))
print('Max diff:', float((y_cpu - y_gpu).abs().max().item()))
