import torch
import torch.nn as nn

assert torch.cuda.is_available(), 'CUDA is required'

class Model(nn.Module):
    def forward(self, x):
        a, b, out = x
        torch.fmax(a, b, out=out)
        return out

model = Model()
a = torch.tensor([100000], dtype=torch.int64)
b = torch.tensor([-1], dtype=torch.int64)
out_cpu = torch.empty(1, dtype=torch.uint8)
out_cuda = torch.empty(1, dtype=torch.uint8, device='cuda')
with torch.no_grad():
    y_cpu = model((a, b, out_cpu))
    y_gpu = model((a.cuda(), b.cuda(), out_cuda)).cpu()
expected = torch.fmax(a, b).to(torch.uint8)

print('CPU:', y_cpu)
print('CUDA:', y_gpu)
print('Expected:', expected)
