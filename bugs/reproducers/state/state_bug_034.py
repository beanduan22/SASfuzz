# Issue: https://github.com/pytorch/pytorch/issues/181805
# Status: fixed
# State: gradient tracking
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
out_gpu = torch.empty(1, dtype=torch.uint8, device='cuda')
with torch.no_grad():
    cpu = model((a, b, out_cpu))
    gpu = model((a.cuda(), b.cuda(), out_gpu)).cpu()

print('CPU:', cpu)
print('GPU:', gpu)
