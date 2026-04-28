# https://github.com/pytorch/pytorch/issues/158172
import torch

a = torch.tensor([100000], dtype=torch.int64)
b = torch.tensor([-1], dtype=torch.int64)

out_cpu = torch.empty(1, dtype=torch.int16)
out_gpu = torch.empty(1, dtype=torch.int16, device="cuda")

torch.fmin(a, b, out=out_cpu)
torch.fmin(a.cuda(), b.cuda(), out=out_gpu)

print("cpu:", out_cpu)
print("gpu:", out_gpu.cpu())
