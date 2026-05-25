                                                                               
import torch

a = torch.tensor([100000], dtype=torch.int64)
b = torch.tensor([-1], dtype=torch.int64)

oc = torch.empty(1, dtype=torch.uint8)
og = torch.empty(1, dtype=torch.uint8, device="cuda")

torch.fmax(a, b, out=oc)
torch.fmax(a.cuda(), b.cuda(), out=og)

print("cpu:", oc)
print("gpu:", og.cpu())
