import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch

def f(device):
    out = torch.tensor([], dtype=torch.int16, device=device)
    torch.fmin(torch.tensor([-3000000], dtype=torch.int64, device=device), torch.tensor([-2], device=device), out=out)
    print(out.cpu())

f("cpu")
f("cuda")
