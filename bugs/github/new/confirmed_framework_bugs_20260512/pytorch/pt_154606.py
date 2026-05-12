import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch

x = torch.tensor([complex(2, float("inf")), complex(-3, 4)], dtype=torch.complex128)
print(torch.cumprod(x, 0))
print(torch.cumprod(x.cuda(), 0).cpu())
