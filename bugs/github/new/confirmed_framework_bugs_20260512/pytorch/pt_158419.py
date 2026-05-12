import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch

x = torch.full((2, 2), 2.0, dtype=torch.float16)
y = torch.tensor([float("nan"), -float("nan")], dtype=torch.float16)
print(torch.copysign(x, y))
print(torch.copysign(x.cuda(), y.cuda()).cpu())
