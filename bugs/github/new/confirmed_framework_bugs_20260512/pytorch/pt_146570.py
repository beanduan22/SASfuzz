import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch

print(torch.full((1, 1), float("inf"), dtype=torch.float16).geometric_(5e-9))
print(torch.full((1, 1), float("inf"), dtype=torch.float16, device="cuda").geometric_(5e-9).cpu())
