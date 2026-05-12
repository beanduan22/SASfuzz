import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch

x = torch.complex(torch.full((2, 2), 2.0, dtype=torch.float64), torch.full((2, 2), torch.finfo(torch.float64).max, dtype=torch.float64))
print(torch.absolute(x))
print(torch.absolute(x.cuda()).cpu())
