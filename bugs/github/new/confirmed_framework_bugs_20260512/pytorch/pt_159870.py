import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch

x = torch.tensor(complex(-float("inf"), 2.0))
print(torch.nn.functional.sigmoid(x))
print(torch.nn.functional.sigmoid(x.cuda()).cpu())
