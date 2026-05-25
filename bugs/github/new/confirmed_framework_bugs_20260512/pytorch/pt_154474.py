import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch

x = torch.tensor([torch.inf, -2.0, 3.0, -torch.inf])
print(torch.fft.fft(x))
print(torch.fft.fft(x.cuda()).cpu())
