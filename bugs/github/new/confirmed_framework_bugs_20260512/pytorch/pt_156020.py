import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch

x = torch.zeros(8, dtype=torch.cfloat) + 1j * 1.0e38
print(torch.fft.ifft(x))
print(torch.fft.ifft(x.cuda()).cpu())
