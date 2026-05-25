                                                  
import torch

x = torch.zeros(4, dtype=torch.cfloat) + 1j * 8.5071e+37
cpu = torch.fft.ifft(x)
gpu = torch.fft.ifft(x.cuda()).cpu()
print("cpu:", cpu)
print("gpu:", gpu)
