                                                                          
import torch

x = torch.zeros(4, dtype=torch.cfloat) + 1j * 3.4e38
print("cpu:", torch.fft.ifft(x))
print("gpu:", torch.fft.ifft(x.cuda()).cpu())
