                                                                    
import torch

x = torch.tensor([-0.0, 0.0])
cpu = torch.median(x)
gpu = torch.median(x.cuda()).cpu()

print("cpu:", cpu, "signbit:", torch.signbit(cpu).item())
print("gpu:", gpu, "signbit:", torch.signbit(gpu).item())
