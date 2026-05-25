                                                  
import torch

cpu = torch.linspace(4.3, -3, 50, dtype=torch.int64, device="cpu")
gpu = torch.linspace(4.3, -3, 50, dtype=torch.int64, device="cuda").cpu()
print("cpu:", cpu.tolist())
print("gpu:", gpu.tolist())
print("differing indices:", (cpu != gpu).nonzero(as_tuple=True)[0].tolist())
