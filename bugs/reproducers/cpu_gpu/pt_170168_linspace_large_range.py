                                                                         
import torch

cpu = torch.linspace(1e6, -1e6, 1000, dtype=torch.int64, device="cpu")
gpu = torch.linspace(1e6, -1e6, 1000, dtype=torch.int64, device="cuda").cpu()

print("# differing indices:", (cpu != gpu).sum().item())
print("first 5 cpu:", cpu[:5].tolist())
print("first 5 gpu:", gpu[:5].tolist())
