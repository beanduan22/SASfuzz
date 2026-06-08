import torch
import torch.nn.functional as F

assert torch.cuda.is_available(), 'CUDA is required'

x = torch.tensor([-0.0])
with torch.no_grad():
    cpu = F.relu(x)
    gpu = F.relu(x.cuda()).cpu()

print('CPU:', cpu, torch.signbit(cpu))
print('GPU:', gpu, torch.signbit(gpu))
