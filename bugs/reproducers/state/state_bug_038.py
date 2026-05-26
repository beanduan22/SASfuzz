import torch
import torch.nn as nn

assert torch.cuda.is_available(), 'CUDA is required'

class Model(nn.Module):
    def forward(self, x):
        h = torch.clamp(x, min=0, max=1)
        return h

model = Model()
x = torch.tensor([-0.0])
with torch.no_grad():
    cpu = model(x)
    gpu = model(x.cuda()).cpu()

print('CPU:', cpu, torch.signbit(cpu))
print('GPU:', gpu, torch.signbit(gpu))
