# Issue: https://github.com/pytorch/pytorch/issues/181533
# Status: confirmed
# State: execution mode
import torch
import torch.nn as nn

assert torch.cuda.is_available(), 'CUDA is required'

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.sens = nn.ReLU()

    def forward(self, x):
        h = self.sens(x)
        return h

model = Model()
x = torch.tensor([-0.0])
model.train()
_ = model(x)
model.eval()
cpu = model(x)
gpu = model(x.cuda()).cpu()

print('CPU:', cpu, torch.signbit(cpu))
print('GPU:', gpu, torch.signbit(gpu))
