# Issue: https://github.com/pytorch/pytorch/issues/121208
# Status: fixed
# State: execution mode
import torch
import torch.nn as nn

assert torch.cuda.is_available(), 'CUDA is required'

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.sens = nn.ChannelShuffle(groups=3)
        self.pool = nn.AvgPool1d(kernel_size=2)

    def forward(self, x):
        h = x.view(-1, 3, 32 * 32)
        h = self.pool(h)
        return self.sens(h)

model = Model()
x = torch.rand(2, 3, 32, 32)
model.train()
_ = model.cpu()(x)
model.eval()
try:
    cpu = model.cpu()(x)
except Exception as exc:
    cpu = type(exc).__name__ + ': ' + str(exc).splitlines()[0]
try:
    gpu = model.cuda()(x.cuda())
except Exception as exc:
    gpu = type(exc).__name__ + ': ' + str(exc).splitlines()[0]

print('CPU:', cpu if isinstance(cpu, str) else tuple(cpu.shape))
print('GPU:', gpu if isinstance(gpu, str) else tuple(gpu.shape))
