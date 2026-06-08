# Issue: https://github.com/pytorch/pytorch/issues/114093
# Status: confirmed
# State: execution mode
import torch
import torch.nn as nn

assert torch.cuda.is_available(), 'CUDA is required'
torch.manual_seed(0)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.sens = nn.LazyBatchNorm1d()

    def forward(self, x):
        h = x.view(x.size(0), -1)
        return self.sens(h)

x = torch.rand(2, 3, 32, 32)
cpu_model = Model()
gpu_model = Model().cuda()
cpu_model.train()
gpu_model.train()
_ = cpu_model(x)
_ = gpu_model(x.cuda())
cpu_model.eval()
gpu_model.eval()
cpu = cpu_model(x)
gpu = gpu_model(x.cuda()).cpu()

print('CPU:', cpu.flatten()[:8])
print('GPU:', gpu.flatten()[:8])
