import torch
assert torch.cuda.is_available(), 'CUDA is required'
import torch.nn as nn
import torch.nn.functional as F

class M(nn.Module):

    def forward(self, x):
        return F.relu(x)
model = M()
model.train()
model.eval()
x_cpu = torch.tensor([-0.0])
x_gpu = x_cpu.cuda()
cpu = model(x_cpu)
gpu = model(x_gpu).cpu()
print(f'state=execution_mode(train/eval switch) cpu={cpu} sign={torch.signbit(cpu)} gpu={gpu} sign={torch.signbit(gpu)}')
