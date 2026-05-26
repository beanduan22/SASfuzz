import os
import tempfile
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

assert torch.cuda.is_available(), 'CUDA is required'
assert dist.is_available(), 'torch.distributed is required'
assert dist.is_nccl_available(), 'NCCL is required'
torch.manual_seed(202311)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, x):
        h = torch.linalg.pinv(x * self.scale)
        return h

init_dir = tempfile.mkdtemp(prefix='state-ddp-')
init_file = os.path.join(init_dir, 'init')
if not dist.is_initialized():
    dist.init_process_group(backend='nccl', init_method=f'file://{init_file}', rank=0, world_size=1)

device = torch.device('cuda:0')
x = -torch.ones(2, 3, 8, 8) + 0.25 * torch.randn(2, 3, 8, 8)
expected = Model()(x).detach()
model = Model().to(device)
ddp = DDP(model, device_ids=[0])
x = x.to(device)
x.requires_grad_(True)
out = ddp(x)
out.sum().backward()
state = out.detach().cpu()

print('State max diff:', float(torch.nan_to_num((expected - state).abs(), nan=0.0, posinf=1e30, neginf=1e30).max().item()))
dist.destroy_process_group()
