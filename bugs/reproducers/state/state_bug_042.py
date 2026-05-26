import torch
assert torch.cuda.is_available(), 'CUDA is required'
import torch.nn as nn

class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.bn = nn.LazyBatchNorm1d()

    def forward(self, x):
        return self.bn(x.view(x.size(0), -1))
torch.manual_seed(0)
x = torch.rand(2, 3, 32, 32)
cpu_model = Model().train()
gpu_model = Model().cuda().train()
cpu = gpu = None
cpu_err = gpu_err = None
try:
    cpu = cpu_model(x)
except Exception as exc:
    cpu_err = type(exc).__name__ + ': ' + str(exc)[:160]
try:
    gpu = gpu_model(x.cuda()).cpu()
except Exception as exc:
    gpu_err = type(exc).__name__ + ': ' + str(exc)[:160]
if cpu_err or gpu_err:
    print(f'state=execution_mode(train/eval switch) cpu_err={cpu_err} gpu_err={gpu_err}')
close = torch.allclose(cpu, gpu, atol=1e-06, rtol=1e-06, equal_nan=True)
diff = float(torch.nan_to_num((cpu - gpu).abs(), nan=0.0).max().item())
print(f'state=execution_mode(train/eval switch) allclose={close} max_diff={diff:.4e}')
