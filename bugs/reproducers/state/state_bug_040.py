import torch


def run():
    assert torch.cuda.is_available(), 'CUDA is required'
    import os
    import tempfile
    import torch.nn as nn
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    if not dist.is_available():
        raise RuntimeError('torch.distributed is not available')
    if not dist.is_nccl_available():
        raise RuntimeError('NCCL is not available for CUDA DDP')
    torch.manual_seed(202311)

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(()))

        def forward(self, inp):
            return torch.linalg.pinv(inp * self.scale)

    def init_group() -> None:
        if dist.is_initialized():
            return
        init_dir = tempfile.mkdtemp(prefix='smolfuzz-ddp-')
        init_file = os.path.join(init_dir, 'init')
        dist.init_process_group(backend='nccl', init_method=f'file://{init_file}', rank=0, world_size=1)
    base = -torch.ones(2, 3, 8, 8)
    noise = 0.25 * torch.randn(2, 3, 8, 8)
    x_cpu = base + noise
    cpu = Model()(x_cpu).detach()
    init_group()
    torch.cuda.set_device(0)
    model = Model().cuda()
    ddp = DDP(model, device_ids=[0])
    x_gpu = x_cpu.cuda().requires_grad_(True)
    gpu_out = ddp(x_gpu)
    gpu_out.sum().backward()
    gpu = gpu_out.detach().cpu()
    diff = float(torch.nan_to_num((cpu - gpu).abs(), nan=0.0, posinf=1e+30, neginf=1e+30).max().item())
    ok = diff > 0.001
    print(f'state=distribution_strategy(DDP) max_diff={diff:.4e}')
    print('BUG_REPRODUCED' if ok else 'NOT_REPRODUCED')
    return


run()
