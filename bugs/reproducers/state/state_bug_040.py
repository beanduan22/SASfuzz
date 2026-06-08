from __future__ import annotations

import math
import os
import traceback
import warnings
from typing import Callable


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
warnings.filterwarnings("ignore")


class SkipCase(Exception):
    pass



def _print_result(ok: bool, detail: str) -> bool:
    print(detail)
    print("BUG_REPRODUCED" if ok else "NOT_REPRODUCED")
    return bool(ok)


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise SkipCase(f"missing torch: {exc}") from exc
    return torch


def _torch_require_cuda(torch) -> None:
    if not torch.cuda.is_available():
        raise SkipCase("CUDA is not visible to PyTorch")


def _case_040() -> bool:
    torch = _torch()
    _torch_require_cuda(torch)
    import os
    import tempfile
    import torch.nn as nn
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    if not dist.is_available():
        raise SkipCase("torch.distributed is not available")
    if not dist.is_nccl_available():
        raise SkipCase("NCCL is not available for CUDA DDP")

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
        init_dir = tempfile.mkdtemp(prefix="smolfuzz-ddp-")
        init_file = os.path.join(init_dir, "init")
        dist.init_process_group(
            backend="nccl",
            init_method=f"file://{init_file}",
            rank=0,
            world_size=1,
        )



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

    diff = float(torch.nan_to_num((cpu - gpu).abs(), nan=0.0, posinf=1e30, neginf=1e30).max().item())
    ok = diff > 1e-3
    return _print_result(ok, f"state=distribution_strategy(DDP) max_diff={diff:.4e}")


def main() -> int:
    print("CASE state_bug_040 [pytorch]")
    print("status=fixed state_dimension=distribution strategy")
    try:
        ok = _case_040()
        return 0 if ok else 1
    except SkipCase as exc:
        print(f"SKIPPED: {exc}")
        return 2
    except Exception:
        print("HARNESS_ERROR:")
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
