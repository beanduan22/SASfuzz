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


def _case_042() -> bool:
    torch = _torch()
    _torch_require_cuda(torch)
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
        cpu_err = type(exc).__name__ + ": " + str(exc)[:160]
    try:
        gpu = gpu_model(x.cuda()).cpu()
    except Exception as exc:
        gpu_err = type(exc).__name__ + ": " + str(exc)[:160]
    if cpu_err or gpu_err:
        ok = (cpu_err is None) != (gpu_err is None)
        return _print_result(ok, f"state=execution_mode(train/eval switch) cpu_err={cpu_err} gpu_err={gpu_err}")
    close = torch.allclose(cpu, gpu, atol=1e-6, rtol=1e-6, equal_nan=True)
    diff = float(torch.nan_to_num((cpu - gpu).abs(), nan=0.0).max().item())
    return _print_result(not close, f"state=execution_mode(train/eval switch) allclose={close} max_diff={diff:.4e}")


def main() -> int:
    print("CASE state_bug_042 [pytorch]")
    print("status=confirmed state_dimension=execution mode")
    try:
        ok = _case_042()
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
