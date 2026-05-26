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


def _case_045() -> bool:
    torch = _torch()
    _torch_require_cuda(torch)
    torch.manual_seed(0)
    x = torch.randn(32, 32)
    with torch.no_grad():
        x_cpu = x.clone()
        x_gpu = x.clone().cuda()
        cpu = torch.mm(x_cpu, x_cpu, out=x_cpu)
        gpu = torch.mm(x_gpu, x_gpu, out=x_gpu).cpu()
    close = torch.allclose(cpu, gpu, atol=1e-5, rtol=1e-5, equal_nan=True)
    diff = float((cpu - gpu).abs().max().item())
    return _print_result(not close, f"state=gradient_tracking(torch.no_grad) allclose={close} max_diff={diff:.4e}")


def main() -> int:
    print("CASE state_bug_045 [pytorch]")
    print("status=confirmed state_dimension=gradient tracking")
    try:
        ok = _case_045()
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
