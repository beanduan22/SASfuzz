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



def _np():
    try:
        import numpy as np
    except ImportError as exc:
        raise SkipCase(f"missing numpy: {exc}") from exc
    return np


def _max_abs_diff(a, b) -> float:
    np = _np()
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        diff = np.abs(aa - bb)
    if diff.size == 0:
        return 0.0
    if np.all(np.isnan(diff)):
        return float("nan")
    return float(np.nanmax(diff))


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


def _case_033() -> bool:
    np = _np()
    torch = _torch()
    _torch_require_cuda(torch)
    torch.manual_seed(0)
    with torch.no_grad():
        n = 1_000_000
        x = 1.0 + 0.0001 * torch.randn(n, dtype=torch.float32)
        ref = np.cumprod(x.numpy().astype(np.float64))
        cpu = torch.cumprod(x, dim=0).numpy()
        gpu = torch.cumprod(x.cuda(), dim=0).cpu().numpy()
    cpu_err = float(np.max(np.abs(cpu - ref)))
    gpu_err = float(np.max(np.abs(gpu - ref)))
    ok = _max_abs_diff(cpu, gpu) > 1e-3
    return _print_result(ok, f"state=gradient_tracking(torch.no_grad) cpu_err={cpu_err:.4e} gpu_err={gpu_err:.4e}")


def main() -> int:
    print("CASE state_bug_033 [pytorch]")
    print("status=fixed state_dimension=gradient tracking")
    try:
        ok = _case_033()
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
