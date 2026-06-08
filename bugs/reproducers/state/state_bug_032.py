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


def _case_032() -> bool:
    torch = _torch()
    _torch_require_cuda(torch)
    with torch.no_grad():
        x_cpu = torch.tensor([1, 2, 3, 4, 5, 6, 7])
        y_cpu = torch.tensor([0, 1, 1, 2])
        z_cpu = torch.tensor([12, 14, 16, 18, 20])
        x_gpu = x_cpu.cuda()
        y_gpu = y_cpu.cuda()
        z_gpu = z_cpu.cuda()
        cpu = torch.Tensor.scatter(x_cpu, 0, y_cpu, z_cpu)
        gpu = torch.Tensor.scatter(x_gpu, 0, y_gpu, z_gpu).cpu()
    ok = not torch.equal(cpu, gpu)
    return _print_result(ok, f"state=gradient_tracking(torch.no_grad) cpu={cpu.tolist()} gpu={gpu.tolist()}")


def main() -> int:
    print("CASE state_bug_032 [pytorch]")
    print("status=fixed state_dimension=gradient tracking")
    try:
        ok = _case_032()
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
