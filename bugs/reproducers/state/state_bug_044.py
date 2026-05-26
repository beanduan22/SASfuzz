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


def _case_044() -> bool:
    torch = _torch()
    _torch_require_cuda(torch)
    x = torch.tensor(
        [
            [0.0100, 0.0000, 0.0000, 0.0000, 0.1000],
            [0.0000, 0.0100, 0.0000, 0.1000, 0.0000],
            [0.0000, 0.0000, 0.0100, 0.0000, 0.0000],
            [0.0000, 0.1000, 0.0000, 0.0100, 0.0000],
            [0.1000, 0.0000, 0.0000, 0.0000, 0.0100],
        ]
    )
    with torch.no_grad():
        cpu_vals, cpu_vecs = torch.lobpcg(x)
        gpu_vals, gpu_vecs = torch.lobpcg(x.cuda())
    diff = float((cpu_vecs.abs() - gpu_vecs.cpu().abs()).abs().max().item())
    ok = diff > 1e-2
    return _print_result(ok, f"state=gradient_tracking(torch.no_grad) eig_cpu={cpu_vals} eig_gpu={gpu_vals.cpu()} vec_abs_diff={diff:.4e}")


def main() -> int:
    print("CASE state_bug_044 [pytorch]")
    print("status=confirmed state_dimension=gradient tracking")
    try:
        ok = _case_044()
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
