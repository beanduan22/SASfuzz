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


def _case_039() -> bool:
    np = _np()
    torch = _torch()
    _torch_require_cuda(torch)
    src = torch.from_numpy(np.array([0xFE00, 0xFE00, 0xFE00], dtype=np.uint16).view(np.float16))
    with torch.no_grad():
        cpu = torch.signbit(src)
        gpu = torch.signbit(src.cuda()).cpu()
    ok = not torch.equal(cpu, gpu)
    return _print_result(ok, f"state=gradient_tracking(torch.no_grad) cpu={cpu.tolist()} gpu={gpu.tolist()}")


def main() -> int:
    print("CASE state_bug_039 [pytorch]")
    print("status=fixed state_dimension=gradient tracking")
    try:
        ok = _case_039()
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
