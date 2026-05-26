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


def _case_036() -> bool:
    torch = _torch()
    _torch_require_cuda(torch)
    import torch.nn as nn
    import torch.nn.functional as F

    class M(nn.Module):
        def forward(self, x):
            return F.relu(x)

    model = M()
    model.train()
    model.eval()
    x_cpu = torch.tensor([-0.0])
    x_gpu = x_cpu.cuda()
    cpu = model(x_cpu)
    gpu = model(x_gpu).cpu()
    ok = bool(torch.signbit(cpu).item()) != bool(torch.signbit(gpu).item())
    return _print_result(ok, f"state=execution_mode(train/eval switch) cpu={cpu} sign={torch.signbit(cpu)} gpu={gpu} sign={torch.signbit(gpu)}")


def main() -> int:
    print("CASE state_bug_036 [pytorch]")
    print("status=fixed state_dimension=execution mode")
    try:
        ok = _case_036()
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
