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


def _case_043() -> bool:
    torch = _torch()
    _torch_require_cuda(torch)
    import torch.nn as nn

    class Model(nn.Module):
        def forward(self, x):
            y = x.view(x.size(0), -1)
            return torch.matrix_exp(y)

    inp = torch.tensor([[1.0, 1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 10.0, 200.0]])
    model = Model()
    traced_cpu = torch.jit.trace(model, inp)
    traced_gpu = torch.jit.trace(model.cuda(), inp.cuda())
    cpu = traced_cpu(inp)
    gpu = traced_gpu(inp.cuda()).cpu()
    ok = not torch.equal(torch.isnan(cpu), torch.isnan(gpu)) or not torch.equal(torch.isinf(cpu), torch.isinf(gpu))
    return _print_result(ok, f"state=execution_mode(torch.jit.trace) cpu={cpu} gpu={gpu}")


def main() -> int:
    print("CASE state_bug_043 [pytorch]")
    print("status=confirmed state_dimension=execution mode")
    try:
        ok = _case_043()
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
