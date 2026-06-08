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


def _case_041() -> bool:
    torch = _torch()
    _torch_require_cuda(torch)
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.avg_pool1d = nn.AvgPool1d(kernel_size=2)
            self.channel_shuffle = nn.ChannelShuffle(groups=3)

        def forward(self, x):
            x = x.view(-1, 3, 32 * 32)
            x = self.avg_pool1d(x)
            return self.channel_shuffle(x)

    model = Model()
    model.train()
    model.eval()
    x = torch.rand(2, 3, 32, 32)
    cpu_err = gpu_err = None
    try:
        cpu = model.cpu()(x)
    except Exception as exc:
        cpu = None
        cpu_err = type(exc).__name__ + ": " + str(exc)[:120]
    try:
        gpu = model.cuda()(x.cuda())
    except Exception as exc:
        gpu = None
        gpu_err = type(exc).__name__ + ": " + str(exc)[:160]
    ok = cpu is not None and gpu_err is not None and "channel_shuffle" in gpu_err
    return _print_result(ok, f"state=execution_mode(train/eval switch) cpu_shape={None if cpu is None else tuple(cpu.shape)} cpu_err={cpu_err} gpu_err={gpu_err}")


def main() -> int:
    print("CASE state_bug_041 [pytorch]")
    print("status=fixed state_dimension=execution mode")
    try:
        ok = _case_041()
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
