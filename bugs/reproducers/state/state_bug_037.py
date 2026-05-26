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


def _case_037() -> bool:
    torch = _torch()
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(123)
    t, n, c = 3, 1, 3
    log_probs = F.log_softmax(torch.randn(t, n, c), dim=-1).double()
    log_probs.requires_grad_(True)
    targets = torch.tensor([[1]])
    input_lens = torch.tensor([t])
    target_lens = torch.tensor([1])
    loss_fn = nn.CTCLoss(blank=0)

    def fn(inp):
        return loss_fn(inp, targets, input_lens, target_lens)

    try:
        torch.autograd.gradcheck(fn, (log_probs,), raise_exception=True)
    except Exception as exc:
        return _print_result(True, f"state=gradient_tracking(torch.autograd.gradcheck) err={type(exc).__name__}: {str(exc)[:200]}")
    return _print_result(False, "state=gradient_tracking(torch.autograd.gradcheck) gradcheck passed")


def main() -> int:
    print("CASE state_bug_037 [pytorch]")
    print("status=fixed state_dimension=gradient tracking")
    try:
        ok = _case_037()
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
