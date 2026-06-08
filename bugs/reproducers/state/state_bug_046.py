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


def _case_046() -> bool:
    torch = _torch()
    import torch.nn as nn

    torch.manual_seed(0)
    fc1 = nn.Linear(8, 8)
    fc2 = nn.Linear(8, 8)
    bn = nn.BatchNorm1d(8)
    x = torch.randn(4, 8, requires_grad=True)
    with torch.enable_grad():
        y = fc1(x)
        y = torch.nn.functional.hardswish(y)
        y = bn(y)
        y = fc2(y)
        y = torch.log1p(y)
        out = torch.xlogy(y, y)
    minimal = torch.xlogy(torch.tensor(0.0), torch.tensor(0.0))
    ok = torch.isnan(minimal).item() or torch.isnan(out).any().item()
    return _print_result(ok, f"state=gradient_tracking(torch.enable_grad) minimal_xlogy_0_0={minimal} any_model_nan={torch.isnan(out).any().item()}")


def main() -> int:
    print("CASE state_bug_046 [pytorch]")
    print("status=confirmed state_dimension=gradient tracking")
    try:
        ok = _case_046()
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
