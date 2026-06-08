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


def _tf():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SkipCase(f"missing tensorflow: {exc}") from exc
    return tf


def _case_004() -> bool:
    tf = _tf()
    print("state=gradient_tracking(fake_quant gradient op)")
    print("about to call fake_quant_with_min_max_vars_gradient with invalid min/max shapes")
    try:
        tf.quantization.fake_quant_with_min_max_vars_gradient(
            gradients=1, inputs=1, min=[1, 1], max=[1, 1]
        )
    except Exception as exc:
        return _print_result(False, f"raised Python exception instead of abort: {type(exc).__name__}: {exc}")
    return _print_result(False, "call returned normally")


def main() -> int:
    print("CASE state_bug_004 [tensorflow]")
    print("status=fixed state_dimension=gradient tracking")
    try:
        ok = _case_004()
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
