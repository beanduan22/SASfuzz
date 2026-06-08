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


def _case_016() -> bool:
    tf = _tf()

    def log_fn(a):
        return tf.py_function(lambda z: tf.math.log(z), [a], a.dtype)

    def test(a):
        with tf.GradientTape() as tape:
            tape.watch(a)
            y = log_fn(a)
        return tape.jacobian(y, a)

    try:
        out = test(tf.constant([0, 2, 3], tf.float32))
    except Exception as exc:
        msg = type(exc).__name__ + ": " + str(exc)
        ok = "pyfunc" in msg.lower() or "EagerPyFunc".lower() in msg.lower() or "UnknownError" in msg
        return _print_result(ok, f"state=gradient_tracking(GradientTape.jacobian + py_function) err={msg[:240]}")
    return _print_result(False, f"state=gradient_tracking(GradientTape.jacobian + py_function) returned={out}")


def main() -> int:
    print("CASE state_bug_016 [tensorflow]")
    print("status=confirmed state_dimension=gradient tracking")
    try:
        ok = _case_016()
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
