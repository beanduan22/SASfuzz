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


def _tf():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SkipCase(f"missing tensorflow: {exc}") from exc
    return tf


def _case_013() -> bool:
    np = _np()
    tf = _tf()

    def test(x, y):
        with tf.GradientTape() as tape:
            tape.watch(x)
            tape.watch(y)
            z = tf.divide(x, y)
        return tape.jacobian(z, x)

    x = tf.constant([1], tf.float32)
    y = tf.constant([0, 1, 2], tf.float32)
    out = test(x, y).numpy()
    expected = np.array([[np.inf], [1.0], [0.5]], dtype=np.float32)
    ok = np.isnan(out[1:]).any() and not np.array_equal(out, expected, equal_nan=True)
    return _print_result(ok, f"state=gradient_tracking(GradientTape.jacobian) out={out.tolist()} expected={expected.tolist()}")


def main() -> int:
    print("CASE state_bug_013 [tensorflow]")
    print("status=confirmed state_dimension=gradient tracking")
    try:
        ok = _case_013()
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
