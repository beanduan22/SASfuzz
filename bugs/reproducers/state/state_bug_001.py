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


def _case_001() -> bool:
    np = _np()
    tf = _tf()
    a = tf.ones((8, 8), tf.float32)

    @tf.function
    def logdet_fn(x):
        return tf.linalg.logdet(x)

    with tf.device("/CPU:0"):
        cpu = float(logdet_fn(a).numpy())
    expected = float(np.linalg.slogdet(np.ones((8, 8), dtype=np.float32))[1])
    gpu = None
    if tf.config.list_physical_devices("GPU"):
        with tf.device("/GPU:0"):
            gpu = float(logdet_fn(a).numpy())
    ok = math.isnan(cpu) and math.isinf(expected) and expected < 0
    if gpu is not None:
        ok = ok or math.isnan(gpu)
    return _print_result(ok, f"state=execution_mode(tf.function) cpu={cpu} gpu={gpu} expected={expected}")


def main() -> int:
    print("CASE state_bug_001 [tensorflow]")
    print("status=fixed state_dimension=execution mode")
    try:
        ok = _case_001()
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
