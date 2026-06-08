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


def _tf_require_gpu(tf) -> None:
    if not tf.config.list_physical_devices("GPU"):
        raise SkipCase("TensorFlow GPU is not visible")
    try:
        tf.config.set_soft_device_placement(False)
    except Exception:
        pass


def _case_010() -> bool:
    np = _np()
    tf = _tf()
    _tf_require_gpu(tf)
    np.random.seed(0)
    x_np = np.random.randn(1000).astype(np.float32) * 1e4
    x = tf.constant(x_np.astype(np.float16))

    @tf.function
    def std_fn(v):
        return tf.cast(tf.math.reduce_std(v), tf.float32)

    with tf.device("/CPU:0"):
        cpu = float(std_fn(x).numpy())
    with tf.device("/GPU:0"):
        gpu = float(std_fn(x).numpy())
    ok = np.isnan(cpu) and np.isinf(gpu)
    return _print_result(ok, f"state=execution_mode(tf.function) cpu={cpu} gpu={gpu}")


def main() -> int:
    print("CASE state_bug_010 [tensorflow]")
    print("status=confirmed state_dimension=execution mode")
    try:
        ok = _case_010()
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
