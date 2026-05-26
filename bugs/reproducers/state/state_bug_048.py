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


def _case_048() -> bool:
    tf = _tf()
    np = _np()
    _tf_require_gpu(tf)
    x = tf.constant([-0.0, -0.0], tf.float64)
    with tf.device("/CPU:0"):
        cpu = tf.clip_by_value(x, 0.0, 2.0).numpy()
    with tf.device("/GPU:0"):
        gpu = tf.clip_by_value(x, 0.0, 2.0).numpy()
    cpu_sign = np.signbit(cpu)
    gpu_sign = np.signbit(gpu)
    ok = not np.array_equal(cpu_sign, gpu_sign)
    return _print_result(
        ok,
        f"state=distribution_strategy(device placement) cpu={cpu.tolist()} gpu={gpu.tolist()} "
        f"cpu_sign={cpu_sign.tolist()} gpu_sign={gpu_sign.tolist()}",
    )


def main() -> int:
    print("CASE state_bug_048 [tensorflow]")
    print("status=confirmed state_dimension=distribution strategy")
    try:
        ok = _case_048()
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
