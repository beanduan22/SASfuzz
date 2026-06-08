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


def _arrays_differ(a, b, *, equal_nan: bool = True, check_sign: bool = False) -> bool:
    np = _np()
    aa = np.asarray(a)
    bb = np.asarray(b)
    if aa.shape != bb.shape:
        return True
    if check_sign:
        if not np.array_equal(np.signbit(aa), np.signbit(bb)):
            return True
    return not np.array_equal(aa, bb, equal_nan=equal_nan)


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


def _to_numpy(value):
    if hasattr(value, "numpy"):
        return value.numpy()
    if hasattr(value, "values") and hasattr(value, "indices"):
        return (_to_numpy(value.values), _to_numpy(value.indices))
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return tuple(_to_numpy(v) for v in value)
    if isinstance(value, (tuple, list)):
        return type(value)(_to_numpy(v) for v in value)
    return value


def _tf_device_result(device: str, fn: Callable[[], object], use_strategy: bool = False):
    tf = _tf()
    if "GPU" in device.upper():
        _tf_require_gpu(tf)
    try:
        if use_strategy:
            strategy = tf.distribute.MirroredStrategy(devices=[device])
            with strategy.scope():
                out = strategy.run(fn)
        else:
            with tf.device(device):
                out = fn()
        return _to_numpy(out), None
    except Exception as exc:
        return None, type(exc).__name__ + ": " + str(exc).splitlines()[0][:160]


def _case_008() -> bool:
    tf = _tf()
    _tf_require_gpu(tf)
    cases = [
        (float("nan"), tf.int32, "nan_to_int32"),
        (float("inf"), tf.int32, "inf_to_int32"),
        (float("inf"), tf.int64, "inf_to_int64"),
    ]
    mismatches = []
    for val, dtype, name in cases:
        def op(val=val, dtype=dtype):
            return tf.cast(tf.constant([val], tf.float32), dtype)

        cpu, cpu_err = _tf_device_result("/CPU:0", op, use_strategy=True)
        gpu, gpu_err = _tf_device_result("/GPU:0", op, use_strategy=True)
        mismatch = cpu_err is None and gpu_err is None and _arrays_differ(cpu, gpu)
        mismatches.append(mismatch)
        print(f"{name}: cpu={cpu} gpu={gpu} cpu_err={cpu_err} gpu_err={gpu_err}")
    return _print_result(any(mismatches), "state=distribution_strategy(MirroredStrategy)")


def _case_020() -> bool:
    return _case_008()


def main() -> int:
    print("CASE state_bug_020 [tensorflow]")
    print("status=confirmed state_dimension=distribution strategy")
    try:
        ok = _case_020()
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
