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


def _case_021() -> bool:
    tf = _tf()
    _tf_require_gpu(tf)

    def op():
        return tf.raw_ops.SparseFillEmptyRowsGrad(
            reverse_index_map=tf.constant([20, 21], tf.int64),
            grad_values=tf.constant([3, 4, 5], tf.int64),
        )

    cpu, cpu_err = _tf_device_result("/CPU:0", op, use_strategy=True)
    gpu, gpu_err = _tf_device_result("/GPU:0", op, use_strategy=True)
    ok = cpu_err is not None and gpu_err is None
    return _print_result(ok, f"state=distribution_strategy(MirroredStrategy) cpu={cpu}/{cpu_err} gpu={gpu}/{gpu_err}")


def main() -> int:
    print("CASE state_bug_021 [tensorflow]")
    print("status=confirmed state_dimension=distribution strategy")
    try:
        ok = _case_021()
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
