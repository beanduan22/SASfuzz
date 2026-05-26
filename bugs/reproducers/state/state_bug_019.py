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


def _case_019() -> bool:
    np = _np()
    tf = _tf()
    x = tf.constant(np.arange(12, dtype=np.float32).reshape(1, 2, 3, 2))

    class ModelEager(tf.keras.Model):
        @tf.function
        def call(self, inp):
            y = tf.slice(inp, [0, 0, 1, 0], [-1, -1, 1, -1])
            _ = tf.slice(inp, [0, 0, 0, 0], [-1, -1, 5, -1])
            return y

    class ModelXLA(tf.keras.Model):
        @tf.function(jit_compile=True)
        def call(self, inp):
            y = tf.slice(inp, [0, 0, 1, 0], [-1, -1, 1, -1])
            _ = tf.slice(inp, [0, 0, 0, 0], [-1, -1, 5, -1])
            return y

    def run(cls):
        try:
            out = cls()(x).numpy()
            return out.shape, None
        except Exception as exc:
            return None, type(exc).__name__ + ": " + str(exc).splitlines()[0][:120]

    eager_shape, eager_err = run(ModelEager)
    xla_shape, xla_err = run(ModelXLA)
    ok = eager_shape is not None and xla_err is not None
    return _print_result(ok, f"state=execution_mode(tf.function jit_compile=True) eager={eager_shape}/{eager_err} xla={xla_shape}/{xla_err}")


def main() -> int:
    print("CASE state_bug_019 [tensorflow]")
    print("status=confirmed state_dimension=execution mode")
    try:
        ok = _case_019()
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
