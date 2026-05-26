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


def _case_018() -> bool:
    tf = _tf()
    w_init = tf.constant([[0.1], [0.2], [0.3], [0.4], [0.5], [0.6]], tf.float32)
    x = tf.constant([[2.0, 4.0, 6.0, 8.0]], tf.float32, shape=[1, 4])

    class ModelEager(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.w = tf.Variable(w_init, shape=tf.TensorShape(None), dtype=tf.float32)

        @tf.function
        def call(self, inp):
            return tf.matmul(inp, self.w)

    class ModelXLA(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.w = tf.Variable(w_init, shape=tf.TensorShape(None), dtype=tf.float32)

        @tf.function(jit_compile=True)
        def call(self, inp):
            return tf.matmul(inp, self.w)

    def run(cls):
        try:
            return cls()(x).numpy(), None
        except Exception as exc:
            return None, type(exc).__name__ + ": " + str(exc).splitlines()[0][:120]

    eager_out, eager_err = run(ModelEager)
    xla_out, xla_err = run(ModelXLA)
    ok = eager_err is not None and xla_out is not None
    return _print_result(ok, f"state=execution_mode(tf.function jit_compile=True) eager={eager_out}/{eager_err} xla={xla_out}/{xla_err}")


def main() -> int:
    print("CASE state_bug_018 [tensorflow]")
    print("status=confirmed state_dimension=execution mode")
    try:
        ok = _case_018()
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
