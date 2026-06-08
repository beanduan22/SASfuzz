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


def _case_005() -> bool:
    tf = _tf()
    x = tf.constant([[3.0, -2.0, -7.0, 4.0, -1.0]], tf.float32)
    eager = tf.argmin(tf.nn.relu(x), axis=-1).numpy().tolist()
    tf.config.optimizer.set_experimental_options({"arithmetic_optimization": True})

    @tf.function
    def arith_on(v):
        return tf.argmin(tf.nn.relu(v), axis=-1)

    on = arith_on(x).numpy().tolist()
    tf.config.optimizer.set_experimental_options({"arithmetic_optimization": False})

    @tf.function
    def arith_off(v):
        return tf.argmin(tf.nn.relu(v), axis=-1)

    off = arith_off(x).numpy().tolist()
    ok = eager == off and on != off
    return _print_result(ok, f"state=execution_mode(tf.function optimizer) eager={eager} on={on} off={off}")


def main() -> int:
    print("CASE state_bug_005 [tensorflow]")
    print("status=confirmed state_dimension=execution mode")
    try:
        ok = _case_005()
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
