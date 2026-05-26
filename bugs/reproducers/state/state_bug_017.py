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


def _case_017() -> bool:
    tf = _tf()
    a = tf.constant([0, -2, 1, -4, 3])

    @tf.function
    def graph_negative_topk(v):
        return tf.negative(tf.math.top_k(tf.negative(v)))

    y = graph_negative_topk(a)
    unary_err = None
    try:
        _ = -tf.math.top_k(-a)
    except Exception as exc:
        unary_err = type(exc).__name__ + ": " + str(exc)
    ok = unary_err is not None
    return _print_result(ok, f"state=execution_mode(tf.function) tf.negative(top_k)={y} unary_minus_err={unary_err}")


def main() -> int:
    print("CASE state_bug_017 [tensorflow]")
    print("status=confirmed state_dimension=execution mode")
    try:
        ok = _case_017()
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
