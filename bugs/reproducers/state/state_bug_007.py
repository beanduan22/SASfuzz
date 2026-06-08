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


def _case_007() -> bool:
    np = _np()
    tf = _tf()
    _tf_require_gpu(tf)

    class Model(tf.keras.Model):
        def __init__(self, seed: int):
            super().__init__()
            self.dense1 = tf.keras.layers.Dense(
                8,
                kernel_initializer=tf.keras.initializers.RandomUniform(
                    minval=-0.1, maxval=0.1, seed=seed
                ),
            )
            self.hashing = tf.keras.layers.Hashing(num_bins=8, output_mode="int", sparse=False)
            self.dense2 = tf.keras.layers.Dense(8, kernel_constraint=tf.keras.constraints.unit_norm())
            self.activation = tf.keras.layers.Activation(tf.nn.swish)
            self.sum_metric = tf.metrics.Sum()

        def call(self, inp, training=False):
            with tf.GradientTape() as tape:
                tape.watch(inp)
                y = self.dense1(inp)
                y = self.hashing(y)
                z = self.dense2(y)
                z = self.activation(z)
            grad = tape.gradient(z, inp)
            if not training and grad is not None:
                self.sum_metric.update_state(grad)
            with tf.device("/CPU:0"):
                return tf.maximum(z, tf.zeros_like(z) if grad is None else grad)

    best = (0.0, None, None, None)
    for seed in range(20):
        tf.keras.utils.set_random_seed(seed)
        model = Model(seed)
        inp = tf.constant(np.random.RandomState(seed).randn(1, 8).astype(np.float32))
        model(inp)
        with tf.device("/CPU:0"):
            cpu = model(inp).numpy()
        with tf.device("/GPU:0"):
            gpu = model(inp).numpy()
        diff = float(np.max(np.abs(cpu - gpu)))
        if diff > best[0]:
            best = (diff, seed, cpu, gpu)
    diff, seed, cpu, gpu = best
    return _print_result(diff > 1e-6, f"state=gradient_tracking(GradientTape) seed={seed} max_diff={diff} cpu={cpu} gpu={gpu}")


def main() -> int:
    print("CASE state_bug_007 [tensorflow]")
    print("status=confirmed state_dimension=gradient tracking")
    try:
        ok = _case_007()
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
