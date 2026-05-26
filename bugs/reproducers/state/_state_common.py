from __future__ import annotations

import argparse
import dataclasses
import math
import os
import subprocess
import sys
import traceback
import warnings
from typing import Callable, Dict, Iterable, Optional, Tuple


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
warnings.filterwarnings("ignore")


class SkipCase(Exception):
    pass


@dataclasses.dataclass(frozen=True)
class Case:
    key: str
    framework: str
    status: str
    state_dimension: str
    func: Callable[[], bool]
    needs_gpu: bool = False
    fatal_expected: bool = False


def _np():
    try:
        import numpy as np
    except ImportError as exc:
        raise SkipCase(f"missing numpy: {exc}") from exc
    return np


def _tf():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SkipCase(f"missing tensorflow: {exc}") from exc
    return tf


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise SkipCase(f"missing torch: {exc}") from exc
    return torch


def _tf_require_gpu(tf) -> None:
    if not tf.config.list_physical_devices("GPU"):
        raise SkipCase("TensorFlow GPU is not visible")
    try:
        tf.config.set_soft_device_placement(False)
    except Exception:
        pass


def _torch_require_cuda(torch) -> None:
    if not torch.cuda.is_available():
        raise SkipCase("CUDA is not visible to PyTorch")


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


def _print_result(ok: bool, detail: str) -> bool:
    print(detail)
    print("BUG_REPRODUCED" if ok else "NOT_REPRODUCED")
    return bool(ok)


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


def _max_abs_diff(a, b) -> float:
    np = _np()
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        diff = np.abs(aa - bb)
    if diff.size == 0:
        return 0.0
    if np.all(np.isnan(diff)):
        return float("nan")
    return float(np.nanmax(diff))







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


def _case_002() -> bool:
    np = _np()
    tf = _tf()
    _tf_require_gpu(tf)

    cases = [complex(np.inf, np.nan), complex(np.nan, np.inf)]
    mismatches = []
    for value in cases:
        def op(value=value):
            return tf.math.abs(tf.constant([value], tf.complex64))

        cpu, cpu_err = _tf_device_result("/CPU:0", op, use_strategy=True)
        gpu, gpu_err = _tf_device_result("/GPU:0", op, use_strategy=True)
        mismatch = cpu_err is None and gpu_err is None and np.isinf(cpu[0]) and np.isnan(gpu[0])
        mismatches.append(mismatch)
        print(f"value={value!r} cpu={cpu} gpu={gpu} cpu_err={cpu_err} gpu_err={gpu_err}")
    return _print_result(any(mismatches), "state=distribution_strategy(MirroredStrategy)")


def _case_003() -> bool:
    np = _np()
    tf = _tf()
    x = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        dtype=np.float32,
    )
    h, w = x.shape

    def numpy_ref(a):
        b = a.reshape(h, w, 1)
        c = np.concatenate(list(np.split(b, w, axis=1)), axis=0)
        return c.reshape(h, w)

    expected = numpy_ref(x)

    class M(tf.keras.Model):
        @tf.function(input_signature=[tf.TensorSpec([h, w], tf.float32)])
        def call(self, inp):
            a = tf.reshape(inp, [h, w, 1])
            b = tf.concat(tf.unstack(a, axis=1), 0)
            return tf.reshape(b, [h, w])

    model = M()
    keras_out = model(tf.constant(x)).numpy()
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    interpreter = tf.lite.Interpreter(
        model_content=tflite_model,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    in_idx = interpreter.get_input_details()[0]["index"]
    interpreter.resize_tensor_input(in_idx, [h, w])
    interpreter.allocate_tensors()
    interpreter.set_tensor(in_idx, x)
    interpreter.invoke()
    tflite_out = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
    ok = np.allclose(keras_out, expected) and not np.allclose(tflite_out, expected)
    return _print_result(
        ok,
        "state=execution_mode(tf.function/TFLite conversion) "
        f"expected={expected.flatten().tolist()} tflite={tflite_out.flatten().tolist()}",
    )


def _case_004() -> bool:
    tf = _tf()
    print("state=gradient_tracking(fake_quant gradient op)")
    print("about to call fake_quant_with_min_max_vars_gradient with invalid min/max shapes")
    try:
        tf.quantization.fake_quant_with_min_max_vars_gradient(
            gradients=1, inputs=1, min=[1, 1], max=[1, 1]
        )
    except Exception as exc:
        return _print_result(False, f"raised Python exception instead of abort: {type(exc).__name__}: {exc}")
    return _print_result(False, "call returned normally")


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


def _case_006() -> bool:
    np = _np()
    tf = _tf()
    _tf_require_gpu(tf)
    x = tf.constant([[[float("nan"), 0.5, 0.5], [0.5, 0.5, 0.5]]], tf.float32)
    delta = tf.constant(0.5, tf.float32)

    @tf.function
    def op():
        return tf.image.adjust_hue(x, delta)

    cpu, cpu_err = _tf_device_result("/CPU:0", op, use_strategy=True)
    gpu, gpu_err = _tf_device_result("/GPU:0", op, use_strategy=True)
    ok = cpu_err is None and gpu_err is None and np.isnan(cpu).any() and not np.isnan(gpu).any()
    return _print_result(ok, f"state=distribution_strategy(MirroredStrategy) cpu={cpu} cpu_err={cpu_err} gpu={gpu} gpu_err={gpu_err}")


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


def _case_009() -> bool:
    np = _np()
    tf = _tf()
    _tf_require_gpu(tf)
    x = tf.constant([float("nan"), 3.0, 1.0, float("nan"), 2.0, float("nan"), 0.5], tf.float32)

    @tf.function
    def sort_fn(v):
        return tf.sort(v)

    with tf.device("/CPU:0"):
        cpu = sort_fn(x).numpy()
    with tf.device("/GPU:0"):
        gpu = sort_fn(x).numpy()
    cpu_pos = np.where(np.isnan(cpu))[0].tolist()
    gpu_pos = np.where(np.isnan(gpu))[0].tolist()
    return _print_result(cpu_pos != gpu_pos, f"state=execution_mode(tf.function) cpu={cpu.tolist()} gpu={gpu.tolist()}")


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


def _case_011() -> bool:
    np = _np()
    tf = _tf()
    np.random.seed(0)
    x_np = np.random.randn(65536).astype(np.float16)
    ref = float(np.mean(x_np.astype(np.float64)))
    x = tf.constant(x_np)

    @tf.function
    def mean_fn(v):
        return tf.cast(tf.reduce_mean(v), tf.float32)

    with tf.device("/CPU:0"):
        cpu = float(mean_fn(x).numpy())
    gpu = None
    if tf.config.list_physical_devices("GPU"):
        with tf.device("/GPU:0"):
            gpu = float(mean_fn(x).numpy())
    ok = abs(cpu) < 1e-8 and abs(ref) > 1e-5
    return _print_result(ok, f"state=execution_mode(tf.function) ref={ref} cpu={cpu} gpu={gpu}")


def _case_012() -> bool:
    np = _np()
    tf = _tf()
    _tf_require_gpu(tf)
    np.random.seed(0)
    x_np = np.random.randn(10000).astype(np.float32)
    ref = np.cumsum(x_np.astype(np.float64)).astype(np.float32)
    x = tf.constant(x_np, dtype=tf.bfloat16)

    def op():
        return tf.cast(tf.math.cumsum(x), tf.float32)

    cpu, cpu_err = _tf_device_result("/CPU:0", op, use_strategy=True)
    gpu, gpu_err = _tf_device_result("/GPU:0", op, use_strategy=True)
    if cpu_err or gpu_err:
        return _print_result(False, f"state=distribution_strategy(MirroredStrategy) cpu_err={cpu_err} gpu_err={gpu_err}")
    cpu_err_val = float(np.max(np.abs(cpu - ref)))
    gpu_err_val = float(np.max(np.abs(gpu - ref)))
    ok = cpu_err_val > gpu_err_val * 5
    return _print_result(ok, f"state=distribution_strategy(MirroredStrategy) cpu_err={cpu_err_val:.4e} gpu_err={gpu_err_val:.4e}")


def _case_013() -> bool:
    np = _np()
    tf = _tf()

    def test(x, y):
        with tf.GradientTape() as tape:
            tape.watch(x)
            tape.watch(y)
            z = tf.divide(x, y)
        return tape.jacobian(z, x)

    x = tf.constant([1], tf.float32)
    y = tf.constant([0, 1, 2], tf.float32)
    out = test(x, y).numpy()
    expected = np.array([[np.inf], [1.0], [0.5]], dtype=np.float32)
    ok = np.isnan(out[1:]).any() and not np.array_equal(out, expected, equal_nan=True)
    return _print_result(ok, f"state=gradient_tracking(GradientTape.jacobian) out={out.tolist()} expected={expected.tolist()}")


def _case_014() -> bool:
    np = _np()
    tf = _tf()

    def test(x):
        with tf.GradientTape() as tape:
            tape.watch(x)
            w = tf.math.reduce_prod(x)
        return tape.gradient(w, x)

    x = tf.constant([[0, 0.1, 0.2], [0, 0.1, 0.2]], tf.float32)
    out = test(x).numpy()
    expected = np.array([[0.0, 0.2, 0.4], [0.0, 2.0, 4.0]], dtype=np.float32)
    ok = np.all(out == 0) and not np.array_equal(out, expected)
    return _print_result(ok, f"state=gradient_tracking(GradientTape.gradient) out={out.tolist()} expected={expected.tolist()}")


def _case_015() -> bool:
    np = _np()
    tf = _tf()

    def test(a, b):
        with tf.GradientTape() as tape:
            tape.watch(a)
            tape.watch(b)
            w = a * tf.math.reciprocal(b)
        return tape.jacobian(w, a)

    a = tf.constant([3], tf.float32)
    b = tf.constant([0, 2, 3], tf.float32)
    out = test(a, b).numpy()
    expected = np.array([[np.inf], [0.5], [1.0 / 3.0]], dtype=np.float32)
    ok = np.isnan(out[1:]).any() and not np.array_equal(out, expected, equal_nan=True)
    return _print_result(ok, f"state=gradient_tracking(GradientTape.jacobian) out={out.tolist()} expected={expected.tolist()}")


def _case_016() -> bool:
    tf = _tf()

    def log_fn(a):
        return tf.py_function(lambda z: tf.math.log(z), [a], a.dtype)

    def test(a):
        with tf.GradientTape() as tape:
            tape.watch(a)
            y = log_fn(a)
        return tape.jacobian(y, a)

    try:
        out = test(tf.constant([0, 2, 3], tf.float32))
    except Exception as exc:
        msg = type(exc).__name__ + ": " + str(exc)
        ok = "pyfunc" in msg.lower() or "EagerPyFunc".lower() in msg.lower() or "UnknownError" in msg
        return _print_result(ok, f"state=gradient_tracking(GradientTape.jacobian + py_function) err={msg[:240]}")
    return _print_result(False, f"state=gradient_tracking(GradientTape.jacobian + py_function) returned={out}")


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


def _case_020() -> bool:
    return _case_008()


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


def _case_022() -> bool:
    np = _np()
    tf = _tf()
    _tf_require_gpu(tf)
    np.random.seed(0)
    x_np = np.random.randn(1000).astype(np.float32) * 1e4
    x = tf.constant(x_np.astype(np.float16))

    def op():
        return tf.cast(tf.math.reduce_std(x), tf.float32)

    cpu, cpu_err = _tf_device_result("/CPU:0", op, use_strategy=True)
    gpu, gpu_err = _tf_device_result("/GPU:0", op, use_strategy=True)
    ok = cpu_err is None and gpu_err is None and np.isnan(float(cpu)) and np.isinf(float(gpu))
    return _print_result(ok, f"state=distribution_strategy(MirroredStrategy) cpu={cpu}/{cpu_err} gpu={gpu}/{gpu_err}")


def _case_023() -> bool:
    tf = _tf()
    _tf_require_gpu(tf)
    x = tf.ones((3, 1), tf.float32)
    y = tf.ones((1, 5, 2), tf.float32)

    def op():
        return tf.raw_ops.NotEqual(x=x, y=y, incompatible_shape_error=False)

    cpu, cpu_err = _tf_device_result("/CPU:0", op, use_strategy=True)
    gpu, gpu_err = _tf_device_result("/GPU:0", op, use_strategy=True)
    ok = cpu_err is None and bool(cpu) is True and gpu_err is not None
    return _print_result(ok, f"state=distribution_strategy(MirroredStrategy) cpu={cpu}/{cpu_err} gpu={gpu}/{gpu_err}")


def _case_024() -> bool:
    tf = _tf()
    _tf_require_gpu(tf)
    x = tf.constant([-47], tf.int64)
    y = tf.constant([66], tf.int64)

    def op():
        return tf.pow(x, y)

    cpu, cpu_err = _tf_device_result("/CPU:0", op, use_strategy=True)
    gpu, gpu_err = _tf_device_result("/GPU:0", op, use_strategy=True)
    ok = cpu_err is None and gpu_err is None and _arrays_differ(cpu, gpu)
    return _print_result(ok, f"state=distribution_strategy(MirroredStrategy) cpu={cpu} gpu={gpu} cpu_err={cpu_err} gpu_err={gpu_err}")


def _case_025() -> bool:
    tf = _tf()
    _tf_require_gpu(tf)
    x = tf.constant([[2.0, 4.0, 6.0], [4.0, 10.0, 12.0], [6.0, 12.0, 18.0]], tf.float32)

    def op():
        return tf.linalg.slogdet(x)

    cpu, cpu_err = _tf_device_result("/CPU:0", op, use_strategy=True)
    gpu, gpu_err = _tf_device_result("/GPU:0", op, use_strategy=True)
    ok = cpu_err is None and gpu_err is None and _arrays_differ(cpu[1], gpu[1])
    return _print_result(ok, f"state=distribution_strategy(MirroredStrategy) cpu={cpu} gpu={gpu} cpu_err={cpu_err} gpu_err={gpu_err}")


def _case_026() -> bool:
    tf = _tf()
    _tf_require_gpu(tf)
    a = tf.constant([[float("inf")], [2.0]], tf.float32)
    b = tf.constant([[5], [6]], tf.int32)

    def op():
        return tf.sparse.cross([a, b]).values

    cpu, cpu_err = _tf_device_result("/CPU:0", op, use_strategy=True)
    gpu, gpu_err = _tf_device_result("/GPU:0", op, use_strategy=True)
    ok = cpu_err is None and gpu_err is None and _arrays_differ(cpu, gpu)
    return _print_result(ok, f"state=distribution_strategy(MirroredStrategy) cpu={cpu} gpu={gpu} cpu_err={cpu_err} gpu_err={gpu_err}")


def _case_027() -> bool:
    np = _np()
    tf = _tf()
    _tf_require_gpu(tf)
    data = tf.constant([7.0, np.nan, -2.0], tf.float32)
    ids = tf.constant([0, 4, 5], tf.int32)

    def op():
        return tf.math.unsorted_segment_max(data, ids, num_segments=9)

    cpu, cpu_err = _tf_device_result("/CPU:0", op, use_strategy=True)
    gpu, gpu_err = _tf_device_result("/GPU:0", op, use_strategy=True)
    ok = cpu_err is None and gpu_err is None and np.isnan(cpu[4]) and not np.isnan(gpu[4])
    return _print_result(ok, f"state=distribution_strategy(MirroredStrategy) cpu={cpu} gpu={gpu} cpu_err={cpu_err} gpu_err={gpu_err}")


def _case_028() -> bool:
    np = _np()
    tf = _tf()
    _tf_require_gpu(tf)
    np.random.seed(0)
    x_np = np.random.randn(10000).astype(np.float32)
    ref = np.cumsum(x_np.astype(np.float64)).astype(np.float32)
    x = tf.constant(x_np, dtype=tf.bfloat16)

    def op():
        return tf.cast(tf.math.cumsum(x), tf.float32)

    cpu, cpu_err = _tf_device_result("/CPU:0", op, use_strategy=True)
    gpu, gpu_err = _tf_device_result("/GPU:0", op, use_strategy=True)
    if cpu_err or gpu_err:
        return _print_result(False, f"state=distribution_strategy(MirroredStrategy) cpu_err={cpu_err} gpu_err={gpu_err}")
    cpu_err_val = float(np.max(np.abs(cpu - ref)))
    gpu_err_val = float(np.max(np.abs(gpu - ref)))
    ok = cpu_err_val > gpu_err_val * 5
    return _print_result(ok, f"state=distribution_strategy(MirroredStrategy) cpu_err={cpu_err_val:.4e} gpu_err={gpu_err_val:.4e}")


def _case_029() -> bool:
    tf = _tf()
    _tf_require_gpu(tf)

    def op():
        return tf.raw_ops.SparseSegmentSumGradV2(
            grad=tf.constant([1.0, 2.0, 3.0], tf.float64),
            indices=tf.constant([-2], tf.int64),
            segment_ids=tf.constant([-2], tf.int64),
            dense_output_dim0=tf.constant(2, tf.int32),
        )

    cpu, cpu_err = _tf_device_result("/CPU:0", op, use_strategy=True)
    gpu, gpu_err = _tf_device_result("/GPU:0", op, use_strategy=True)
    ok = cpu_err is not None and gpu_err is None
    return _print_result(ok, f"state=distribution_strategy(MirroredStrategy) cpu={cpu}/{cpu_err} gpu={gpu}/{gpu_err}")








def _case_030() -> bool:
    torch = _torch()
    _torch_require_cuda(torch)
    with torch.no_grad():
        cpu = torch.linspace(3.7, -3, 10, dtype=torch.int64, device="cpu")
        gpu = torch.linspace(3.7, -3, 10, dtype=torch.int64, device="cuda").cpu()
    ok = not torch.equal(cpu, gpu)
    return _print_result(ok, f"state=gradient_tracking(torch.no_grad) cpu={cpu.tolist()} gpu={gpu.tolist()}")


def _case_031() -> bool:
    torch = _torch()
    _torch_require_cuda(torch)
    torch.manual_seed(0)
    with torch.no_grad():
        x = torch.randn(1000, dtype=torch.float32) * 1e19 + 1e20
        ref = torch.std(x.double()).item()
        cpu = torch.std(x).item()
        gpu = torch.std(x.cuda()).cpu().item()
    ok = math.isfinite(cpu) and math.isinf(gpu)
    return _print_result(ok, f"state=gradient_tracking(torch.no_grad) ref={ref:.4e} cpu={cpu:.4e} gpu={gpu}")


def _case_032() -> bool:
    torch = _torch()
    _torch_require_cuda(torch)
    with torch.no_grad():
        x_cpu = torch.tensor([1, 2, 3, 4, 5, 6, 7])
        y_cpu = torch.tensor([0, 1, 1, 2])
        z_cpu = torch.tensor([12, 14, 16, 18, 20])
        x_gpu = x_cpu.cuda()
        y_gpu = y_cpu.cuda()
        z_gpu = z_cpu.cuda()
        cpu = torch.Tensor.scatter(x_cpu, 0, y_cpu, z_cpu)
        gpu = torch.Tensor.scatter(x_gpu, 0, y_gpu, z_gpu).cpu()
    ok = not torch.equal(cpu, gpu)
    return _print_result(ok, f"state=gradient_tracking(torch.no_grad) cpu={cpu.tolist()} gpu={gpu.tolist()}")


def _case_033() -> bool:
    np = _np()
    torch = _torch()
    _torch_require_cuda(torch)
    torch.manual_seed(0)
    with torch.no_grad():
        n = 1_000_000
        x = 1.0 + 0.0001 * torch.randn(n, dtype=torch.float32)
        ref = np.cumprod(x.numpy().astype(np.float64))
        cpu = torch.cumprod(x, dim=0).numpy()
        gpu = torch.cumprod(x.cuda(), dim=0).cpu().numpy()
    cpu_err = float(np.max(np.abs(cpu - ref)))
    gpu_err = float(np.max(np.abs(gpu - ref)))
    ok = _max_abs_diff(cpu, gpu) > 1e-3
    return _print_result(ok, f"state=gradient_tracking(torch.no_grad) cpu_err={cpu_err:.4e} gpu_err={gpu_err:.4e}")


def _case_034() -> bool:
    torch = _torch()
    _torch_require_cuda(torch)
    with torch.no_grad():
        lhs = torch.tensor([100000], dtype=torch.int64)
        rhs = torch.tensor([-1], dtype=torch.int64)
        out_cpu = torch.empty(1, dtype=torch.uint8)
        out_cuda = torch.empty(1, dtype=torch.uint8, device="cuda")
        torch.fmax(lhs, rhs, out=out_cpu)
        torch.fmax(lhs.cuda(), rhs.cuda(), out=out_cuda)
        gpu = out_cuda.cpu()
    ok = not torch.equal(out_cpu, gpu)
    return _print_result(ok, f"state=gradient_tracking(torch.no_grad) cpu={out_cpu.tolist()} gpu={gpu.tolist()}")


def _case_035() -> bool:
    np = _np()
    torch = _torch()
    _torch_require_cuda(torch)
    with torch.no_grad():
        mag = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float16)
        sign = torch.from_numpy(np.array([0x7E00, 0xFE00, 0x3C00], dtype=np.uint16).view(np.float16))
        cpu = torch.copysign(mag, sign)
        gpu = torch.copysign(mag.cuda(), sign.cuda()).cpu()
    ok = not torch.equal(torch.signbit(cpu), torch.signbit(gpu))
    return _print_result(ok, f"state=gradient_tracking(torch.no_grad) cpu={cpu} sign={torch.signbit(cpu)} gpu={gpu} sign={torch.signbit(gpu)}")


def _case_036() -> bool:
    torch = _torch()
    _torch_require_cuda(torch)
    import torch.nn as nn
    import torch.nn.functional as F

    class M(nn.Module):
        def forward(self, x):
            return F.relu(x)

    model = M()
    model.train()
    model.eval()
    x_cpu = torch.tensor([-0.0])
    x_gpu = x_cpu.cuda()
    cpu = model(x_cpu)
    gpu = model(x_gpu).cpu()
    ok = bool(torch.signbit(cpu).item()) != bool(torch.signbit(gpu).item())
    return _print_result(ok, f"state=execution_mode(train/eval switch) cpu={cpu} sign={torch.signbit(cpu)} gpu={gpu} sign={torch.signbit(gpu)}")


def _case_037() -> bool:
    torch = _torch()
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(123)
    t, n, c = 3, 1, 3
    log_probs = F.log_softmax(torch.randn(t, n, c), dim=-1).double()
    log_probs.requires_grad_(True)
    targets = torch.tensor([[1]])
    input_lens = torch.tensor([t])
    target_lens = torch.tensor([1])
    loss_fn = nn.CTCLoss(blank=0)

    def fn(inp):
        return loss_fn(inp, targets, input_lens, target_lens)

    try:
        torch.autograd.gradcheck(fn, (log_probs,), raise_exception=True)
    except Exception as exc:
        return _print_result(True, f"state=gradient_tracking(torch.autograd.gradcheck) err={type(exc).__name__}: {str(exc)[:200]}")
    return _print_result(False, "state=gradient_tracking(torch.autograd.gradcheck) gradcheck passed")


def _case_038() -> bool:
    torch = _torch()
    _torch_require_cuda(torch)
    with torch.no_grad():
        x = torch.tensor([-0.0])
        cpu = torch.clamp(x, min=0, max=1)
        gpu = torch.clamp(x.cuda(), min=0, max=1).cpu()
    ok = bool(torch.signbit(cpu).item()) != bool(torch.signbit(gpu).item())
    return _print_result(ok, f"state=gradient_tracking(torch.no_grad) cpu={cpu} sign={torch.signbit(cpu)} gpu={gpu} sign={torch.signbit(gpu)}")


def _case_039() -> bool:
    np = _np()
    torch = _torch()
    _torch_require_cuda(torch)
    src = torch.from_numpy(np.array([0xFE00, 0xFE00, 0xFE00], dtype=np.uint16).view(np.float16))
    with torch.no_grad():
        cpu = torch.signbit(src)
        gpu = torch.signbit(src.cuda()).cpu()
    ok = not torch.equal(cpu, gpu)
    return _print_result(ok, f"state=gradient_tracking(torch.no_grad) cpu={cpu.tolist()} gpu={gpu.tolist()}")


def _case_040() -> bool:
    torch = _torch()
    _torch_require_cuda(torch)
    import os
    import tempfile
    import torch.nn as nn
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    if not dist.is_available():
        raise SkipCase("torch.distributed is not available")
    if not dist.is_nccl_available():
        raise SkipCase("NCCL is not available for CUDA DDP")

    torch.manual_seed(202311)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(()))

        def forward(self, inp):
            return torch.linalg.pinv(inp * self.scale)

    def init_group() -> None:
        if dist.is_initialized():
            return
        init_dir = tempfile.mkdtemp(prefix="smolfuzz-ddp-")
        init_file = os.path.join(init_dir, "init")
        dist.init_process_group(
            backend="nccl",
            init_method=f"file://{init_file}",
            rank=0,
            world_size=1,
        )



    base = -torch.ones(2, 3, 8, 8)
    noise = 0.25 * torch.randn(2, 3, 8, 8)
    x_cpu = base + noise

    cpu = Model()(x_cpu).detach()

    init_group()
    torch.cuda.set_device(0)
    model = Model().cuda()
    ddp = DDP(model, device_ids=[0])
    x_gpu = x_cpu.cuda().requires_grad_(True)
    gpu_out = ddp(x_gpu)
    gpu_out.sum().backward()
    gpu = gpu_out.detach().cpu()

    diff = float(torch.nan_to_num((cpu - gpu).abs(), nan=0.0, posinf=1e30, neginf=1e30).max().item())
    ok = diff > 1e-3
    return _print_result(ok, f"state=distribution_strategy(DDP) max_diff={diff:.4e}")


def _case_041() -> bool:
    torch = _torch()
    _torch_require_cuda(torch)
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.avg_pool1d = nn.AvgPool1d(kernel_size=2)
            self.channel_shuffle = nn.ChannelShuffle(groups=3)

        def forward(self, x):
            x = x.view(-1, 3, 32 * 32)
            x = self.avg_pool1d(x)
            return self.channel_shuffle(x)

    model = Model()
    model.train()
    model.eval()
    x = torch.rand(2, 3, 32, 32)
    cpu_err = gpu_err = None
    try:
        cpu = model.cpu()(x)
    except Exception as exc:
        cpu = None
        cpu_err = type(exc).__name__ + ": " + str(exc)[:120]
    try:
        gpu = model.cuda()(x.cuda())
    except Exception as exc:
        gpu = None
        gpu_err = type(exc).__name__ + ": " + str(exc)[:160]
    ok = cpu is not None and gpu_err is not None and "channel_shuffle" in gpu_err
    return _print_result(ok, f"state=execution_mode(train/eval switch) cpu_shape={None if cpu is None else tuple(cpu.shape)} cpu_err={cpu_err} gpu_err={gpu_err}")


def _case_042() -> bool:
    torch = _torch()
    _torch_require_cuda(torch)
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.LazyBatchNorm1d()

        def forward(self, x):
            return self.bn(x.view(x.size(0), -1))

    torch.manual_seed(0)
    x = torch.rand(2, 3, 32, 32)
    cpu_model = Model().train()
    gpu_model = Model().cuda().train()
    cpu = gpu = None
    cpu_err = gpu_err = None
    try:
        cpu = cpu_model(x)
    except Exception as exc:
        cpu_err = type(exc).__name__ + ": " + str(exc)[:160]
    try:
        gpu = gpu_model(x.cuda()).cpu()
    except Exception as exc:
        gpu_err = type(exc).__name__ + ": " + str(exc)[:160]
    if cpu_err or gpu_err:
        ok = (cpu_err is None) != (gpu_err is None)
        return _print_result(ok, f"state=execution_mode(train/eval switch) cpu_err={cpu_err} gpu_err={gpu_err}")
    close = torch.allclose(cpu, gpu, atol=1e-6, rtol=1e-6, equal_nan=True)
    diff = float(torch.nan_to_num((cpu - gpu).abs(), nan=0.0).max().item())
    return _print_result(not close, f"state=execution_mode(train/eval switch) allclose={close} max_diff={diff:.4e}")


def _case_043() -> bool:
    torch = _torch()
    _torch_require_cuda(torch)
    import torch.nn as nn

    class Model(nn.Module):
        def forward(self, x):
            y = x.view(x.size(0), -1)
            return torch.matrix_exp(y)

    inp = torch.tensor([[1.0, 1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 10.0, 200.0]])
    model = Model()
    traced_cpu = torch.jit.trace(model, inp)
    traced_gpu = torch.jit.trace(model.cuda(), inp.cuda())
    cpu = traced_cpu(inp)
    gpu = traced_gpu(inp.cuda()).cpu()
    ok = not torch.equal(torch.isnan(cpu), torch.isnan(gpu)) or not torch.equal(torch.isinf(cpu), torch.isinf(gpu))
    return _print_result(ok, f"state=execution_mode(torch.jit.trace) cpu={cpu} gpu={gpu}")


def _case_044() -> bool:
    torch = _torch()
    _torch_require_cuda(torch)
    x = torch.tensor(
        [
            [0.0100, 0.0000, 0.0000, 0.0000, 0.1000],
            [0.0000, 0.0100, 0.0000, 0.1000, 0.0000],
            [0.0000, 0.0000, 0.0100, 0.0000, 0.0000],
            [0.0000, 0.1000, 0.0000, 0.0100, 0.0000],
            [0.1000, 0.0000, 0.0000, 0.0000, 0.0100],
        ]
    )
    with torch.no_grad():
        cpu_vals, cpu_vecs = torch.lobpcg(x)
        gpu_vals, gpu_vecs = torch.lobpcg(x.cuda())
    diff = float((cpu_vecs.abs() - gpu_vecs.cpu().abs()).abs().max().item())
    ok = diff > 1e-2
    return _print_result(ok, f"state=gradient_tracking(torch.no_grad) eig_cpu={cpu_vals} eig_gpu={gpu_vals.cpu()} vec_abs_diff={diff:.4e}")


def _case_045() -> bool:
    torch = _torch()
    _torch_require_cuda(torch)
    torch.manual_seed(0)
    x = torch.randn(32, 32)
    with torch.no_grad():
        x_cpu = x.clone()
        x_gpu = x.clone().cuda()
        cpu = torch.mm(x_cpu, x_cpu, out=x_cpu)
        gpu = torch.mm(x_gpu, x_gpu, out=x_gpu).cpu()
    close = torch.allclose(cpu, gpu, atol=1e-5, rtol=1e-5, equal_nan=True)
    diff = float((cpu - gpu).abs().max().item())
    return _print_result(not close, f"state=gradient_tracking(torch.no_grad) allclose={close} max_diff={diff:.4e}")


def _case_046() -> bool:
    torch = _torch()
    import torch.nn as nn

    torch.manual_seed(0)
    fc1 = nn.Linear(8, 8)
    fc2 = nn.Linear(8, 8)
    bn = nn.BatchNorm1d(8)
    x = torch.randn(4, 8, requires_grad=True)
    with torch.enable_grad():
        y = fc1(x)
        y = torch.nn.functional.hardswish(y)
        y = bn(y)
        y = fc2(y)
        y = torch.log1p(y)
        out = torch.xlogy(y, y)
    minimal = torch.xlogy(torch.tensor(0.0), torch.tensor(0.0))
    ok = torch.isnan(minimal).item() or torch.isnan(out).any().item()
    return _print_result(ok, f"state=gradient_tracking(torch.enable_grad) minimal_xlogy_0_0={minimal} any_model_nan={torch.isnan(out).any().item()}")


def _case_047() -> bool:
    return _case_036()



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

def _register() -> Dict[str, Case]:
    cases = [
        Case('state_bug_001', 'tensorflow', 'fixed', 'execution mode', _case_001),
        Case('state_bug_002', 'tensorflow', 'fixed', 'distribution strategy', _case_002, needs_gpu=True),
        Case('state_bug_003', 'tensorflow', 'fixed', 'execution mode', _case_003),
        Case('state_bug_004', 'tensorflow', 'fixed', 'gradient tracking', _case_004, fatal_expected=True),
        Case('state_bug_005', 'tensorflow', 'confirmed', 'execution mode', _case_005),
        Case('state_bug_006', 'tensorflow', 'confirmed', 'distribution strategy', _case_006, needs_gpu=True),
        Case('state_bug_007', 'tensorflow', 'confirmed', 'gradient tracking', _case_007, needs_gpu=True),
        Case('state_bug_008', 'tensorflow', 'confirmed', 'distribution strategy', _case_008, needs_gpu=True),
        Case('state_bug_009', 'tensorflow', 'confirmed', 'execution mode', _case_009, needs_gpu=True),
        Case('state_bug_010', 'tensorflow', 'confirmed', 'execution mode', _case_010, needs_gpu=True),
        Case('state_bug_011', 'tensorflow', 'confirmed', 'execution mode', _case_011),
        Case('state_bug_012', 'tensorflow', 'confirmed', 'distribution strategy', _case_012, needs_gpu=True),
        Case('state_bug_013', 'tensorflow', 'confirmed', 'gradient tracking', _case_013),
        Case('state_bug_014', 'tensorflow', 'confirmed', 'gradient tracking', _case_014),
        Case('state_bug_015', 'tensorflow', 'confirmed', 'gradient tracking', _case_015),
        Case('state_bug_016', 'tensorflow', 'confirmed', 'gradient tracking', _case_016, fatal_expected=True),
        Case('state_bug_017', 'tensorflow', 'confirmed', 'execution mode', _case_017),
        Case('state_bug_018', 'tensorflow', 'confirmed', 'execution mode', _case_018),
        Case('state_bug_019', 'tensorflow', 'confirmed', 'execution mode', _case_019),
        Case('state_bug_020', 'tensorflow', 'confirmed', 'distribution strategy', _case_020, needs_gpu=True),
        Case('state_bug_021', 'tensorflow', 'confirmed', 'distribution strategy', _case_021, needs_gpu=True),
        Case('state_bug_022', 'tensorflow', 'confirmed', 'distribution strategy', _case_022, needs_gpu=True),
        Case('state_bug_023', 'tensorflow', 'confirmed', 'distribution strategy', _case_023, needs_gpu=True),
        Case('state_bug_024', 'tensorflow', 'confirmed', 'distribution strategy', _case_024, needs_gpu=True),
        Case('state_bug_025', 'tensorflow', 'confirmed', 'distribution strategy', _case_025, needs_gpu=True),
        Case('state_bug_026', 'tensorflow', 'confirmed', 'distribution strategy', _case_026, needs_gpu=True),
        Case('state_bug_027', 'tensorflow', 'confirmed', 'distribution strategy', _case_027, needs_gpu=True),
        Case('state_bug_028', 'tensorflow', 'confirmed', 'distribution strategy', _case_028, needs_gpu=True),
        Case('state_bug_029', 'tensorflow', 'confirmed', 'distribution strategy', _case_029, needs_gpu=True, fatal_expected=True),
        Case('state_bug_030', 'pytorch', 'fixed', 'gradient tracking', _case_030, needs_gpu=True),
        Case('state_bug_031', 'pytorch', 'fixed', 'gradient tracking', _case_031, needs_gpu=True),
        Case('state_bug_032', 'pytorch', 'fixed', 'gradient tracking', _case_032, needs_gpu=True),
        Case('state_bug_033', 'pytorch', 'fixed', 'gradient tracking', _case_033, needs_gpu=True),
        Case('state_bug_034', 'pytorch', 'fixed', 'gradient tracking', _case_034, needs_gpu=True),
        Case('state_bug_035', 'pytorch', 'fixed', 'gradient tracking', _case_035, needs_gpu=True),
        Case('state_bug_036', 'pytorch', 'fixed', 'execution mode', _case_036, needs_gpu=True),
        Case('state_bug_037', 'pytorch', 'fixed', 'gradient tracking', _case_037),
        Case('state_bug_038', 'pytorch', 'fixed', 'gradient tracking', _case_038, needs_gpu=True),
        Case('state_bug_039', 'pytorch', 'fixed', 'gradient tracking', _case_039, needs_gpu=True),
        Case('state_bug_040', 'pytorch', 'fixed', 'distribution strategy', _case_040, needs_gpu=True),
        Case('state_bug_041', 'pytorch', 'fixed', 'execution mode', _case_041, needs_gpu=True),
        Case('state_bug_042', 'pytorch', 'confirmed', 'execution mode', _case_042, needs_gpu=True),
        Case('state_bug_043', 'pytorch', 'confirmed', 'execution mode', _case_043, needs_gpu=True),
        Case('state_bug_044', 'pytorch', 'confirmed', 'gradient tracking', _case_044, needs_gpu=True),
        Case('state_bug_045', 'pytorch', 'confirmed', 'gradient tracking', _case_045, needs_gpu=True),
        Case('state_bug_046', 'pytorch', 'confirmed', 'gradient tracking', _case_046),
        Case('state_bug_047', 'pytorch', 'confirmed', 'execution mode', _case_047, needs_gpu=True),
        Case('state_bug_048', 'tensorflow', 'confirmed', 'distribution strategy', _case_048, needs_gpu=True),
    ]
    return {case.key: case for case in cases}



CASES = _register()


def run_one(case: Case) -> int:
    print(f"CASE {case.key} [{case.framework}]")
    print(f"status={case.status} state_dimension={case.state_dimension}")
    try:
        ok = case.func()
        return 0 if ok else 1
    except SkipCase as exc:
        print(f"SKIPPED: {exc}")
        return 2
    except Exception:
        print("HARNESS_ERROR:")
        traceback.print_exc()
        return 3


def _iter_cases(keys: Optional[Iterable[str]]) -> Iterable[Case]:
    if keys:
        for key in keys:
            if key not in CASES:
                raise SystemExit(f"unknown case: {key}")
            yield CASES[key]
    else:
        yield from CASES.values()


def run_all(keys: Optional[Iterable[str]], timeout: int) -> int:
    selected = list(_iter_cases(keys))
    summary = []
    for case in selected:
        cmd = [sys.executable, os.path.abspath(__file__), "--case", case.key, "--child"]
        print(f"\n===== {case.key} ({case.state_dimension}) =====")
        try:
            proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            print(f"TIMEOUT after {timeout}s")
            summary.append((case.key, "TIMEOUT"))
            if exc.stdout:
                print(exc.stdout)
            if exc.stderr:
                print(exc.stderr, file=sys.stderr)
            continue
        if proc.stdout:
            print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
        if proc.stderr:
            print(proc.stderr, file=sys.stderr, end="" if proc.stderr.endswith("\n") else "\n")
        if proc.returncode == 0:
            status = "REPRO"
        elif proc.returncode == 2:
            status = "SKIP"
        elif proc.returncode < 0 and case.fatal_expected:
            status = f"REPRO_FATAL(signal={-proc.returncode})"
        elif proc.returncode >= 128 and case.fatal_expected:
            status = f"REPRO_FATAL(exit={proc.returncode})"
        elif proc.returncode == 1:
            status = "NOT_REPRO"
        else:
            status = f"HARNESS_ERROR(exit={proc.returncode})"
        print(f"RESULT {case.key}: {status}")
        summary.append((case.key, status))
    print("\n===== SUMMARY =====")
    for key, status in summary:
        print(f"{key:10s} {status}")
    return 0 if any(status.startswith("REPRO") for _, status in summary) else 1


def list_cases() -> None:
    for case in CASES.values():
        gpu = " gpu" if case.needs_gpu else ""
        fatal = " fatal" if case.fatal_expected else ""
        print(
            f"{case.key:13s} {case.framework:10s} "
            f"{case.status:9s} {case.state_dimension:22s}{gpu}{fatal}"
        )


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list available repro cases")
    parser.add_argument("--case", action="append", dest="cases", help="run a specific case key, e.g. state_bug_001")
    parser.add_argument("--all", action="store_true", help="run all cases in isolated child processes")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--timeout", type=int, default=120, help="per-case timeout when running --all")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.list:
        list_cases()
        return 0
    if args.child:
        if not args.cases or len(args.cases) != 1:
            raise SystemExit("--child requires exactly one --case")
        return run_one(CASES[args.cases[0]])
    if args.cases and not args.all:
        if len(args.cases) == 1:
            return run_one(CASES[args.cases[0]])
        return run_all(args.cases, args.timeout)
    return run_all(args.cases, args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
