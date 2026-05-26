import math
import os
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")
import tensorflow as tf
import numpy as np


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

def _tf_device_result(device, fn, use_strategy=False):
    if "GPU" in device.upper():
        assert tf.config.list_physical_devices("GPU"), "GPU is required"
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


def run():
    assert tf.config.list_physical_devices('GPU'), 'GPU is required'
    np.random.seed(0)
    x_np = np.random.randn(10000).astype(np.float32)
    ref = np.cumsum(x_np.astype(np.float64)).astype(np.float32)
    x = tf.constant(x_np, dtype=tf.bfloat16)

    def op():
        return tf.cast(tf.math.cumsum(x), tf.float32)
    cpu, cpu_err = _tf_device_result('/CPU:0', op, use_strategy=True)
    gpu, gpu_err = _tf_device_result('/GPU:0', op, use_strategy=True)
    if cpu_err or gpu_err:
        print(f'state=distribution_strategy(MirroredStrategy) cpu_err={cpu_err} gpu_err={gpu_err}')
        print('BUG_REPRODUCED' if False else 'NOT_REPRODUCED')
        return
    cpu_err_val = float(np.max(np.abs(cpu - ref)))
    gpu_err_val = float(np.max(np.abs(gpu - ref)))
    ok = cpu_err_val > gpu_err_val * 5
    print(f'state=distribution_strategy(MirroredStrategy) cpu_err={cpu_err_val:.4e} gpu_err={gpu_err_val:.4e}')
    print('BUG_REPRODUCED' if ok else 'NOT_REPRODUCED')
    return


run()
