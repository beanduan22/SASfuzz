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

def _arrays_differ(a, b, equal_nan=True, check_sign=False):
    aa = np.asarray(a)
    bb = np.asarray(b)
    if aa.shape != bb.shape:
        return True
    if check_sign and not np.array_equal(np.signbit(aa), np.signbit(bb)):
        return True
    return not np.array_equal(aa, bb, equal_nan=equal_nan)


def run():
    assert tf.config.list_physical_devices('GPU'), 'GPU is required'
    cases = [(float('nan'), tf.int32, 'nan_to_int32'), (float('inf'), tf.int32, 'inf_to_int32'), (float('inf'), tf.int64, 'inf_to_int64')]
    mismatches = []
    for val, dtype, name in cases:

        def op(val=val, dtype=dtype):
            return tf.cast(tf.constant([val], tf.float32), dtype)
        cpu, cpu_err = _tf_device_result('/CPU:0', op, use_strategy=True)
        gpu, gpu_err = _tf_device_result('/GPU:0', op, use_strategy=True)
        mismatch = cpu_err is None and gpu_err is None and _arrays_differ(cpu, gpu)
        mismatches.append(mismatch)
        print(f'{name}: cpu={cpu} gpu={gpu} cpu_err={cpu_err} gpu_err={gpu_err}')
    print('state=distribution_strategy(MirroredStrategy)')
    print('BUG_REPRODUCED' if any(mismatches) else 'NOT_REPRODUCED')
    return


run()
