import os
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")
import tensorflow as tf


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
    x = tf.ones((3, 1), tf.float32)
    y = tf.ones((1, 5, 2), tf.float32)

    def op():
        return tf.raw_ops.NotEqual(x=x, y=y, incompatible_shape_error=False)
    cpu, cpu_err = _tf_device_result('/CPU:0', op, use_strategy=True)
    gpu, gpu_err = _tf_device_result('/GPU:0', op, use_strategy=True)
    ok = cpu_err is None and bool(cpu) is True and (gpu_err is not None)
    print(f'state=distribution_strategy(MirroredStrategy) cpu={cpu}/{cpu_err} gpu={gpu}/{gpu_err}')
    print('BUG_REPRODUCED' if ok else 'NOT_REPRODUCED')
    return


run()
