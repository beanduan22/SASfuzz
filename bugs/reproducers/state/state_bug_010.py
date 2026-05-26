import math
import os
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")
import tensorflow as tf
import numpy as np


def run():
    assert tf.config.list_physical_devices('GPU'), 'GPU is required'
    np.random.seed(0)
    x_np = np.random.randn(1000).astype(np.float32) * 10000.0
    x = tf.constant(x_np.astype(np.float16))

    @tf.function
    def std_fn(v):
        return tf.cast(tf.math.reduce_std(v), tf.float32)
    with tf.device('/CPU:0'):
        cpu = float(std_fn(x).numpy())
    with tf.device('/GPU:0'):
        gpu = float(std_fn(x).numpy())
    ok = np.isnan(cpu) and np.isinf(gpu)
    print(f'state=execution_mode(tf.function) cpu={cpu} gpu={gpu}')
    print('BUG_REPRODUCED' if ok else 'NOT_REPRODUCED')
    return


run()
