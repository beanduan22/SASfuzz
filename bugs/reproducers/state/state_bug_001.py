import math
import os
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")
import tensorflow as tf
import numpy as np


def run():
    a = tf.ones((8, 8), tf.float32)

    @tf.function
    def logdet_fn(x):
        return tf.linalg.logdet(x)
    with tf.device('/CPU:0'):
        cpu = float(logdet_fn(a).numpy())
    expected = float(np.linalg.slogdet(np.ones((8, 8), dtype=np.float32))[1])
    gpu = None
    if tf.config.list_physical_devices('GPU'):
        with tf.device('/GPU:0'):
            gpu = float(logdet_fn(a).numpy())
    ok = math.isnan(cpu) and math.isinf(expected) and (expected < 0)
    if gpu is not None:
        ok = ok or math.isnan(gpu)
    print(f'state=execution_mode(tf.function) cpu={cpu} gpu={gpu} expected={expected}')
    print('BUG_REPRODUCED' if ok else 'NOT_REPRODUCED')
    return


run()
