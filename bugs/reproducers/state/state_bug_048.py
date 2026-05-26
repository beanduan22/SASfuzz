import os
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")
import tensorflow as tf
import numpy as np


def run():
    assert tf.config.list_physical_devices('GPU'), 'GPU is required'
    x = tf.constant([-0.0, -0.0], tf.float64)
    with tf.device('/CPU:0'):
        cpu = tf.clip_by_value(x, 0.0, 2.0).numpy()
    with tf.device('/GPU:0'):
        gpu = tf.clip_by_value(x, 0.0, 2.0).numpy()
    cpu_sign = np.signbit(cpu)
    gpu_sign = np.signbit(gpu)
    ok = not np.array_equal(cpu_sign, gpu_sign)
    print(f'state=distribution_strategy(device placement) cpu={cpu.tolist()} gpu={gpu.tolist()} cpu_sign={cpu_sign.tolist()} gpu_sign={gpu_sign.tolist()}')
    print('BUG_REPRODUCED' if ok else 'NOT_REPRODUCED')
    return


run()
