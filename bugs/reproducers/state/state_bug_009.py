import os
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")
import tensorflow as tf
import numpy as np


def run():
    assert tf.config.list_physical_devices('GPU'), 'GPU is required'
    x = tf.constant([float('nan'), 3.0, 1.0, float('nan'), 2.0, float('nan'), 0.5], tf.float32)

    @tf.function
    def sort_fn(v):
        return tf.sort(v)
    with tf.device('/CPU:0'):
        cpu = sort_fn(x).numpy()
    with tf.device('/GPU:0'):
        gpu = sort_fn(x).numpy()
    cpu_pos = np.where(np.isnan(cpu))[0].tolist()
    gpu_pos = np.where(np.isnan(gpu))[0].tolist()
    print(f'state=execution_mode(tf.function) cpu={cpu.tolist()} gpu={gpu.tolist()}')
    print('BUG_REPRODUCED' if cpu_pos != gpu_pos else 'NOT_REPRODUCED')
    return


run()
