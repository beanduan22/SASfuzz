import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')
import tensorflow as tf
import numpy as np
np.random.seed(0)
x_np = np.random.randn(65536).astype(np.float16)
ref = float(np.mean(x_np.astype(np.float64)))
x = tf.constant(x_np)

@tf.function
def mean_fn(v):
    return tf.cast(tf.reduce_mean(v), tf.float32)
with tf.device('/CPU:0'):
    cpu = float(mean_fn(x).numpy())
gpu = None
if tf.config.list_physical_devices('GPU'):
    with tf.device('/GPU:0'):
        gpu = float(mean_fn(x).numpy())
print(f'state=execution_mode(tf.function) ref={ref} cpu={cpu} gpu={gpu}')
