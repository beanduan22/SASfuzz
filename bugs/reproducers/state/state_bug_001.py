# Issue: https://github.com/tensorflow/tensorflow/issues/115768
# Status: fixed
# State: execution mode
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

a = tf.ones((8, 8), tf.float32)

@tf.function
def logdet_fn(x):
    return tf.linalg.logdet(x)

with tf.device('/CPU:0'):
    cpu = logdet_fn(a).numpy()
expected = np.linalg.slogdet(np.ones((8, 8), dtype=np.float32))[1]

gpu = None
if tf.config.list_physical_devices('GPU'):
    with tf.device('/GPU:0'):
        gpu = logdet_fn(a).numpy()

print('CPU:', cpu)
print('GPU:', gpu)
print('Expected:', expected)
