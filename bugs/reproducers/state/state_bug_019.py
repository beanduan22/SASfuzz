import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

class Model(tf.keras.Model):
    def call(self, x):
        h = tf.slice(x, [0, 0, 1, 0], [-1, -1, 1, -1])
        _ = tf.slice(x, [0, 0, 0, 0], [-1, -1, 5, -1])
        return h

model = Model()
x = tf.constant(np.arange(12, dtype=np.float32).reshape(1, 2, 3, 2))
@tf.function(jit_compile=True)
def xla_fn(x):
    return model(x)
try:
    with tf.device('/CPU:0'):
        cpu = xla_fn(x)
except Exception as exc:
    cpu = type(exc).__name__ + ': ' + str(exc).splitlines()[0]
try:
    with tf.device('/GPU:0'):
        gpu = xla_fn(x)
except Exception as exc:
    gpu = type(exc).__name__ + ': ' + str(exc).splitlines()[0]

print('CPU:', cpu if isinstance(cpu, str) else cpu.numpy())
print('GPU:', gpu if isinstance(gpu, str) else gpu.numpy())
