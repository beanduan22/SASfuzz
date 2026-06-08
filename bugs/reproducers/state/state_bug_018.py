# Issue: https://github.com/tensorflow/tensorflow/issues/117771
# Status: confirmed
# State: execution mode
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.w = tf.Variable(tf.constant([[0.1], [0.2], [0.3], [0.4], [0.5], [0.6]], tf.float32), shape=tf.TensorShape(None), dtype=tf.float32)

    def call(self, x):
        h = tf.matmul(x, self.w)
        return h

model = Model()
x = tf.constant([[2.0, 4.0, 6.0, 8.0]], tf.float32, shape=[1, 4])
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
