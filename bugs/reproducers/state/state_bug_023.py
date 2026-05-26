import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

class Model(tf.keras.Model):
    def call(self, x):
        a, b = x
        h = tf.raw_ops.NotEqual(x=a, y=b, incompatible_shape_error=False)
        return h

x = (tf.ones((3, 1), tf.float32), tf.ones((1, 5, 2), tf.float32))
strategy_cpu = tf.distribute.MirroredStrategy(devices=['/CPU:0'])
with strategy_cpu.scope():
    cpu_model = Model()
try:
    cpu = strategy_cpu.run(lambda x: cpu_model(x), args=(x,))
except Exception as exc:
    cpu = type(exc).__name__ + ': ' + str(exc).splitlines()[0]
strategy_gpu = tf.distribute.MirroredStrategy(devices=['/GPU:0'])
with strategy_gpu.scope():
    gpu_model = Model()
try:
    gpu = strategy_gpu.run(lambda x: gpu_model(x), args=(x,))
except Exception as exc:
    gpu = type(exc).__name__ + ': ' + str(exc).splitlines()[0]

print('CPU:', cpu)
print('GPU:', gpu)
