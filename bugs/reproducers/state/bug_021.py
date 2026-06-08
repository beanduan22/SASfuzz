import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

class Model(tf.keras.Model):
    def call(self, x):
        h = tf.raw_ops.SparseFillEmptyRowsGrad(reverse_index_map=tf.constant([20, 21], tf.int64), grad_values=x)
        return h

x = tf.constant([3, 4, 5], tf.int64)
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
