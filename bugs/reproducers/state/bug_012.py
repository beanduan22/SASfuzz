import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

class Model(tf.keras.Model):
    def call(self, x):
        h = tf.math.cumsum(x)
        return tf.cast(h, tf.float32)

np.random.seed(0)
x = tf.constant(np.random.randn(10000).astype(np.float32), dtype=tf.bfloat16)
strategy_cpu = tf.distribute.MirroredStrategy(devices=['/CPU:0'])
with strategy_cpu.scope():
    cpu_model = Model()
cpu = strategy_cpu.run(lambda x: cpu_model(x), args=(x,))
strategy_gpu = tf.distribute.MirroredStrategy(devices=['/GPU:0'])
with strategy_gpu.scope():
    gpu_model = Model()
gpu = strategy_gpu.run(lambda x: gpu_model(x), args=(x,))

print('CPU:', strategy_cpu.experimental_local_results(cpu)[0].numpy()[-5:])
print('GPU:', strategy_gpu.experimental_local_results(gpu)[0].numpy()[-5:])
