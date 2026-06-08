import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

class Model(tf.keras.Model):
    def call(self, x):
        h = tf.image.adjust_hue(x, tf.constant(0.5, tf.float32))
        return h

x = tf.constant([[[np.nan, 0.5, 0.5], [0.5, 0.5, 0.5]]], tf.float32)
strategy_cpu = tf.distribute.MirroredStrategy(devices=['/CPU:0'])
with strategy_cpu.scope():
    cpu_model = Model()
cpu = strategy_cpu.run(lambda x: cpu_model(x), args=(x,))
strategy_gpu = tf.distribute.MirroredStrategy(devices=['/GPU:0'])
with strategy_gpu.scope():
    gpu_model = Model()
gpu = strategy_gpu.run(lambda x: gpu_model(x), args=(x,))

print('CPU:', [v.numpy() for v in strategy_cpu.experimental_local_results(cpu)])
print('GPU:', [v.numpy() for v in strategy_gpu.experimental_local_results(gpu)])
