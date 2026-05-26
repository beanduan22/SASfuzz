import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

class Model(tf.keras.Model):
    def call(self, x):
        h = tf.clip_by_value(x, 0.0, 2.0)
        ctx = tf.distribute.get_replica_context()
        h = ctx.all_reduce('sum', h)
        return h

x = tf.constant([-0.0, -0.0], tf.float64)
strategy_cpu = tf.distribute.MirroredStrategy(devices=['/CPU:0'])
with strategy_cpu.scope():
    cpu_model = Model()
cpu = strategy_cpu.run(lambda x: cpu_model(x), args=(x,))
strategy_gpu = tf.distribute.MirroredStrategy(devices=['/GPU:0'])
with strategy_gpu.scope():
    gpu_model = Model()
gpu = strategy_gpu.run(lambda x: gpu_model(x), args=(x,))
cpu = strategy_cpu.experimental_local_results(cpu)[0].numpy()
gpu = strategy_gpu.experimental_local_results(gpu)[0].numpy()

print('CPU:', cpu, np.signbit(cpu))
print('GPU:', gpu, np.signbit(gpu))
