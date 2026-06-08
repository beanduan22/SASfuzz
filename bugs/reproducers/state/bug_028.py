import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

np.random.seed(1)
x_np = np.random.randn(10000).astype(np.float32)
x = tf.constant(x_np, dtype=tf.bfloat16)
ref = np.cumsum(x_np.astype(np.float64)).astype(np.float32)

strategy_cpu = tf.distribute.MirroredStrategy(devices=['/CPU:0'])
with strategy_cpu.scope():
    cpu = strategy_cpu.run(lambda x: tf.cast(tf.math.cumsum(x), tf.float32), args=(x,))
strategy_gpu = tf.distribute.MirroredStrategy(devices=['/GPU:0'])
with strategy_gpu.scope():
    gpu = strategy_gpu.run(lambda x: tf.cast(tf.math.cumsum(x), tf.float32), args=(x,))

cpu_arr = strategy_cpu.experimental_local_results(cpu)[0].numpy()
gpu_arr = strategy_gpu.experimental_local_results(gpu)[0].numpy()
print('CPU tail:', cpu_arr[-5:])
print('GPU tail:', gpu_arr[-5:])
print('Ref tail:', ref[-5:])
