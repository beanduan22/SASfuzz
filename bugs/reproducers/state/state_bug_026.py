import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

class Model(tf.keras.Model):
    def call(self, x):
        a, b = x
        h = tf.sparse.cross([a, b]).values
        return h

x = (tf.constant([[float('inf')], [2.0]], tf.float32), tf.constant([[5], [6]], tf.int32))
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
