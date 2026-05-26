import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

class Model(tf.keras.Model):
    def call(self, x):
        h = tf.raw_ops.SparseSegmentSumGradV2(grad=x, indices=tf.constant([-2], tf.int64), segment_ids=tf.constant([-2], tf.int64), dense_output_dim0=tf.constant(2, tf.int32))
        return h

x = tf.constant([1.0, 2.0, 3.0], tf.float64)
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
