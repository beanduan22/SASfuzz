import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

class Model(tf.keras.Model):
    def __init__(self, dtype):
        super().__init__()
        self.dtype_out = dtype

    def call(self, x):
        h = tf.cast(x, self.dtype_out)
        return h

for value, dtype in [(float('nan'), tf.int32), (float('inf'), tf.int32), (float('inf'), tf.int64)]:
    strategy_cpu = tf.distribute.MirroredStrategy(devices=['/CPU:0'])
    with strategy_cpu.scope():
        cpu_model = Model(dtype)
    x = tf.constant([value], tf.float32)
    cpu = strategy_cpu.run(lambda x: cpu_model(x), args=(x,))
    strategy_gpu = tf.distribute.MirroredStrategy(devices=['/GPU:0'])
    with strategy_gpu.scope():
        gpu_model = Model(dtype)
    gpu = strategy_gpu.run(lambda x: gpu_model(x), args=(x,))
    print('CPU:', [v.numpy() for v in strategy_cpu.experimental_local_results(cpu)])
    print('GPU:', [v.numpy() for v in strategy_gpu.experimental_local_results(gpu)])
