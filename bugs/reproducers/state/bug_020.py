import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

values = [float('nan'), float('inf'), float('-inf')]
dtypes = [tf.int32, tf.int64]

for value in values:
    for dtype in dtypes:
        x = tf.constant([value], tf.float32)
        strategy_cpu = tf.distribute.MirroredStrategy(devices=['/CPU:0'])
        with strategy_cpu.scope():
            cpu = strategy_cpu.run(lambda x: tf.cast(x, dtype), args=(x,))
        strategy_gpu = tf.distribute.MirroredStrategy(devices=['/GPU:0'])
        with strategy_gpu.scope():
            gpu = strategy_gpu.run(lambda x: tf.cast(x, dtype), args=(x,))
        print(value, dtype.name)
        print('CPU:', [v.numpy() for v in strategy_cpu.experimental_local_results(cpu)])
        print('GPU:', [v.numpy() for v in strategy_gpu.experimental_local_results(gpu)])
