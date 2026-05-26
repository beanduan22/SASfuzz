import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

class Model(tf.keras.Model):
    def call(self, x):
        h = tf.math.reduce_std(x)
        return tf.cast(h, tf.float32)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = Model()
np.random.seed(0)
x = tf.constant((np.random.randn(1000).astype(np.float32) * 1e4).astype(np.float16))
out = strategy.run(lambda x: model(x), args=(x,))
state = [v.numpy() for v in strategy.experimental_local_results(out)]

print('State:', state)
