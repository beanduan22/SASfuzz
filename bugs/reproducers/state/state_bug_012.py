import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

class Model(tf.keras.Model):
    def call(self, x):
        h = tf.math.cumsum(x)
        return tf.cast(h, tf.float32)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = Model()
np.random.seed(0)
x_np = np.random.randn(10000).astype(np.float32)
x = tf.constant(x_np, dtype=tf.bfloat16)
out = strategy.run(lambda x: model(x), args=(x,))
state = strategy.experimental_local_results(out)[0].numpy()
expected = np.cumsum(x_np.astype(np.float64)).astype(np.float32)

print('State error:', float(np.max(np.abs(state - expected))))
print('Expected last:', expected[-5:])
print('State last:', state[-5:])
