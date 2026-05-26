import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

class Model(tf.keras.Model):
    def call(self, x):
        h = tf.math.abs(x)
        return h

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = Model()
x = tf.constant([complex(np.inf, np.nan), complex(np.nan, np.inf)], tf.complex64)
out = strategy.run(lambda x: model(x), args=(x,))
state = [v.numpy() for v in strategy.experimental_local_results(out)]
expected = np.array([np.inf, np.inf], dtype=np.float32)

print('State:', state)
print('Expected:', expected)
