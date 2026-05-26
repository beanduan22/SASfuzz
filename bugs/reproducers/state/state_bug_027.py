import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

class Model(tf.keras.Model):
    def call(self, x):
        data, ids = x
        h = tf.math.unsorted_segment_max(data, ids, num_segments=9)
        return h

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = Model()
x = (tf.constant([7.0, np.nan, -2.0], tf.float32), tf.constant([0, 4, 5], tf.int32))
out = strategy.run(lambda x: model(x), args=(x,))
state = [v.numpy() for v in strategy.experimental_local_results(out)]

print('State:', state)
