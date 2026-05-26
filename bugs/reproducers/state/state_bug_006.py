import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

class Model(tf.keras.Model):
    def call(self, x):
        h = tf.image.adjust_hue(x, tf.constant(0.5, tf.float32))
        return h

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = Model()
x = tf.constant([[[np.nan, 0.5, 0.5], [0.5, 0.5, 0.5]]], tf.float32)
out = strategy.run(lambda x: model(x), args=(x,))
state = [v.numpy() for v in strategy.experimental_local_results(out)]
expected = tf.image.adjust_hue(x, 0.5).numpy()

print('State:', state)
print('Expected:', expected)
