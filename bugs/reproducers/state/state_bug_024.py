import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

class Model(tf.keras.Model):
    def call(self, x):
        a, b = x
        h = tf.pow(a, b)
        return h

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = Model()
x = (tf.constant([-47], tf.int64), tf.constant([66], tf.int64))
out = strategy.run(lambda x: model(x), args=(x,))
state = [v.numpy() for v in strategy.experimental_local_results(out)]

print('State:', state)
