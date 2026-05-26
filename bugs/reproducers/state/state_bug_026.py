import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

class Model(tf.keras.Model):
    def call(self, x):
        a, b = x
        h = tf.sparse.cross([a, b]).values
        return h

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = Model()
x = (tf.constant([[float('inf')], [2.0]], tf.float32), tf.constant([[5], [6]], tf.int32))
out = strategy.run(lambda x: model(x), args=(x,))
state = [v.numpy() for v in strategy.experimental_local_results(out)]

print('State:', state)
