import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

class Model(tf.keras.Model):
    def call(self, x):
        h = tf.raw_ops.SparseFillEmptyRowsGrad(reverse_index_map=tf.constant([20, 21], tf.int64), grad_values=x)
        return h

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = Model()
x = tf.constant([3, 4, 5], tf.int64)
try:
    out = strategy.run(lambda x: model(x), args=(x,))
    state = strategy.experimental_local_results(out)
except Exception as exc:
    state = type(exc).__name__ + ': ' + str(exc).splitlines()[0]

print('State:', state)
