import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

class Model(tf.keras.Model):
    def call(self, x):
        h = tf.clip_by_value(x, 0.0, 2.0)
        ctx = tf.distribute.get_replica_context()
        h = ctx.all_reduce('sum', h)
        return h

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = Model()
x = tf.constant([-0.0, -0.0], tf.float64)
out = strategy.run(lambda x: model(x), args=(x,))
state = [v.numpy() for v in strategy.experimental_local_results(out)]
expected = tf.clip_by_value(x, 0.0, 2.0).numpy()

print('State:', state)
print('Expected:', expected)
print('State sign:', [np.signbit(v).tolist() for v in state])
print('Expected sign:', np.signbit(expected).tolist())
