import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

class Model(tf.keras.Model):
    def call(self, x):
        a, b = x
        h = tf.raw_ops.NotEqual(x=a, y=b, incompatible_shape_error=False)
        return h

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = Model()
x = (tf.ones((3, 1), tf.float32), tf.ones((1, 5, 2), tf.float32))
try:
    out = strategy.run(lambda x: model(x), args=(x,))
    state = [v.numpy() for v in strategy.experimental_local_results(out)]
except Exception as exc:
    state = type(exc).__name__ + ': ' + str(exc).splitlines()[0]

print('State:', state)
