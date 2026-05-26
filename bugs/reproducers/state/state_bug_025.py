import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

class Model(tf.keras.Model):
    def call(self, x):
        h = tf.linalg.slogdet(x)
        return h

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = Model()
x = tf.constant([[2.0, 4.0, 6.0], [4.0, 10.0, 12.0], [6.0, 12.0, 18.0]], tf.float32)
out = strategy.run(lambda x: model(x), args=(x,))
state = strategy.experimental_local_results(out)

print('State:', state)
