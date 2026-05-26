import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

class Model(tf.keras.Model):
    def call(self, x):
        h = tf.math.reduce_prod(x)
        return h

model = Model()
x = tf.constant([[0, 0.1, 0.2], [0, 0.1, 0.2]], tf.float32)
with tf.GradientTape() as tape:
    tape.watch(x)
    out = model(x)
grad = tape.gradient(out, x)
expected = np.array([[0.0, 0.2, 0.4], [0.0, 2.0, 4.0]], dtype=np.float32)

print('Output:', out.numpy())
print('Gradient:', grad.numpy())
print('Expected:', expected)
