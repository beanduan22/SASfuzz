import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

class Model(tf.keras.Model):
    def call(self, x):
        a, b = x
        h = tf.divide(a, b)
        return h

model = Model()
x = tf.constant([1], tf.float32)
y = tf.constant([0, 1, 2], tf.float32)
with tf.GradientTape() as tape:
    tape.watch(x)
    out = model((x, y))
grad = tape.jacobian(out, x)
reference_grad = np.array([[np.inf], [1.0], [0.5]], dtype=np.float32)

print('Gradient:', grad.numpy())
print('Reference gradient:', reference_grad)
