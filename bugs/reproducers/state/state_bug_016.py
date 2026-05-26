import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

class Model(tf.keras.Model):
    def call(self, x):
        h = tf.py_function(lambda z: tf.math.log(z), [x], x.dtype)
        return h

model = Model()
x = tf.constant([0, 2, 3], tf.float32)
try:
    with tf.GradientTape() as tape:
        tape.watch(x)
        out = model(x)
    grad = tape.jacobian(out, x)
    print('Output:', out.numpy())
    print('Jacobian:', grad.numpy())
except Exception as exc:
    print('Error:', type(exc).__name__, str(exc).splitlines()[0])
