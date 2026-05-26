import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

class Model(tf.keras.Model):
    def call(self, x):
        h = tf.quantization.fake_quant_with_min_max_vars_gradient(gradients=x, inputs=x, min=[1, 1], max=[1, 1])
        return h[0]

model = Model()
x = tf.constant(1.0)
try:
    with tf.GradientTape() as tape:
        tape.watch(x)
        out = model(x)
    grad = tape.gradient(out, x)
    print('Output:', out.numpy())
    print('Gradient:', None if grad is None else grad.numpy())
except Exception as exc:
    print('Error:', type(exc).__name__, str(exc).splitlines()[0])
