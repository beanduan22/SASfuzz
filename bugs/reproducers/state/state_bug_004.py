# Issue: https://github.com/tensorflow/tensorflow/issues/46910
# Status: fixed
# State: gradient tracking
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
    with tf.device('/CPU:0'):
        with tf.GradientTape() as tape:
            tape.watch(x)
            out = model(x)
        cpu = tape.gradient(out, x)
except Exception as exc:
    cpu = type(exc).__name__ + ': ' + str(exc).splitlines()[0]
try:
    with tf.device('/GPU:0'):
        with tf.GradientTape() as tape:
            tape.watch(x)
            out = model(x)
        gpu = tape.gradient(out, x)
except Exception as exc:
    gpu = type(exc).__name__ + ': ' + str(exc).splitlines()[0]

print('CPU:', cpu)
print('GPU:', gpu)
