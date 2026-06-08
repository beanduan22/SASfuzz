import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(8, kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=0))
        self.activation = tf.keras.layers.Activation(tf.nn.swish)

    def call(self, x):
        h = self.dense(x)
        return self.activation(h)

tf.keras.utils.set_random_seed(0)
x = tf.constant(np.random.RandomState(0).randn(1, 8).astype(np.float32))
with tf.device('/CPU:0'):
    cpu_model = Model()
    with tf.GradientTape() as tape:
        tape.watch(x)
        cpu = cpu_model(x)
try:
    with tf.device('/GPU:0'):
        gpu_model = Model()
        with tf.GradientTape() as tape:
            tape.watch(x)
            gpu = gpu_model(x)
except Exception as exc:
    gpu = type(exc).__name__ + ': ' + str(exc).splitlines()[0]

print('CPU:', cpu.numpy())
print('GPU:', gpu if isinstance(gpu, str) else gpu.numpy())
