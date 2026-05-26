import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(8, kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=0))
        self.hashing = tf.keras.layers.Hashing(num_bins=8, output_mode='int', sparse=False)
        self.dense2 = tf.keras.layers.Dense(8, kernel_constraint=tf.keras.constraints.unit_norm())
        self.activation = tf.keras.layers.Activation(tf.nn.swish)

    def call(self, x):
        h = self.dense1(x)
        h = self.hashing(h)
        h = self.dense2(h)
        return self.activation(h)

tf.keras.utils.set_random_seed(0)
with tf.device('/CPU:0'):
    model = Model()
    x = tf.constant(np.random.RandomState(0).randn(1, 8).astype(np.float32))
    with tf.GradientTape() as tape:
        tape.watch(x)
        out = model(x)
    grad = tape.jacobian(out, x)

print('Output:', out.numpy())
print('Jacobian:', None if grad is None else grad.numpy())
