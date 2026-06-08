import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

x = tf.constant([[3.0, -2.0, -7.0, 4.0, -1.0]], tf.float32)
eager = tf.argmin(tf.nn.relu(x), axis=-1)

tf.config.optimizer.set_experimental_options({'arithmetic_optimization': True})
@tf.function
def arith_on(v):
    return tf.argmin(tf.nn.relu(v), axis=-1)

on = arith_on(x)
tf.config.optimizer.set_experimental_options({'arithmetic_optimization': False})
@tf.function
def arith_off(v):
    return tf.argmin(tf.nn.relu(v), axis=-1)

off = arith_off(x)

print('Eager:', eager.numpy())
print('Arithmetic on:', on.numpy())
print('Arithmetic off:', off.numpy())
