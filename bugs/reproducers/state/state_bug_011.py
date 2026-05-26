import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

class Model(tf.keras.Model):
    def call(self, x):
        h = tf.reduce_mean(x)
        return tf.cast(h, tf.float32)

model = Model()
np.random.seed(0)
x_np = np.random.randn(65536).astype(np.float16)
x = tf.constant(x_np)
y_eager = model(x)
graph_fn = tf.function(model.__call__)
y_graph = graph_fn(x)
expected = np.mean(x_np.astype(np.float64))

print('Eager:', y_eager.numpy())
print('Graph:', y_graph.numpy())
print('Expected:', expected)
