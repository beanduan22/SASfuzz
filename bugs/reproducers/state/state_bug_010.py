import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

class Model(tf.keras.Model):
    def call(self, x):
        h = tf.math.reduce_std(x)
        return tf.cast(h, tf.float32)

model = Model()
np.random.seed(0)
x = tf.constant((np.random.randn(1000).astype(np.float32) * 1e4).astype(np.float16))
y_eager = model(x)
graph_fn = tf.function(model.__call__)
y_graph = graph_fn(x)

print('Eager:', y_eager.numpy())
print('Graph:', y_graph.numpy())
