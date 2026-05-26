import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

class Model(tf.keras.Model):
    def call(self, x):
        h = tf.linalg.logdet(x)
        return h

model = Model()
x = tf.ones((8, 8), tf.float32)
y_eager = model(x)
graph_fn = tf.function(model.__call__)
y_graph = graph_fn(x)
expected = np.linalg.slogdet(np.ones((8, 8), dtype=np.float32))[1]

print('Eager:', y_eager.numpy())
print('Graph:', y_graph.numpy())
print('Expected:', expected)
