import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

class Model(tf.keras.Model):
    def call(self, x):
        h = tf.negative(tf.math.top_k(tf.negative(x)))
        return h

model = Model()
x = tf.constant([0, -2, 1, -4, 3])
y_eager = model(x)
graph_fn = tf.function(model.__call__)
y_graph = graph_fn(x)
try:
    expected = -tf.math.top_k(-x)
except Exception as exc:
    expected = type(exc).__name__ + ': ' + str(exc).splitlines()[0]

print('Eager:', y_eager)
print('Graph:', y_graph)
print('Expected:', expected)
