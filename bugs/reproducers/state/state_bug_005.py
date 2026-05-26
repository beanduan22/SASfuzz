import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

class Model(tf.keras.Model):
    def call(self, x):
        h = tf.nn.relu(x)
        return tf.argmin(h, axis=-1)

model = Model()
x = tf.constant([[3.0, -2.0, -7.0, 4.0, -1.0]], tf.float32)
y_eager = model(x)
tf.config.optimizer.set_experimental_options({'arithmetic_optimization': True})
graph_fn = tf.function(model.__call__)
y_graph = graph_fn(x)
tf.config.optimizer.set_experimental_options({'arithmetic_optimization': False})
expected = tf.argmin(tf.nn.relu(x), axis=-1)

print('Eager:', y_eager.numpy())
print('Graph:', y_graph.numpy())
print('Expected:', expected.numpy())
