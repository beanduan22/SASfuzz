import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

class Model(tf.keras.Model):
    def call(self, x):
        h = tf.negative(tf.math.top_k(tf.negative(x)).values)
        return h

model = Model()
x = tf.constant([0, -2, 1, -4, 3])
y_eager = model(x)
graph_fn = tf.function(model.__call__)
with tf.device('/CPU:0'):
    cpu = graph_fn(x).numpy()
with tf.device('/GPU:0'):
    gpu = graph_fn(x).numpy()

print('CPU:', cpu)
print('GPU:', gpu)
