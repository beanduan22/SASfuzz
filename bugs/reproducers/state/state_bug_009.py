import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

class Model(tf.keras.Model):
    def call(self, x):
        h = tf.sort(x)
        return h

model = Model()
x = tf.constant([np.nan, 3.0, 1.0, np.nan, 2.0, np.nan, 0.5], tf.float32)
y_eager = model(x)
graph_fn = tf.function(model.__call__)
with tf.device('/CPU:0'):
    cpu = graph_fn(x).numpy()
with tf.device('/GPU:0'):
    gpu = graph_fn(x).numpy()

print('CPU:', cpu)
print('GPU:', gpu)
