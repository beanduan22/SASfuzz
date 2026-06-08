# Issue: https://github.com/tensorflow/tensorflow/issues/115732
# Status: confirmed
# State: execution mode
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
x = tf.constant(np.random.randn(65536).astype(np.float16))
y_eager = model(x)
graph_fn = tf.function(model.__call__)
with tf.device('/CPU:0'):
    cpu = graph_fn(x).numpy()
with tf.device('/GPU:0'):
    gpu = graph_fn(x).numpy()

print('CPU:', cpu)
print('GPU:', gpu)
