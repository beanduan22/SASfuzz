# https://github.com/tensorflow/tensorflow/issues/115733  (reduce_mean variant)
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, tensorflow as tf

np.random.seed(0)
x = tf.constant((np.random.randn(1000).astype(np.float32) * 1e4).astype(np.float16))

with tf.device("/CPU:0"):
    cpu = float(tf.cast(tf.math.reduce_mean(x), tf.float32).numpy())
with tf.device("/GPU:0"):
    gpu = float(tf.cast(tf.math.reduce_mean(x), tf.float32).numpy())

print(f"reduce_mean  cpu={cpu}  gpu={gpu}")
