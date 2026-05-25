                                                                          
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

x = tf.constant([float("inf")], dtype=tf.float32)

with tf.device("/CPU:0"):
    cpu = tf.cast(x, tf.uint64).numpy()
with tf.device("/GPU:0"):
    gpu = tf.cast(x, tf.uint64).numpy()

print("cpu:", cpu)
print("gpu:", gpu)
