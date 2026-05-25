                                                                                 
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

x = tf.constant([[[float("nan"), 0.5, 0.5], [0.5, 0.5, 0.5]]], dtype=tf.float32)
d = tf.constant(0.5, dtype=tf.float32)

with tf.device("CPU:0"):
    cpu = tf.image.adjust_hue(x, d)
with tf.device("GPU:0"):
    gpu = tf.image.adjust_hue(x, d)

print("cpu:", cpu.numpy().flatten())
print("gpu:", gpu.numpy().flatten())
