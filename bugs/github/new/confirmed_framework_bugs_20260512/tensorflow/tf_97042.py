import os
import warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
import tensorflow as tf

x = tf.constant([[[[70000.0], [90000.0]], [[110000.0], [130000.0]]]], tf.float32)
with tf.device("/CPU:0"):
    print(tf.experimental.numpy.cumsum(x, axis=1, dtype=tf.int16).numpy())
with tf.device("/GPU:0"):
    print(tf.experimental.numpy.cumsum(x, axis=1, dtype=tf.int16).numpy())
