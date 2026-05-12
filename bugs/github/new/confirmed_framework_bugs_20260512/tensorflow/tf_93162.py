import os
import warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
import tensorflow as tf

a = tf.constant([[float("inf")], [2.0]], tf.float32)
b = tf.constant([[5], [6]], tf.int32)
with tf.device("/CPU:0"):
    print(tf.sparse.cross([a, b]).values.numpy())
with tf.device("/GPU:0"):
    print(tf.sparse.cross([a, b]).values.numpy())
