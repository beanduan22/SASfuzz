import os
import warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
import tensorflow as tf

x = tf.constant([-47], tf.int64)
y = tf.constant([66], tf.int64)
with tf.device("/CPU:0"):
    print(tf.pow(x, y).numpy())
with tf.device("/GPU:0"):
    print(tf.pow(x, y).numpy())
