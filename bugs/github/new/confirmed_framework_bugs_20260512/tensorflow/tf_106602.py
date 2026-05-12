import os
import warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
import numpy as np
import tensorflow as tf

x = tf.constant([7.0, np.nan, -2.0], tf.float32)
ids = tf.constant([0, 4, 5], tf.int32)
with tf.device("/CPU:0"):
    print(tf.math.unsorted_segment_max(x, ids, 9).numpy())
with tf.device("/GPU:0"):
    print(tf.math.unsorted_segment_max(x, ids, 9).numpy())
