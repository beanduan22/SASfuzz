import os
import warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
import tensorflow as tf

x = tf.constant([[2.0, 4.0, 6.0], [4.0, 10.0, 12.0], [6.0, 12.0, 18.0]], tf.float32)
with tf.device("/CPU:0"):
    print([v.numpy() for v in tf.linalg.slogdet(x)])
with tf.device("/GPU:0"):
    print([v.numpy() for v in tf.linalg.slogdet(x)])
