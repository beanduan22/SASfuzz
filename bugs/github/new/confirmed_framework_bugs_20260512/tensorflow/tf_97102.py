import os
import warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
import tensorflow as tf

x = tf.ones((32, 53), tf.float64) * -88917319269045.0
with tf.device("/CPU:0"):
    print(tf.linalg.matrix_rank(x, tol=6.0).numpy())
with tf.device("/GPU:0"):
    print(tf.linalg.matrix_rank(x, tol=6.0).numpy())
