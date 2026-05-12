import os
import warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
import tensorflow as tf

x = tf.constant([complex(float("inf"), float("nan")), complex(float("nan"), -float("inf"))], tf.complex128)
with tf.device("/CPU:0"):
    print(tf.math.abs(x).numpy())
with tf.device("/GPU:0"):
    print(tf.math.abs(x).numpy())
