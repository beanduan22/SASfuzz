import os
import warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
import tensorflow as tf

x = tf.ones((3, 1), tf.float32)
y = tf.ones((1, 5, 2), tf.float32)
def f(device):
    with tf.device(device):
        try:
            print(tf.raw_ops.NotEqual(x=x, y=y, incompatible_shape_error=False).numpy())
        except Exception as e:
            print(type(e).__name__)

f("/CPU:0")
f("/GPU:0")
