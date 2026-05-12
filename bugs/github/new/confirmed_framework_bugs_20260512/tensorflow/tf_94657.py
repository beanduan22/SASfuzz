import os
import warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
import tensorflow as tf

x = tf.constant([[2.0, 4.0, 6.0], [4.0, 10.0, 12.0], [6.0, 12.0, 18.0]], tf.float64)
def f(device):
    with tf.device(device):
        try:
            print(tf.linalg.solve(x, x).numpy())
        except Exception as e:
            print(type(e).__name__)

f("/CPU:0")
f("/GPU:0")
