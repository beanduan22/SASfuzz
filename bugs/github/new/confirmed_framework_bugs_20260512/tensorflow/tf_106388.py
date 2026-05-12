import os
import warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
import tensorflow as tf

def f(device):
    with tf.device(device):
        try:
            print(tf.raw_ops.SparseFillEmptyRowsGrad(
                reverse_index_map=tf.constant([20, 21], tf.int64),
                grad_values=tf.constant([3, 4, 5], tf.int64),
            ))
        except Exception as e:
            print(type(e).__name__)

f("/CPU:0")
f("/GPU:0")
