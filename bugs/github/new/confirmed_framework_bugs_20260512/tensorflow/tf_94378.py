import os
import warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
import tensorflow as tf

def f(device):
    with tf.device(device):
        try:
            print(tf.raw_ops.SparseToDense(
                sparse_indices=tf.constant([2], tf.int32),
                output_shape=tf.constant([1], tf.int32),
                sparse_values=tf.constant([7], tf.uint16),
                default_value=tf.constant(0, tf.uint16),
                validate_indices=False,
            ).numpy())
        except Exception as e:
            print(type(e).__name__)

f("/CPU:0")
f("/GPU:0")
