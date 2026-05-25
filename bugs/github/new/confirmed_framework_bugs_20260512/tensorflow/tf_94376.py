import os
import warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
import tensorflow as tf

def f(device):
    with tf.device(device):
        try:
            print(tf.raw_ops.SparseSegmentSqrtNGradV2(
                grad=tf.constant([1.0, 2.0, 3.0], tf.float64),
                indices=tf.constant([-2], tf.int64),
                segment_ids=tf.constant([-2], tf.int64),
                dense_output_dim0=tf.constant(2, tf.int32),
            ))
        except Exception as e:
            print(type(e).__name__)

f("/CPU:0")
f("/GPU:0")
