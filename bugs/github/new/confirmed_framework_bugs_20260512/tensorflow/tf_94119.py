import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
import numpy as np
import tensorflow as tf

shape = tf.constant([6, 8, 5, 1, 4], tf.int32)
grad = tf.constant(np.full((6, 8, 5, 1, 4), 1.4013e-45, np.float32), tf.bfloat16)
print(tf.raw_ops.AvgPool3DGrad(
    orig_input_shape=shape,
    grad=grad,
    ksize=[1, 2, 2, 2, 1],
    strides=[1, 1, 1, 1, 1],
    padding="VALID",
    data_format="NDHWC",
))
