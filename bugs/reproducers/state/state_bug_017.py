import math
import os
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")
import tensorflow as tf


def run():
    a = tf.constant([0, -2, 1, -4, 3])

    @tf.function
    def graph_negative_topk(v):
        return tf.negative(tf.math.top_k(tf.negative(v)))
    y = graph_negative_topk(a)
    unary_err = None
    try:
        _ = -tf.math.top_k(-a)
    except Exception as exc:
        unary_err = type(exc).__name__ + ': ' + str(exc)
    ok = unary_err is not None
    print(f'state=execution_mode(tf.function) tf.negative(top_k)={y} unary_minus_err={unary_err}')
    print('BUG_REPRODUCED' if ok else 'NOT_REPRODUCED')
    return


run()
