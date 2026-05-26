import os
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")
import tensorflow as tf
import numpy as np


def run():

    def test(x, y):
        with tf.GradientTape() as tape:
            tape.watch(x)
            tape.watch(y)
            z = tf.divide(x, y)
        return tape.jacobian(z, x)
    x = tf.constant([1], tf.float32)
    y = tf.constant([0, 1, 2], tf.float32)
    out = test(x, y).numpy()
    expected = np.array([[np.inf], [1.0], [0.5]], dtype=np.float32)
    ok = np.isnan(out[1:]).any() and (not np.array_equal(out, expected, equal_nan=True))
    print(f'state=gradient_tracking(GradientTape.jacobian) out={out.tolist()} expected={expected.tolist()}')
    print('BUG_REPRODUCED' if ok else 'NOT_REPRODUCED')
    return


run()
