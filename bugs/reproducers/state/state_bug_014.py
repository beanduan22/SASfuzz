import math
import os
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")
import tensorflow as tf
import numpy as np


def run():

    def test(x):
        with tf.GradientTape() as tape:
            tape.watch(x)
            w = tf.math.reduce_prod(x)
        return tape.gradient(w, x)
    x = tf.constant([[0, 0.1, 0.2], [0, 0.1, 0.2]], tf.float32)
    out = test(x).numpy()
    expected = np.array([[0.0, 0.2, 0.4], [0.0, 2.0, 4.0]], dtype=np.float32)
    ok = np.all(out == 0) and (not np.array_equal(out, expected))
    print(f'state=gradient_tracking(GradientTape.gradient) out={out.tolist()} expected={expected.tolist()}')
    print('BUG_REPRODUCED' if ok else 'NOT_REPRODUCED')
    return


run()
