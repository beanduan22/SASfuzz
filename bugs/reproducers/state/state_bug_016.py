import math
import os
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")
import tensorflow as tf


def run():

    def log_fn(a):
        return tf.py_function(lambda z: tf.math.log(z), [a], a.dtype)

    def test(a):
        with tf.GradientTape() as tape:
            tape.watch(a)
            y = log_fn(a)
        return tape.jacobian(y, a)
    try:
        out = test(tf.constant([0, 2, 3], tf.float32))
    except Exception as exc:
        msg = type(exc).__name__ + ': ' + str(exc)
        ok = 'pyfunc' in msg.lower() or 'EagerPyFunc'.lower() in msg.lower() or 'UnknownError' in msg
        print(f'state=gradient_tracking(GradientTape.jacobian + py_function) err={msg[:240]}')
        print('BUG_REPRODUCED' if ok else 'NOT_REPRODUCED')
        return
    print(f'state=gradient_tracking(GradientTape.jacobian + py_function) returned={out}')
    print('BUG_REPRODUCED' if False else 'NOT_REPRODUCED')
    return


run()
