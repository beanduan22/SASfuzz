import os
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")
import tensorflow as tf
import numpy as np


def run():
    x = tf.constant(np.arange(12, dtype=np.float32).reshape(1, 2, 3, 2))

    class ModelEager(tf.keras.Model):

        @tf.function
        def call(self, inp):
            y = tf.slice(inp, [0, 0, 1, 0], [-1, -1, 1, -1])
            _ = tf.slice(inp, [0, 0, 0, 0], [-1, -1, 5, -1])
            return y

    class ModelXLA(tf.keras.Model):

        @tf.function(jit_compile=True)
        def call(self, inp):
            y = tf.slice(inp, [0, 0, 1, 0], [-1, -1, 1, -1])
            _ = tf.slice(inp, [0, 0, 0, 0], [-1, -1, 5, -1])
            return y

    def run(cls):
        try:
            out = cls()(x).numpy()
            return (out.shape, None)
        except Exception as exc:
            return (None, type(exc).__name__ + ': ' + str(exc).splitlines()[0][:120])
    eager_shape, eager_err = run(ModelEager)
    xla_shape, xla_err = run(ModelXLA)
    ok = eager_shape is not None and xla_err is not None
    print(f'state=execution_mode(tf.function jit_compile=True) eager={eager_shape}/{eager_err} xla={xla_shape}/{xla_err}')
    print('BUG_REPRODUCED' if ok else 'NOT_REPRODUCED')
    return


run()
