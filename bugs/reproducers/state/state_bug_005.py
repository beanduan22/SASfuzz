import os
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")
import tensorflow as tf


def run():
    x = tf.constant([[3.0, -2.0, -7.0, 4.0, -1.0]], tf.float32)
    eager = tf.argmin(tf.nn.relu(x), axis=-1).numpy().tolist()
    tf.config.optimizer.set_experimental_options({'arithmetic_optimization': True})

    @tf.function
    def arith_on(v):
        return tf.argmin(tf.nn.relu(v), axis=-1)
    on = arith_on(x).numpy().tolist()
    tf.config.optimizer.set_experimental_options({'arithmetic_optimization': False})

    @tf.function
    def arith_off(v):
        return tf.argmin(tf.nn.relu(v), axis=-1)
    off = arith_off(x).numpy().tolist()
    ok = eager == off and on != off
    print(f'state=execution_mode(tf.function optimizer) eager={eager} on={on} off={off}')
    print('BUG_REPRODUCED' if ok else 'NOT_REPRODUCED')
    return


run()
