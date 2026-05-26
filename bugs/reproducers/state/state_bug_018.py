import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')
import tensorflow as tf
w_init = tf.constant([[0.1], [0.2], [0.3], [0.4], [0.5], [0.6]], tf.float32)
x = tf.constant([[2.0, 4.0, 6.0, 8.0]], tf.float32, shape=[1, 4])

class ModelEager(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.w = tf.Variable(w_init, shape=tf.TensorShape(None), dtype=tf.float32)

    @tf.function
    def call(self, inp):
        return tf.matmul(inp, self.w)

class ModelXLA(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.w = tf.Variable(w_init, shape=tf.TensorShape(None), dtype=tf.float32)

    @tf.function(jit_compile=True)
    def call(self, inp):
        return tf.matmul(inp, self.w)

def invoke(cls):
    try:
        return (cls()(x).numpy(), None)
    except Exception as exc:
        return (None, type(exc).__name__ + ': ' + str(exc).splitlines()[0][:120])
eager_out, eager_err = invoke(ModelEager)
xla_out, xla_err = invoke(ModelXLA)
print(f'state=execution_mode(tf.function jit_compile=True) eager={eager_out}/{eager_err} xla={xla_out}/{xla_err}')
