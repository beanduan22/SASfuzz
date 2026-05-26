import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

class Model(tf.keras.Model):
    def __init__(self, dtype):
        super().__init__()
        self.dtype_out = dtype

    def call(self, x):
        h = tf.cast(x, self.dtype_out)
        return h

strategy = tf.distribute.MirroredStrategy()
for value, dtype in [(float('nan'), tf.int32), (float('inf'), tf.int32), (float('inf'), tf.int64)]:
    with strategy.scope():
        model = Model(dtype)
    x = tf.constant([value], tf.float32)
    out = strategy.run(lambda x: model(x), args=(x,))
    state = [v.numpy() for v in strategy.experimental_local_results(out)]
    print('State:', state, 'Input:', value, 'DType:', dtype.name)
