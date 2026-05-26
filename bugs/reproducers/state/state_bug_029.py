import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

class Model(tf.keras.Model):
    def call(self, x):
        h = tf.raw_ops.SparseSegmentSumGradV2(grad=x, indices=tf.constant([-2], tf.int64), segment_ids=tf.constant([-2], tf.int64), dense_output_dim0=tf.constant(2, tf.int32))
        return h

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = Model()
x = tf.constant([1.0, 2.0, 3.0], tf.float64)
out = strategy.run(lambda x: model(x), args=(x,))
state = strategy.experimental_local_results(out)

print('State:', state)
