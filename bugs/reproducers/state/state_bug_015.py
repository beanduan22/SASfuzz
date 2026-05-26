import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')
import tensorflow as tf
import numpy as np

def test(a, b):
    with tf.GradientTape() as tape:
        tape.watch(a)
        tape.watch(b)
        w = a * tf.math.reciprocal(b)
    return tape.jacobian(w, a)
a = tf.constant([3], tf.float32)
b = tf.constant([0, 2, 3], tf.float32)
out = test(a, b).numpy()
expected = np.array([[np.inf], [0.5], [1.0 / 3.0]], dtype=np.float32)
print(f'state=gradient_tracking(GradientTape.jacobian) out={out.tolist()} expected={expected.tolist()}')
