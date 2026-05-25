                                                        
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, tensorflow as tf

np.random.seed(0)
x_np = np.random.randn(1000).astype(np.float32) * 1e4
x_f16 = tf.constant(x_np.astype(np.float16))

with tf.device("/CPU:0"):
    cpu = float(tf.cast(tf.math.reduce_std(x_f16), tf.float32).numpy())
with tf.device("/GPU:0"):
    gpu = float(tf.cast(tf.math.reduce_std(x_f16), tf.float32).numpy())

print(f"reference fp64: {float(np.std(x_np.astype(np.float64))):.4e}")
print(f"reduce_std  cpu={cpu}  gpu={gpu}")
