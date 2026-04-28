# https://github.com/tensorflow/tensorflow/issues/115731
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, tensorflow as tf

np.random.seed(0)
x_np = np.random.randn(10_000).astype(np.float32)
ref = np.cumsum(x_np.astype(np.float64)).astype(np.float32)
x_bf16 = tf.constant(x_np, dtype=tf.bfloat16)

with tf.device("/CPU:0"):
    cpu = tf.cast(tf.math.cumsum(x_bf16), tf.float32).numpy()
with tf.device("/GPU:0"):
    gpu = tf.cast(tf.math.cumsum(x_bf16), tf.float32).numpy()

cpu_err = float(np.linalg.norm(cpu - ref))
gpu_err = float(np.linalg.norm(gpu - ref))
print(f"CPU error vs fp64 ref: {cpu_err:.4e}")
print(f"GPU error vs fp64 ref: {gpu_err:.4e}")
print(f"CPU/GPU error ratio: {cpu_err / gpu_err:.1f}x")
