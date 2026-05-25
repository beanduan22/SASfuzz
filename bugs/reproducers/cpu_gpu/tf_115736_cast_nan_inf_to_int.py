                                                        
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

specials = tf.constant([float("nan"), float("inf"), -float("inf")], dtype=tf.float32)

for to_dtype in [tf.int32, tf.int64]:
    with tf.device("/CPU:0"):
        cpu = tf.cast(specials, to_dtype).numpy()
    with tf.device("/GPU:0"):
        gpu = tf.cast(specials, to_dtype).numpy()
    print(f"to={to_dtype.name:5s}  cpu={cpu}  gpu={gpu}")
