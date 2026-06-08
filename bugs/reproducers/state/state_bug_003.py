# Issue: https://github.com/tensorflow/tensorflow/issues/116047
# Status: fixed
# State: execution mode
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

x = np.array(
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
    dtype=np.float32,
)
h, w = x.shape

class Model(tf.keras.Model):
    @tf.function(input_signature=[tf.TensorSpec([h, w], tf.float32)])
    def call(self, inp):
        a = tf.reshape(inp, [h, w, 1])
        b = tf.concat(tf.unstack(a, axis=1), 0)
        return tf.reshape(b, [h, w])

expected = np.concatenate(list(np.split(x.reshape(h, w, 1), w, axis=1)), axis=0).reshape(h, w)
model = Model()
keras_out = model(tf.constant(x)).numpy()
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
interpreter = tf.lite.Interpreter(
    model_content=tflite_model,
    experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
)
in_idx = interpreter.get_input_details()[0]['index']
interpreter.resize_tensor_input(in_idx, [h, w])
interpreter.allocate_tensors()
interpreter.set_tensor(in_idx, x)
interpreter.invoke()
tflite_out = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

print('Expected:', expected)
print('Keras:', keras_out)
print('TFLite:', tflite_out)
