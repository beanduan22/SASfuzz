import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')
import tensorflow as tf
import numpy as np
x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=np.float32)
h, w = x.shape

def numpy_ref(a):
    b = a.reshape(h, w, 1)
    c = np.concatenate(list(np.split(b, w, axis=1)), axis=0)
    return c.reshape(h, w)
expected = numpy_ref(x)

class M(tf.keras.Model):

    @tf.function(input_signature=[tf.TensorSpec([h, w], tf.float32)])
    def call(self, inp):
        a = tf.reshape(inp, [h, w, 1])
        b = tf.concat(tf.unstack(a, axis=1), 0)
        return tf.reshape(b, [h, w])
model = M()
keras_out = model(tf.constant(x)).numpy()
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
interpreter = tf.lite.Interpreter(model_content=tflite_model, experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
in_idx = interpreter.get_input_details()[0]['index']
interpreter.resize_tensor_input(in_idx, [h, w])
interpreter.allocate_tensors()
interpreter.set_tensor(in_idx, x)
interpreter.invoke()
tflite_out = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
print(f'state=execution_mode(tf.function/TFLite conversion) expected={expected.flatten().tolist()} tflite={tflite_out.flatten().tolist()}')
