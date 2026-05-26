import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

class Model(tf.keras.Model):
    @tf.function(input_signature=[tf.TensorSpec([4, 3], tf.float32)])
    def call(self, x):
        h = tf.reshape(x, [4, 3, 1])
        h = tf.concat(tf.unstack(h, axis=1), 0)
        return tf.reshape(h, [4, 3])

model = Model()
x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=np.float32)
y_eager = model(tf.constant(x))
graph_fn = tf.function(model.__call__)
y_graph = graph_fn(tf.constant(x))
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
interpreter = tf.lite.Interpreter(model_content=tflite_model, experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
in_idx = interpreter.get_input_details()[0]['index']
interpreter.resize_tensor_input(in_idx, [4, 3])
interpreter.allocate_tensors()
interpreter.set_tensor(in_idx, x)
interpreter.invoke()
y_tflite = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
expected = np.concatenate(list(np.split(x.reshape(4, 3, 1), 3, axis=1)), axis=0).reshape(4, 3)

print('Eager:', y_eager.numpy().flatten().tolist())
print('Graph:', y_graph.numpy().flatten().tolist())
print('TFLite:', y_tflite.flatten().tolist())
print('Expected:', expected.flatten().tolist())
