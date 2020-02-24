import cv2
import numpy as np
import tensorflow as tf

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="models/tflite_models/bamboo.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)
print(output_details)
# Test model on random input data.
# [[9.9969351e-01 1.0274102e-07 1.0283768e-07 8.5078852e-08 4.6199983e-07
#   5.7678426e-09 2.2887256e-05 9.9612553e-05 3.1634997e-07 1.7962180e-04
#   3.3116617e-06 3.6966561e-08]]


# [[4.9366648e-08 2.5977779e-06 1.3787770e-05 2.3988930e-06 1.7501095e-07
#   9.9991322e-01 2.0010489e-07 9.2797529e-07 2.2842062e-07 7.0880419e-06
#   4.1521480e-06 5.5201122e-05]]
image = cv2.cvtColor(cv2.imread("data/individualImage.png"), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
interpreter.set_tensor(input_details[0]['index'], np.expand_dims(image, axis=0))

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(np.argmax(output_data[0]))
