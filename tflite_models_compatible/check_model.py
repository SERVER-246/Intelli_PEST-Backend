"""Check TFLite model compatibility."""
import tensorflow as tf

model_path = r'D:\Intelli_PEST-Backend\tflite_models_compatible\android_models\mobilenet_v2.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)

print('TFLite Model Analysis')
print('=' * 50)

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Input shape: {input_details[0]['shape']}")
print(f"Input dtype: {input_details[0]['dtype']}")
print(f"Output shape: {output_details[0]['shape']}")
print(f"Output dtype: {output_details[0]['dtype']}")
print(f"TensorFlow version: {tf.__version__}")
