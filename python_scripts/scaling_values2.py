import numpy as np
import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path='python_scripts\quantized_WWD_model.tflite')
interpreter.allocate_tensors()

# Function to get tensor details
def get_tensor_details(interpreter, tensor_name):
    for detail in interpreter.get_tensor_details():
        if detail['name'] == tensor_name:
            scale = detail['quantization_parameters']['scales'][0]
            zero_point = detail['quantization_parameters']['zero_points'][0]
            return scale, zero_point
    raise ValueError(f"Tensor {tensor_name} not found in the model.")

# Function to calculate quantization parameters (multiplier and shift)
def calculate_quant_params(scale):
    if scale == 0:
        return 0, 0
    significand, exponent = np.frexp(scale)
    multiplier = int(np.round(significand * (1 << 31)))
    shift = 31 - exponent  # Adjust the shift calculation to match TensorFlow Lite's expectations
    return multiplier, shift

# List of layers with correct tensor names
layers = [
    {
        "input": "serving_default_dense_input:0",
        "weight": "sequential/dense/MatMul",
        "output": "sequential/dense/MatMul;sequential/activation/Relu;sequential/dense/BiasAdd"
    },
    {
        "input": "sequential/dense/MatMul;sequential/activation/Relu;sequential/dense/BiasAdd",
        "weight": "sequential/dense_1/MatMul",
        "output": "sequential/dense_1/MatMul;sequential/activation_1/Relu;sequential/dense_1/BiasAdd"
    },
    {
        "input": "sequential/dense_1/MatMul;sequential/activation_1/Relu;sequential/dense_1/BiasAdd",
        "weight": "sequential/dense_2/MatMul",
        "output": "sequential/dense_2/MatMul;sequential/dense_2/BiasAdd"
    }
]

# Extract and print quantization parameters for each layer
# Open a text file to write the quantization parameters
with open('python_scripts\quantization_parameters.txt', 'w') as f:
    # Extract and print quantization parameters for each layer
    for layer in layers:
        input_scale, input_zero_point = get_tensor_details(interpreter, layer["input"])
        weight_scale, weight_zero_point = get_tensor_details(interpreter, layer["weight"])
        output_scale, output_zero_point = get_tensor_details(interpreter, layer["output"])

        overall_scale = (input_scale * weight_scale) / output_scale
        multiplier, shift = calculate_quant_params(overall_scale)

        # Print to console
        print(f"Layer: {layer['weight']}")
        print(f"  Input scale: {input_scale}, zero point: {input_zero_point}")
        print(f"  Weight scale: {weight_scale}, zero point: {weight_zero_point}")
        print(f"  Output scale: {output_scale}, zero point: {output_zero_point}")
        print(f"  Overall scale: {overall_scale}")
        print(f"  Multiplier: {multiplier}, Shift: {shift}")
        print()

        # Write to text file
        f.write(f"Layer: {layer['weight']}\n")
        f.write(f"  Input scale: {input_scale}, zero point: {input_zero_point}\n")
        f.write(f"  Weight scale: {weight_scale}, zero point: {weight_zero_point}\n")
        f.write(f"  Output scale: {output_scale}, zero point: {output_zero_point}\n")
        f.write(f"  Overall scale: {overall_scale}\n")
        f.write(f"  Multiplier: {multiplier}, Shift: {shift}\n")
        f.write("\n")
