import tensorflow as tf
import numpy as np
import os

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="python_scripts\quantized_WWD_model.tflite")
interpreter.allocate_tensors()

# Get the details of the tensors in the model
tensor_details = interpreter.get_tensor_details()

# Extract the weights from the tensors
weights = {}
for tensor in tensor_details:
    tensor_name = tensor['name']
    tensor_index = tensor['index']
    tensor_data = interpreter.tensor(tensor_index)()
    weights[tensor_name] = tensor_data

# Print or save the weights
for name, data in weights.items():
    print(f"Name: {name}")
    print(f"Shape: {data.shape}")
    print(f"Data: {data}\n")

os.makedirs('weights', exist_ok=True)

for name, data in weights.items():
    # Clean the tensor name to be a valid file name
    clean_name = name.replace('/', '_')
    # Save each tensor's data as a numpy array
    np.save(os.path.join('weights', f"{clean_name}.npy"), data)

# Open the header file for writing
with open(os.path.join('include', 'weights.h'), 'w') as header_file:
    header_file.write("#ifndef WEIGHTS_H\n")
    header_file.write("#include <stdint.h>\n\n")

    # Write each tensor's data as a C array
    for name, data in weights.items():
        # Clean the tensor name to be a valid C identifier
        clean_name = name.replace('/', '_').replace(':', '_').replace(';','_');

        # Write the array definition
        header_file.write(f"// {name}\n")
        header_file.write(f"const int8_t {clean_name}[] = {{\n")

        # Write the array data
        flat_data = data.flatten()
        for i, value in enumerate(flat_data):
            if i % 10 == 0:
                header_file.write("\n    ")
            header_file.write(f"{value}, ")
        header_file.write("\n};\n\n")

    header_file.write("#endif // WEIGHTS_H\n")