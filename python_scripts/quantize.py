import tensorflow as tf
import numpy as np
import os
import librosa

# Load your trained Keras model
model = tf.keras.models.load_model('python_scripts\WWD.h5')

data_path_dict = {
    0: ["..\WakeWordDetection\\background_sound\\" + file_path for file_path in os.listdir("..\WakeWordDetection\\background_sound\\")],
    1: ["..\WakeWordDetection\\audio_data\\" + file_path for file_path in os.listdir("..\WakeWordDetection\\audio_data\\")]
}
# Define a representative dataset generator for quantization
def representative_data_gen():
    count = 1
    print('Processing Data:\n')
    for _, list_of_files in data_path_dict.items():
        # Generate sample data (replace with your actual input data)
        # sample_input = np.random.rand(1, 40).astype(np.float32)
        for single_file in list_of_files:
            audio, sample_rate = librosa.load(single_file) ## Loading file
            mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40) ## Apllying mfcc
            mfcc_processed = np.mean(mfcc.T, axis=0) ## some pre-processing
            print(f'Completed {count}/317')
            count+=1
            yield [mfcc_processed]

# Convert the model to TensorFlow Lite format with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen()
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # Quantize input
converter.inference_output_type = tf.int8  # Quantize output
tflite_quant_model = converter.convert()

# Save the quantized model
with open('python_scripts\quantized_WWD_model.tflite', 'wb') as f:
    f.write(tflite_quant_model)
