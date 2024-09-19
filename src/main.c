#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "weights.h"
#include "arm_nnfunctions.h"

#define INPUT_DIM 40
#define HIDDEN_DIM_1 256
#define HIDDEN_DIM_2 256
#define OUTPUT_DIM 2

// Define the quantization parameters
#define INPUT_OFFSET -78
#define OUTPUT_OFFSET 128
#define INPUT_SCALE 2.9606902599334717
#define INPUT_ZERO_POINT 78
#define OUTPUT_SCALE 0.00390625
#define OUTPUT_ZERO_POINT -128

// Define the extracted multipliers and shifts for each layer
#define FC1_MULTIPLIER 1617803136
#define FC1_SHIFT 7

#define FC2_MULTIPLIER 1272522880
#define FC2_SHIFT 7

#define FC3_MULTIPLIER 1668264448
#define FC3_SHIFT 9


void quantize_input(const float* input_data, int8_t* quantized_data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        quantized_data[i] = (int8_t)round(input_data[i] / INPUT_SCALE) + INPUT_ZERO_POINT;
    }
    // Print quantized input data
    printf("Quantized input data:\n");
    for (size_t i = 0; i < size; ++i) {
        printf("%d ", quantized_data[i]);
    }
    printf("\n");
}

void dequantize_output(const int8_t* quantized_data, float* output_data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        output_data[i] = (quantized_data[i] - OUTPUT_ZERO_POINT) * OUTPUT_SCALE;
    }
}

void run_inference(const int8_t* input_data, int8_t* output_data) {
    printf("Entered run_inference function.\n");
    // Fully Connected Layer 1
    cmsis_nn_fc_params fc_params;
    cmsis_nn_per_tensor_quant_params quant_params;
    cmsis_nn_dims input_dims, filter_dims, bias_dims, output_dims;
    int8_t hidden_1[HIDDEN_DIM_1];
    int8_t hidden_2[HIDDEN_DIM_2];

    fc_params.input_offset = INPUT_OFFSET;
    fc_params.filter_offset = 0;  // Typically 0 for symmetric quantization
    fc_params.output_offset = OUTPUT_OFFSET;
    fc_params.activation.max = 127;
    fc_params.activation.min = -128;

    quant_params.multiplier = FC1_MULTIPLIER; // Replace with actual multiplier
    quant_params.shift = FC1_SHIFT;               // Replace with actual shift

    input_dims.n = 1;
    input_dims.h = 1;
    input_dims.w = 1;
    input_dims.c = INPUT_DIM;

    filter_dims.n = INPUT_DIM;
    filter_dims.h = 1;
    filter_dims.w = 1;
    filter_dims.c = HIDDEN_DIM_1;

    bias_dims.n = 1;
    bias_dims.h = 1;
    bias_dims.w = 1;
    bias_dims.c = HIDDEN_DIM_1;

    output_dims.n = 1;
    output_dims.h = 1;
    output_dims.w = 1;
    output_dims.c = HIDDEN_DIM_1;
    // Print the dimensions and parameters for debugging
    printf("input_dims: %d x %d x %d x %d\n", input_dims.n, input_dims.h, input_dims.w, input_dims.c);
    printf("filter_dims: %d x %d x %d x %d\n", filter_dims.n, filter_dims.h, filter_dims.w, filter_dims.c);
    printf("bias_dims: %d x %d x %d x %d\n", bias_dims.n, bias_dims.h, bias_dims.w, bias_dims.c);
    printf("output_dims: %d x %d x %d x %d\n", output_dims.n, output_dims.h, output_dims.w, output_dims.c);
    printf("fc_params: input_offset=%d, filter_offset=%d, output_offset=%d\n", fc_params.input_offset, fc_params.filter_offset, fc_params.output_offset);
    printf("quant_params: multiplier=%d, shift=%d\n", quant_params.multiplier, quant_params.shift);

    // Print the input data to the fully connected layer
    printf("Input data to fully connected layer:\n");
    for (int i = 0; i < INPUT_DIM; i++) {
        printf("%d ", input_data[i]);
    }
    printf("\n");

    // Print the weights and biases
    printf("Weights of fully connected layer:\n");
    for (int i = 0; i < HIDDEN_DIM_1 * INPUT_DIM; i++) {
        printf("%d ", sequential_dense_Weights[i]);
    }
    printf("\n");

    printf("Biases of fully connected layer:\n");
    for (int i = 0; i < HIDDEN_DIM_1; i++) {
        printf("%d ", sequential_dense_Biases[i]);
    }
    printf("\n");
    arm_cmsis_nn_status status;
    status = arm_fully_connected_s4(NULL, &fc_params, &quant_params,
                                    &input_dims, input_data,
                                    &filter_dims, sequential_dense_Weights,
                                    &bias_dims, sequential_dense_Biases,
                                    &output_dims, hidden_1);

    if (status != ARM_CMSIS_NN_SUCCESS) {
        printf("Error in fully connected layer 1: %d\n", status);
        return;
    }

    printf("Fully connected layer 1 output:\n");
    for (int i = 0; i < HIDDEN_DIM_1; i++) {
        printf("%d ", hidden_1[i]);
    }
    printf("\n");

    // ReLU Activation 1
    arm_relu_q7(hidden_1, HIDDEN_DIM_1);

    // printf("ReLU activation 1 output:\n");
    // for (int i = 0; i < HIDDEN_DIM_1; i++) {
    //     printf("%d ", hidden_1[i]);
    // }
    // printf("\n");

    // Fully Connected Layer 2
    quant_params.multiplier = FC2_MULTIPLIER; // Replace with actual multiplier
    quant_params.shift = FC2_SHIFT;               // Replace with actual shift

    input_dims.c = HIDDEN_DIM_1;

    filter_dims.n = HIDDEN_DIM_1;
    filter_dims.c = HIDDEN_DIM_2;

    bias_dims.c = HIDDEN_DIM_2;

    output_dims.c = HIDDEN_DIM_2;

    status = arm_fully_connected_s4(NULL, &fc_params, &quant_params,
                                    &input_dims, hidden_1,
                                    &filter_dims, sequential_dense_1_Weights,
                                    &bias_dims, sequential_dense_1_Biases,
                                    &output_dims, hidden_2);

    if (status != ARM_CMSIS_NN_SUCCESS) {
        printf("Error in fully connected layer 2: %d\n", status);
        return;
    }

    printf("Fully connected layer 2 output:\n");
    for (int i = 0; i < HIDDEN_DIM_2; i++) {
        printf("%d ", hidden_2[i]);
    }
    printf("\n");

    // ReLU Activation 2
    arm_relu_q7(hidden_2, HIDDEN_DIM_2);

    // printf("ReLU activation 2 output:\n");
    // for (int i = 0; i < HIDDEN_DIM_2; i++) {
    //     printf("%d ", hidden_2[i]);
    // }
    // printf("\n");

    // Fully Connected Layer 3
    quant_params.multiplier = FC3_MULTIPLIER; // Replace with actual multiplier
    quant_params.shift = FC3_SHIFT;               // Replace with actual shift

    input_dims.c = HIDDEN_DIM_2;

    filter_dims.n = HIDDEN_DIM_2;
    filter_dims.c = OUTPUT_DIM;

    bias_dims.c = OUTPUT_DIM;

    output_dims.c = OUTPUT_DIM;

    status = arm_fully_connected_s4(NULL, &fc_params, &quant_params,
                                    &input_dims, hidden_2,
                                    &filter_dims, sequential_dense_2_Weights,
                                    &bias_dims, sequential_dense_2_Biases,
                                    &output_dims, output_data);

    if (status != ARM_CMSIS_NN_SUCCESS) {
        printf("Error in fully connected layer 3: %d\n", status);
        return;
    }

    printf("Fully connected layer 3 output:\n");
    for (int i = 0; i < OUTPUT_DIM; i++) {
        printf("%d ", output_data[i]);
    }
    printf("\n");

    // Softmax Activation
    arm_softmax_s8(output_data, 1, OUTPUT_DIM, 1, 0, 0, output_data);

    // printf("Softmax activation output:\n");
    // for (int i = 0; i < OUTPUT_DIM; i++) {
    //     printf("%d ", output_data[i]);
    // }
    // printf("\n");
}

void test_cmsis_nn() {
    printf("Testing CMSIS-NN function.\n");

    int8_t test_input[HIDDEN_DIM_1];
    for (int i = 0; i < HIDDEN_DIM_1; i++) {
        test_input[i] = i - 128; // Sample data
    }

    printf("Input to ReLU:\n");
    for (int i = 0; i < HIDDEN_DIM_1; i++) {
        printf("%d ", test_input[i]);
    }
    printf("\n");

    // Call a simple CMSIS-NN function
    arm_relu_q7(test_input, HIDDEN_DIM_1);

    printf("Output from ReLU:\n");
    for (int i = 0; i < HIDDEN_DIM_1; i++) {
        printf("%d ", test_input[i]);
    }
    printf("\n");
}

void test_fully_connected() {
    printf("Testing Fully Connected Layer.\n");

    int8_t input_data[INPUT_DIM];
    for (int i = 0; i < INPUT_DIM; i++) {
        input_data[i] = i - 20; // Sample data
    }

    int8_t output_data[HIDDEN_DIM_1];

    // Set up parameters
    cmsis_nn_fc_params fc_params;
    fc_params.input_offset = INPUT_OFFSET;
    fc_params.filter_offset = 0;
    fc_params.output_offset = OUTPUT_OFFSET;

    cmsis_nn_per_tensor_quant_params quant_params;
    quant_params.multiplier = 1617803136; // Replace with actual multiplier
    quant_params.shift = 7;               // Replace with actual shift

    cmsis_nn_dims input_dims;
    input_dims.n = 1;
    input_dims.h = 1;
    input_dims.w = 1;
    input_dims.c = INPUT_DIM;

    cmsis_nn_dims filter_dims;
    filter_dims.n = INPUT_DIM;
    filter_dims.h = 1;
    filter_dims.w = 1;
    filter_dims.c = HIDDEN_DIM_1;

    cmsis_nn_dims bias_dims;
    bias_dims.n = 1;
    bias_dims.h = 1;
    bias_dims.w = 1;
    bias_dims.c = HIDDEN_DIM_1;

    cmsis_nn_dims output_dims;
    output_dims.n = 1;
    output_dims.h = 1;
    output_dims.w = 1;
    output_dims.c = HIDDEN_DIM_1;

    int32_t biases[HIDDEN_DIM_1];
    int8_t weights[INPUT_DIM*HIDDEN_DIM_1];

    // Print the dimensions and parameters for debugging
    printf("input_dims: %d x %d x %d x %d\n", input_dims.n, input_dims.h, input_dims.w, input_dims.c);
    printf("filter_dims: %d x %d x %d x %d\n", filter_dims.n, filter_dims.h, filter_dims.w, filter_dims.c);
    printf("bias_dims: %d x %d x %d x %d\n", bias_dims.n, bias_dims.h, bias_dims.w, bias_dims.c);
    printf("output_dims: %d x %d x %d x %d\n", output_dims.n, output_dims.h, output_dims.w, output_dims.c);
    printf("fc_params: input_offset=%d, filter_offset=%d, output_offset=%d\n", fc_params.input_offset, fc_params.filter_offset, fc_params.output_offset);
    printf("quant_params: multiplier=%d, shift=%d\n", quant_params.multiplier, quant_params.shift);

    // Print the input data to the fully connected layer
    printf("Input data to fully connected layer:\n");
    for (int i = 0; i < INPUT_DIM; i++) {
        printf("%d ", input_data[i]);
    }
    printf("\n");

    // Print the weights and biases
    printf("Weights of fully connected layer:\n");
    for (int i = 0; i < HIDDEN_DIM_1 * INPUT_DIM; i++) {
        printf("%d ", weights[i]);
    }
    printf("\n");

    printf("Biases of fully connected layer:\n");
    for (int i = 0; i < HIDDEN_DIM_1; i++) {
        printf("%d ", biases[i]);
    }
    printf("\n");

    // Call the fully connected layer
    arm_cmsis_nn_status status = arm_fully_connected_s4(NULL, &fc_params, &quant_params,
                                    &input_dims, input_data,
                                    &filter_dims, weights,
                                    &bias_dims, biases,
                                    &output_dims, output_data);

    if (status != ARM_CMSIS_NN_SUCCESS) {
        printf("Error in fully connected layer: %d\n", status);
        return;
    }

    printf("Fully connected layer output:\n");
    for (int i = 0; i < HIDDEN_DIM_1; i++) {
        printf("%d ", output_data[i]);
    }
    printf("\n");
}

int main() {
    // Initialize input data (example)
    const float mfcc[INPUT_DIM] = {-4.9774164e+02, 1.1922049e+02, -2.4169176e+01, 1.8644506e+01, -4.4777770e+00, 2.1558158e+00, -8.1305456e+00, 3.8457525e+00, -5.7973795e+00, 4.6593275e+00, -4.7527426e-01, 1.0221251e+01, 1.1862351e+00, 4.2147369e+00, 1.5054088e+00, 6.4555473e+00, -2.0362167e+00, 3.6587176e+00, -2.6960170e+00, 1.0111256e+00, -2.9357400e+00, 5.6378288e+00, 1.2900375e+00, 2.2063334e+00, -3.2787466e+00, 4.1726622e-01, -2.7677703e+00, 3.3359590e+00, -1.1464676e+00, 2.4075580e+00, 1.0437737e+00, 5.2428107e+00, 1.1765444e+00, 1.2050209e+00, -3.2540300e+00, 5.3529816e+00, 8.4979695e-01, 2.6537712e+00, -1.0808622e+00, 3.5724106e+00};
    int8_t input_data[INPUT_DIM];
    int8_t output_data[OUTPUT_DIM];
    float prediction[OUTPUT_DIM];

    printf("Program started.\n");
    // Quantize input data
    quantize_input(mfcc, input_data, INPUT_DIM);

    printf("Starting Inference\n");
    // Perform inference
    run_inference(input_data, output_data);
    // test_cmsis_nn();
    // test_fully_connected();

    printf("Starting Dequantization\n");
    // Dequantize output data
    dequantize_output(output_data, prediction, OUTPUT_DIM);

    // Print the output data
    for (int i = 0; i < OUTPUT_DIM; i++) {
        printf("Pre-Quant Output[%d]: %d\n", i, output_data[i]);
    }

    // Print the output data
    for (int i = 0; i < OUTPUT_DIM; i++) {
        printf("Output[%d]: %f\n", i, prediction[i]);
    }

    return 0;
}