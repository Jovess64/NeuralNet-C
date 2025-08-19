#include <stdlib.h>
#include <stdint.h>
#include <math.h>

/*
    Contains a bunch of helper functions related to the actual learning and matrix operations
*/

// returns a double to the power of a long
double lpow(double a, long b) {
    double result = 1.0;
    int sign = b < 0 ? -1 : 1;
    for (unsigned long absb = sign * b; absb > 0; absb--) {
        result *= a;
    }
    return sign < 0 ? (1.0 / result) : result;
}

// activation function
// (ReLU)
double Activation(double input) {
    return input >= 0.0 ? input : 0.0;
}

// activation function for the last (output) layer
// (sigmoid)
double OutputActivation(double input) {
    return 1.0 / (1.0 + exp(-input));
}

// Populates `output` with the jacobian of the activation function with respect to `input`
void ActivationPrime(size_t len, double *output, double *input) {
    for (size_t i = 0; i < len; i++) {
        output[i] = input[i] >= 0.0 ? 1.0 : 0.0;
    }
}

// Total Squared Error function
// `len` is the width of the output layer
// `output` is the network's output
// `intended` is the output the network is supposed to produce
double Cost(size_t len, double *output, double *intended) {
    double totalError = 0.0;
    for (size_t i = 0; i < len; i++) {
        totalError += (output[i] - intended[i]) * (output[i] - intended[i]);
    }
    return totalError;
}

// Populates `funcoutput` with the jacobian of the cost function with respect to the DEACTIVATED output neurons
// `len` is the width of the output layer
// `funcOutput` is the jacobian output
// `deactivated_netoutput` is the deactivated neurons in the output (like applying the inverse of the output activation function to `netoutput`, if possible)
// `netoutput` is the network's output
// `intended` is the output the network is supposed to produce
//
// NOTE: `deactivated_output` is commented out because finding the derivative of OutputActivation given its output is trivial for the specific function
void CostPrimeWrtDeactivated(size_t len, double *funcOutput, /*double *deactivated_netoutput,*/ double *netoutput, double *intended) {
    for (size_t i = 0; i < len; i++) {
        // partial derivative of cost function
        funcOutput[i] = 2.0 * (netoutput[i] - intended[i]);
        // apply chain rule with partial derivative of `OutputActivation`
        funcOutput[i] *= netoutput[i] * (1.0 - netoutput[i]);
    }
}

// `outVector = Activation(inVector)`
void ActivateVector(size_t length, double *inVector, double *outVector) {
    for (size_t i = 0; i < length; i++) outVector[i] = Activation(inVector[i]);
}

// `outVector = OutputActivation(inVector)`
void ActivateOutputVector(size_t length, double *inVector, double *outVector) {
    for (size_t i = 0; i < length; i++) outVector[i] = OutputActivation(inVector[i]);
}

// `vectorA += vectorB`
void AddVector(size_t length, double *vectorA, double *vectorB) {
    for (size_t i = 0; i < length; i++) vectorA[i] += vectorB[i];
}

// `outvector = (matrix)(invector)`
// `matrix` should be row-major
void TransformVector(size_t width, size_t height, double *matrix, double *invector, double *outvector) {
    for (size_t i = 0; i < height; i++) {
        double sum = 0.0;
        double *matrix_row = &matrix[i * width];
        for (size_t j = 0; j < width; j++) {
            sum += matrix_row[j] * invector[j];
        }
        outvector[i] = sum;
    }
}

// Performs gradient descent
void Descend(size_t layers_count, size_t *layer_lengths, size_t inputSize, double *inputLayer, double **activated_neurons, double **weights, double **biases, double **biasesJacobian, double learningRate) {
    size_t weightsWidth = inputSize;
    double *prevLayer = inputLayer;
    for (size_t layer = 0; layer < layers_count; layer++) {
        for (size_t row = 0; row < layer_lengths[layer]; row++) {
            for (size_t col = 0; col < weightsWidth; col++) {
                weights[layer][(row * weightsWidth) + col] -= learningRate * prevLayer[col] * biasesJacobian[layer][row];
            }
            biases[layer][row] -= learningRate * biasesJacobian[layer][row];
        }
        weightsWidth = layer_lengths[layer];
        prevLayer = activated_neurons[layer];
    }
}