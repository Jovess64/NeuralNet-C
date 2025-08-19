#ifndef _MAIN_H
    #define _MAIN_H


    #ifdef _MSC_VER
        #pragma message("\tWARNING: MSVC may not properly adhere to the C standard.")
    #elif __STDC_VERSION__ < 199901L
        #error C99 or newer required.
    #endif

    #include <stdio.h>
    #include <stdlib.h>
    #include <stdint.h>
    #include <string.h>
    #include <stdbool.h>
    #include <math.h>
    #include <time.h>
    #include "config_context.h" // contains `MAX_PATH`

    #define CONFIG_FILENAME "config.cfg" // contains everything that would be manually input, or doesn't


    /* `readData.c` */

    // Where `char **results = GetImages();`, only free `results[0]` and `results`, if and only if function returns non-NULL
    // Result is an array of images
    extern char **GetImages(const char *filename, uint32_t *image_count, uint32_t *row_count, uint32_t *col_count);

    // Where `char *results = GetLabels();`, only free `results`, if and only if function returns non-NULL
    // Result is an array of single-byte labels
    extern char *GetLabels(const char *filename, uint32_t *label_count);

    // Returns 0 on success, 1 on failure
    // Only fails if something has gone catastrophically wrong (e.g. malloc failure or irreparably invalidly formatted config file)
    //
    // NOTE: must be manually changed when GetConfigContext is changed
    //       assumes fields in `context` are set to `0` or `NULL` or `'\0'` already
    int GetConfig(const char *config_filename, GetConfigContext *context);

    /* `helpers.c` */

    // returns a double to the power of a long
    extern double lpow(double a, long b);

    // activation function
    // (ReLU)
    extern double Activation(double input);

    // activation function for the last (output) layer
    // (sigmoid)
    extern double OutputActivation(double input);

    // Populates `output` with the jacobian of the activation function with respect to `input`
    extern void ActivationPrime(size_t len, double *output, double *input);

    // Populates `output` with the jacobian of the output activation function (`OutputActivation()`) with respect to `input`
    extern void OutputActivationPrime(size_t len, double *output, double *input);

    // Total Squared Error function
    // `len` is the width of the output layer
    // `output` is the network's output
    // `intended` is the output the network is supposed to produce
    extern double Cost(size_t len, double *output, double *intended);

    // Populates `funcoutput` with the jacobian of the cost function with respect to the DEACTIVATED output neurons
    // `len` is the width of the output layer
    // `funcOutput` is the jacobian
    // `deactivated_netoutput` is the deactivated neurons in the output (like applying the inverse of the output activation function to `netoutput`, if possible)
    // `netoutput` is the network's output
    // `intended` is the output the network is supposed to produce
    //
    // NOTE: `deactivated_output` is commented out because finding the derivative of OutputActivation given its output is trivial for the specific function
    extern void CostPrimeWrtDeactivated(size_t len, double *funcOutput, /*double *deactivated_netoutput,*/ double *netoutput, double *intended);

    // `outVector = Activation(inVector)`
    extern void ActivateVector(size_t length, double *inVector, double *outVector);

    // `outVector = OutputActivation(inVector)`
    extern void ActivateOutputVector(size_t length, double *inVector, double *outVector);

    // `vectorA += vectorB`
    extern void AddVector(size_t length, double *vectorA, double *vectorB);

    // `outvector = (matrix)(invector)`
    // `matrix` should be row-major
    extern void TransformVector(size_t width, size_t height, double *matrix, double *invector, double *outvector);

    // Performs gradient descent
    extern void Descend(size_t layers_count, size_t *layer_lengths, size_t inputSize, double *inputLayer, double **activated_neurons, double **weights, double **biases, 
        double **biasesJacobian, double learningRate);


#endif