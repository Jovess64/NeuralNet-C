#include "main.h"

// Performs a forward pass on the network
void ForwardPass(size_t inputLayerSize, double *inputLayer, size_t layers_count, size_t *layer_lengths, double **weights, double **biases,
    double **deactivated_neurons, double **activated_neurons) {
    // case for `layer == 0` is considered manually because it maps to `inputLayer` which is special
    size_t layer = 0;
    TransformVector(inputLayerSize, layer_lengths[layer], weights[layer], inputLayer, deactivated_neurons[layer]);
    AddVector(layer_lengths[layer], deactivated_neurons[layer], biases[layer]);
    ActivateVector(layer_lengths[layer], deactivated_neurons[layer], activated_neurons[layer]);
    layer++;
    while (layer < layers_count - 1) {
        TransformVector(layer_lengths[layer - 1], layer_lengths[layer], weights[layer], activated_neurons[layer - 1], deactivated_neurons[layer]);
        AddVector(layer_lengths[layer], deactivated_neurons[layer], biases[layer]);
        ActivateVector(layer_lengths[layer], deactivated_neurons[layer], activated_neurons[layer]);
        layer++;
    }
    // case for `layer == layers_count - 1` is considered manually because it uses a different activation function
    TransformVector(layer_lengths[layer - 1], layer_lengths[layer], weights[layer], activated_neurons[layer - 1], deactivated_neurons[layer]);
    AddVector(layer_lengths[layer], deactivated_neurons[layer], biases[layer]);
    ActivateOutputVector(layer_lengths[layer], deactivated_neurons[layer], activated_neurons[layer]);
}

// Propagates backwards through the network and acquires jacobians; does not perform gradient descent
// `intended` is the ideal output that the network is training to achieve
void BackPropagate(size_t layers_count, size_t *layer_lengths, double **weights, double **deactivated_neurons, double **activated_neurons, double *intended, double **biasJacobian) {

    // NOTE: jacobians are ALL with respect to cost
    //       `biasJacobian` used as derivative of deactivated neurons with respect to cost
    //       `biasJacobian` of `layer - 1` used as derivative of activated neurons with respect to cost

    size_t layer = layers_count - 1;
    CostPrimeWrtDeactivated(layer_lengths[layer], biasJacobian[layer], activated_neurons[layer], intended);
    while (layer-- > 0) {
        // Compute activation derivative once for the layer
        ActivationPrime(layer_lengths[layer], biasJacobian[layer], deactivated_neurons[layer]);

        for (size_t row = 0; row < layer_lengths[layer]; row++) {
            double sum = 0.0;
            for (size_t i = 0; i < layer_lengths[layer + 1]; i++) {
                // Weighted sum of next-layer errors
                sum += biasJacobian[layer + 1][i] * weights[layer + 1][(i * layer_lengths[layer]) + row];
            }
            // Apply chain rule
            biasJacobian[layer][row] *= sum;
        }
    }
}

int main() {
    int returnValue = 0;

    char training_images_filename[MAX_PATH] = { 0 };
    char training_labels_filename[MAX_PATH] = { 0 };
    char testing_images_filename[MAX_PATH] = { 0 };
    char testing_labels_filename[MAX_PATH] = { 0 };
    // declared here to allow `goto Cleanup;`
    char **images = NULL;
    char *labels = NULL;
    char **test_images = NULL;
    char *test_labels = NULL;
    size_t *layer_lengths = NULL;
    double *inputLayer = NULL;
    double **activated_neurons = NULL;
    double **deactivated_neurons = NULL;
    double *all_deactivated_neurons = NULL;
    double *all_activated_neurons = NULL;
    double **weights = NULL; // each element is a row-major weight matrix; col is source neuron, row is dest neuron
    double *all_weights = NULL;
    double **biases = NULL;
    double *all_biases = NULL;
    double *intendedOutput = NULL;
    double **biasJacobians = NULL;
    double *all_biases_j = NULL;

    bool layers_count_set = false;
    size_t layers_count = 0;
    bool learningRate_set = false;
    double learningRate = 0.0;
    bool learningRateMultiplier_set = false;
    double learningRateMultiplier = 0.0; // multiplier to learning rate between epochs

    GetConfigContext configContext = { 0 };
    configContext.learningRate_set = &learningRate_set; // `true` if `learningRate` has been modified already
    configContext.learningRate = &learningRate;
    configContext.learningRateMultiplier_set = &learningRateMultiplier_set; // `true` if `learningRateMultiplier` has been modified already
    configContext.learningRateMultiplier = &learningRateMultiplier;
    configContext.layersCount_set = &layers_count_set; // `true` if `layers_count` has been modified already
    configContext.layersCount = &layers_count;
    configContext.layerLengths = &layer_lengths;
    configContext.training_images_filename = training_images_filename;
    configContext.training_labels_filename = training_labels_filename;
    configContext.testing_images_filename = testing_images_filename;
    configContext.testing_labels_filename = testing_labels_filename;
    if (GetConfig(CONFIG_FILENAME, &configContext)) {
        fprintf(stderr, "Failed to read config file \"%s\".\n", CONFIG_FILENAME);
        returnValue = 1;
        goto CleanupLabel;
    }
    if (layers_count_set && layers_count < 2) {
        fprintf(stderr, "Invalid number of layers from config file \"%s\".\n", CONFIG_FILENAME);
        returnValue = 1;
        goto CleanupLabel;
    }

    /* RETRIEVE TRAINING DATA */

    printf("Retrieving training data...\n");

    if (training_images_filename[0] == '\0') {
        printf("Enter the filename of the training image data file (.idx3-ubyte): ");
        int c;
        size_t index = 0;
        while ((c = getchar()) != '\n' && c != EOF) {
            if (index < MAX_PATH - 1) {
                training_images_filename[index] = c;
                index++;
            }
        }
        training_images_filename[index] = '\0';
    }
    //printf("Retrieving training images from file \"%s\"...\n", training_images_filename);
    uint32_t image_count;
    uint32_t row_count;
    uint32_t col_count;
    images = GetImages(training_images_filename, &image_count, &row_count, &col_count);
    if (images == NULL) {
        fprintf(stderr, "Failed to retrieve training images from file \"%s\".\n", training_images_filename);
        returnValue = 1;
        goto CleanupLabel;
    }
    printf("Successfully retrieved training images from file \"%s\".\n", training_images_filename);

    if (training_labels_filename[0] == '\0') {
        printf("Enter the filename of the training label data file (.idx1-ubyte): ");
        int c;
        size_t index = 0;
        while ((c = getchar()) != '\n' && c != EOF) {
            if (index < MAX_PATH - 1) {
                training_labels_filename[index] = c;
                index++;
            }
        }
        training_labels_filename[index] = '\0';
    }
    //printf("Retrieving training labels from file \"%s\"...\n", training_labels_filename);
    uint32_t label_count;
    labels = GetLabels(training_labels_filename, &label_count);
    if (labels == NULL) {
        fprintf(stderr, "Failed to retrieve training labels from file \"%s\".\n", training_labels_filename);
        returnValue = 1;
        goto CleanupLabel;
    }
    printf("Successfully retrieved training labels from file \"%s\".\n", training_labels_filename);
    printf("Training images: %u\n", image_count);
    printf("\tResolution: %u x %u\n", col_count, row_count);
    if (image_count != label_count) {
        printf("Training labels: %u\n", label_count);
    }

    if (testing_images_filename[0] == '\0') {
        printf("Enter the filename of the testing image data file (.idx3-ubyte): ");
        int c;
        size_t index = 0;
        while ((c = getchar()) != '\n' && c != EOF) {
            if (index < MAX_PATH - 1) {
                testing_images_filename[index] = c;
                index++;
            }
        }
        testing_images_filename[index] = '\0';
    }
    //printf("Retrieving testing images from file \"%s\"...\n", testing_images_filename);
    uint32_t test_image_count;
    uint32_t test_row_count;
    uint32_t test_col_count;
    test_images = GetImages(testing_images_filename, &test_image_count, &test_row_count, &test_col_count);
    if (test_images == NULL) {
        fprintf(stderr, "Failed to retrieve testing images from file \"%s\".\n", testing_images_filename);
        returnValue = 1;
        goto CleanupLabel;
    }
    printf("Successfully retrieved testing images from file \"%s\".\n", testing_images_filename);

    if (testing_labels_filename[0] == '\0') {
        printf("Enter the filename of the testing label data file (.idx1-ubyte): ");
        int c;
        size_t index = 0;
        while ((c = getchar()) != '\n' && c != EOF) {
            if (index < MAX_PATH - 1) {
                testing_labels_filename[index] = c;
                index++;
            }
        }
        testing_labels_filename[index] = '\0';
    }
    //printf("Retrieving testing labels from file \"%s\"...\n", testing_labels_filename);
    uint32_t test_label_count;
    test_labels = GetLabels(testing_labels_filename, &test_label_count);
    if (test_labels == NULL) {
        fprintf(stderr, "Failed to retrieve testing labels from file \"%s\".\n", testing_labels_filename);
        returnValue = 1;
        goto CleanupLabel;
    }
    printf("Successfully retrieved testing labels from file \"%s\".\n", testing_labels_filename);
    if (row_count != test_row_count || col_count != test_col_count) {
        fprintf(stderr, "Test dataset was formatted incorrectly.\n");
        returnValue = 1;
        goto CleanupLabel;
    }
    printf("Testing images: %u\n", test_image_count);
    printf("\tResolution: %u x %u\n", test_col_count, test_row_count);
    if (test_image_count != test_label_count) {
        printf("Testing labels: %u\n", test_label_count);
    }
    putchar('\n');

    /* INITIALISE NETWORK */

    printf("Initialising network...\n");

    if (!layers_count_set) {
        printf("Enter the number of network layers (excluding the input layer, including the output layer): ");
        int c;
        while ((c = getchar()) != EOF && c != '\n') {
            if (c >= '0' && c <= '9') {
                if ((SIZE_MAX - (c - '0')) / 10 < layers_count) continue;
                layers_count_set = true; // technically pointless right now
                layers_count *= 10;
                layers_count += c - '0';
            }
        }
        if (layers_count < 2) {
            fprintf(stderr, "Invalid number of layers.\n");
            returnValue = 1;
            goto CleanupLabel;
        }
        layer_lengths = calloc(layers_count, sizeof(size_t));
        if (layer_lengths == NULL) {
            fprintf(stderr, "Failed to allocate memory on the heap for input layer lengths.\n");
            returnValue = 1;
            goto CleanupLabel;
        }
        printf("Enter the length of each layer:\n");
        for (size_t i = 0; i < layers_count; i++) {
            printf("\tLayer %zu: ", i + 1);
            while ((c = getchar()) != EOF && c != '\n') {
                if (c >= '0' && c <= '9') {
                    if ((SIZE_MAX - (c - '0')) / 10 < layer_lengths[i]) continue;
                    layer_lengths[i] *= 10;
                    layer_lengths[i] += c - '0';
                }
            }
            if (c == EOF && i < layers_count - 1) return 1; // irreparably invalid formatting
        }
    }
    if (!learningRate_set) {
        printf("Enter the learning rate for network training: ");
        int c;
        long index = 0;
        while ((c = getchar()) != EOF && c != '\n') {
            if (c == '.') {
                index = 1;
            } else if (c >= '0' && c <= '9') {
                learningRate_set = true; // technically pointless right now
                if (index) {
                    learningRate += (double)(c - '0') * lpow(10, -index);
                    index++;
                } else {
                    learningRate *= 10.0;
                    learningRate += (double)(c - '0');
                }
            }
        }
    }
    if (!learningRateMultiplier_set) {
        printf("Enter the learning rate multiplier, to be applied to the learning rate each epoch: ");
        int c;
        long index = 0;
        while ((c = getchar()) != EOF && c != '\n') {
            if (c == '.') {
                index = 1;
            } else if (c >= '0' && c <= '9') {
                learningRateMultiplier_set = true; // technically pointless right now
                if (index) {
                    learningRateMultiplier += (double)(c - '0') * lpow(10, -index);
                    index++;
                } else {
                    learningRateMultiplier *= 10.0;
                    learningRateMultiplier += (double)(c - '0');
                }
            }
        }
    }

    inputLayer = malloc(row_count * col_count * sizeof(double));
    if (inputLayer == NULL) {
        fprintf(stderr, "Failed to allocate memory on the heap for input layer.\n");
        returnValue = 1;
        goto CleanupLabel;
    }

    // The below is all done to for contiguity and cache locality
    activated_neurons = malloc(layers_count * sizeof(double*));
    deactivated_neurons = malloc(layers_count * sizeof(double*));
    size_t total_neuron_count = 0;
    for (size_t i = 0; i < layers_count; i++) {
        total_neuron_count += layer_lengths[i];
    }
    all_activated_neurons = malloc(total_neuron_count * sizeof(double));
    all_deactivated_neurons = malloc(total_neuron_count * sizeof(double));
    if (activated_neurons == NULL || deactivated_neurons == NULL || all_activated_neurons == NULL || all_deactivated_neurons == NULL) {
        fprintf(stderr, "Failed to allocate memory on the heap for neurons.\n");
        returnValue = 1;
        goto CleanupLabel;
    }
    size_t offset = 0;
    for (size_t i = 0; i < layers_count; i++) {
        activated_neurons[i] = &all_activated_neurons[offset];
        deactivated_neurons[i] = &all_deactivated_neurons[offset];
        offset += layer_lengths[i];
    }

    weights = malloc(layers_count * sizeof(double*));
    size_t total_weight_count = 0;
    total_weight_count += (row_count * col_count) * layer_lengths[0];
    for (size_t i = 1; i < layers_count; i++) total_weight_count += layer_lengths[i - 1] * layer_lengths[i];
    all_weights = malloc(total_weight_count * sizeof(double));
    if (weights == NULL || all_weights == NULL) {
        fprintf(stderr, "Failed to allocate memory on the heap for weights.\n");
        returnValue = 1;
        goto CleanupLabel;
    }
    offset = 0;
    // case for `i == 0` is considered manually because it maps to `inputLayer` which is special
    weights[0] = &all_weights[offset];
    offset += (row_count * col_count) * layer_lengths[0];
    for (size_t i = 1; i < layers_count; i++) {
        weights[i] = &all_weights[offset];
        offset += layer_lengths[i - 1] * layer_lengths[i];
    }

    biases = malloc(layers_count * sizeof(double*));
    all_biases = malloc(total_neuron_count * sizeof(double));
    if (biases == NULL || all_biases == NULL) {
        fprintf(stderr, "Failed to allocate memory on the heap for biases.\n");
        returnValue = 1;
        goto CleanupLabel;
    }
    offset = 0;
    for (size_t i = 0; i < layers_count; i++) {
        biases[i] = &all_biases[offset];
        offset += layer_lengths[i];
    }

    intendedOutput = malloc(layer_lengths[layers_count - 1] * sizeof(double));
    if (intendedOutput == NULL) {
        fprintf(stderr, "Failed to allocate memory on the heap for vectorised labels.\n");
        returnValue = 1;
        goto CleanupLabel;
    }

    // Get persistent space for jacobians
    biasJacobians = malloc(layers_count * sizeof(double*));
    all_biases_j = malloc(total_neuron_count * sizeof(double));
    if (biasJacobians == NULL || all_biases_j == NULL) {
        fprintf(stderr, "Failed to allocate memory on the heap for bias jacobian.\n");
        returnValue = 1;
        goto CleanupLabel;
    }
    offset = 0;
    for (size_t i = 0; i < layers_count; i++) {
        biasJacobians[i] = all_biases_j + offset;
        offset += layer_lengths[i];
    }

    // Initialise weights to random (He initialisation)
    srand((unsigned int)time(NULL));
    //memset(weights[0], 0, row_count * col_count * layer_lengths[0] * sizeof(double));
    for (size_t j = 0; j < row_count * col_count * layer_lengths[0]; j++) weights[0][j] = sqrt(2.0 / (row_count * col_count)) * ((double)rand() / (double)RAND_MAX - 0.5);
    for (size_t i = 1; i < layers_count; i++) {
        //memset(weights[i], 0, layer_lengths[i - 1] * layer_lengths[i] * sizeof(double));
        for (size_t j = 0; j < layer_lengths[i - 1] * layer_lengths[i]; j++) weights[i][j] = sqrt(2.0 / layer_lengths[i - 1]) * ((double)rand() / (double)RAND_MAX - 0.5);
    }
    // Initialise biases to 0
    for (size_t i = 0; i < layers_count; i++) {
        memset(biases[i], 0, layer_lengths[i] * sizeof(double));
    }

    printf("Initialisation complete.\n");
    putchar('\n');

    /* TRAIN NETWORK */

    printf("Training...\n");


    size_t trainingCount = image_count < label_count ? image_count : label_count;
    for (size_t epoch = 1; epoch < SIZE_MAX; epoch++, learningRate *= learningRateMultiplier) {
        printf("Epoch %zu:\n", epoch);
        clock_t clockStart = clock();
        for (size_t image = 0; image < trainingCount; image++) {
            for (size_t i = 0; i < row_count * col_count; i++) inputLayer[i] = (double)(unsigned char)images[image][i] / UCHAR_MAX; // set input layer
            ForwardPass(row_count * col_count, inputLayer, layers_count, (size_t*)layer_lengths, weights, biases, deactivated_neurons, activated_neurons);

            (void)memset(intendedOutput, 0, layer_lengths[layers_count - 1] * sizeof(double));
            intendedOutput[(unsigned char)labels[image]] = 1.0;
            //double cost = Cost(layer_lengths[layers_count - 1], activated_neurons[layers_count - 1], intendedOutput); // unnecessary

            BackPropagate(layers_count, (size_t*)layer_lengths, weights, deactivated_neurons, activated_neurons, intendedOutput, biasJacobians);
            Descend(layers_count, (size_t*)layer_lengths, row_count * col_count, inputLayer, activated_neurons, weights, biases, biasJacobians, learningRate);
        }
        clock_t clockEnd = clock();
        double elapsed_ms = (double)(clockEnd - clockStart) * 1000.0 / CLOCKS_PER_SEC;
        printf("\tTraining time: %fms.\n", elapsed_ms);

        /* TESTING */

        size_t numRight = 0;
        double totalCost = 0.0;
        clockStart = clock();
        for (size_t image = 0; image < test_image_count; image++) {
            for (size_t i = 0; i < row_count * col_count; i++) inputLayer[i] = (double)(unsigned char)test_images[image][i] / UCHAR_MAX; // set input layer
            ForwardPass(row_count * col_count, inputLayer, layers_count, (size_t*)layer_lengths, weights, biases, deactivated_neurons, activated_neurons);
            
            size_t largestindex = 0;
            for (size_t i = 0; i < layer_lengths[layers_count - 1]; i++) {
                if (activated_neurons[layers_count - 1][i] > activated_neurons[layers_count - 1][largestindex]) largestindex = i;
            }
            memset(intendedOutput, 0, layer_lengths[layers_count - 1] * sizeof(double));
            intendedOutput[(unsigned char)test_labels[image]] = 1.0;
            totalCost += Cost(layer_lengths[layers_count - 1], activated_neurons[layers_count - 1], intendedOutput);
            if ((size_t)test_labels[image] == largestindex) {
                numRight++;
            }
        }
        clockEnd = clock();
        elapsed_ms = (double)(clockEnd - clockStart) * 1000.0 / CLOCKS_PER_SEC;
        printf("\tTesting time: %.0fms.\n", elapsed_ms);
        printf("\tAccuracy: %.4f\n", (double)numRight / test_image_count);
        printf("\tAvg cost: %.4f\n", totalCost / test_image_count);

        CheckSaveLabel:
        printf("\tEnter a filename to save this network to disk (.nn extension recommended): ");
        {
            char saveFilename[MAX_PATH] = { 0 };
            int c;
            size_t index = 0;
            while ((c = getchar()) != '\n' && c != EOF) {
                if (index < MAX_PATH - 1) {
                    saveFilename[index] = c;
                    index++;
                }
            }
            saveFilename[index] = '\0';
            if (saveFilename[0] != '\0') {
                FILE *savefile = fopen(saveFilename, "wb");
                if (savefile == NULL) {
                    printf("\tThere was an error opening the file \"%s\".\n", saveFilename);
                    goto CheckSaveLabel;
                }
                uint16_t endianness = 1;
                uint64_t double_size = (uint64_t)sizeof(double); // the furthest i was willing to go to conform to the C standard
                uint64_t inputSize = (uint64_t)(row_count * col_count);
                uint64_t layers_count_64 = (uint64_t)layers_count;
                uint64_t *layer_lengths_64 = malloc(layers_count_64 * sizeof(uint64_t));
                if (layer_lengths_64 == NULL) {
                    printf("\tThere was an error allocating memory while saving to file \"%s\".\n", saveFilename);
                    goto CheckSaveLabel;
                }
                if (fwrite(&endianness, sizeof(uint16_t), 1, savefile) != 1) goto fWriteError;
                if (fwrite(&inputSize, sizeof(uint64_t), 1, savefile) != 1) goto fWriteError;
                if (fwrite(&layers_count_64, sizeof(uint64_t), 1, savefile) != 1) goto fWriteError;
                if (fwrite(layer_lengths_64, sizeof(uint64_t), layers_count_64, savefile) != layers_count_64) goto fWriteError;
                if (fwrite(&double_size, sizeof(uint64_t), 1, savefile) != 1) goto fWriteError; // records `sizeof(double)` for correct interpretation during loading
                if (fwrite(all_weights, sizeof(double), total_weight_count, savefile) != total_weight_count) goto fWriteError;
                if (fwrite(all_biases, sizeof(double), total_neuron_count, savefile) != total_neuron_count) goto fWriteError;
                goto SkipFWriteError;
                fWriteError:
                printf("\tFailed to finish writing to the file \"%s\".\n", saveFilename);
                free(layer_lengths_64);
                fclose(savefile);
                goto CheckSaveLabel;
                SkipFWriteError:
                printf("\tSuccessfully written to the file \"%s\".\n", saveFilename);
                free(layer_lengths_64);
                fclose(savefile);
            }
        }
    }

    /* TERMINATE */

    CleanupLabel:

    printf("Press RETURN to terminate.");
    getchar();
    printf("Terminating...\n");

    free(all_biases_j);
    free(intendedOutput);
    free(all_biases);
    free(all_weights);
    free(all_activated_neurons);
    free(all_deactivated_neurons);
    free(inputLayer);
    free(test_labels);
    if (test_images != NULL) free(test_images[0]);
    free(test_images);
    free(labels);
    if (images != NULL) free(images[0]);
    free(images);
    free(layer_lengths);
    return returnValue;
}