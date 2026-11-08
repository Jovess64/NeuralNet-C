#ifndef _MAIN_FILEHANDLING_SHARED_H
    #define _MAIN_FILEHANDLING_SHARED_H


    #include <stdint.h>
    #include <stdbool.h>
    #ifndef MAX_PATH // this is here in case the platform has a different max path that's already defined; that should be used instead
        #define MAX_PATH 260 /* including null terminator */
    #endif

    // put pointers to variables in here
    typedef struct GetConfigContext {
        bool *learningRate_set; // should be set to `false` before calling `GetConfig()`
        double *learningRate;
        bool *learningRateMultiplier_set; // should be set to `false` before calling `GetConfig()`
        double *learningRateMultiplier;
        bool *layersCount_set; // should be set to `false` before calling `GetConfig()`
        size_t *layersCount;
        size_t **layerLengths;
        char *training_images_filename;
        char *training_labels_filename;
        char *testing_images_filename;
        char *testing_labels_filename;
    } GetConfigContext;


#endif