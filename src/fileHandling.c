#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <stdbool.h>
#include "config_context.h" // contains `MAX_PATH`

/*
    Contains file handling functions
*/

// Returns 1 if little-endian, 0 if big-endian
int GetEndianness(void) {
    uint16_t x = 1;
    uint8_t *y = (uint8_t*)&x;
    return *y == 1;
}

// Assumes `CHAR_BIT == 8`
// Equivalent to `ntohl(uint32_t netlong)`
uint32_t CorrectEndiannessFromBig(uint32_t bignum) {
    if (!GetEndianness()) return bignum;
    bignum = ((bignum & 0x000000ffu) << 24) | ((bignum & 0x0000ff00u) << 8) | ((bignum & 0x00ff0000u) >> 8) | ((bignum & 0xff000000u) >> 24);
    return bignum;
}

#pragma pack(push, 1)
typedef struct IDX_Images_Header {
    uint32_t magic; // 0x00000803 big-endian
    uint32_t image_count;
    uint32_t row_count;
    uint32_t col_count;
} IDX_Images_Header;
typedef struct IDX_Labels_Header {
    uint32_t magic; // 0x00000801 big-endian
    uint32_t label_count;
} IDX_Labels_Header;
#pragma pack(pop)

// Where `char **results = GetImages();`, only free `results[0]` and `results`, if and only if function returns non-NULL
// Result is an array of images
//
// NOTE: consider using `restrict` on arguments
char **GetImages(const char *filename, uint32_t *image_count, uint32_t *row_count, uint32_t *col_count) {
    FILE *f = fopen(filename, "rb");
    if (f == NULL) return NULL;
    IDX_Images_Header header;
    if (fread(&header, sizeof(header), 1, f) != 1) { (void)fclose(f); return NULL; }
    if (header.magic != CorrectEndiannessFromBig(0x00000803)) { (void)fclose(f); return NULL; };
    header.image_count = CorrectEndiannessFromBig(header.image_count);
    header.row_count = CorrectEndiannessFromBig(header.row_count);
    header.col_count = CorrectEndiannessFromBig(header.col_count);
    char **data = malloc(header.image_count * sizeof(char*));
    if (data == NULL) { (void)fclose(f); return NULL; }
    size_t readbytes = header.image_count * header.row_count * header.col_count;
    char *rawdata = malloc(readbytes);
    if (rawdata == NULL) { free(data); (void)fclose(f); return NULL; };
    //(void)setvbuf(f, NULL, _IONBF, 0); // Turn off `fread()` buffering for performance; failure is acceptable
    if (fread(rawdata, sizeof(char), readbytes, f) != readbytes) { free(rawdata); free(data); (void)fclose(f); return NULL; }
    (void)fclose(f);
    for (size_t i = 0; i < header.image_count; i++) data[i] = rawdata + (i * (header.row_count * header.col_count));
    if (image_count != NULL) *image_count = header.image_count;
    if (row_count != NULL) *row_count = header.row_count;
    if (col_count != NULL) *col_count = header.col_count;
    return data;
}

// Where `char *results = GetLabels();`, only free `results`, if and only if function returns non-NULL
// Result is an array of single-byte labels
//
// NOTE: consider using `restrict` on arguments
char *GetLabels(const char *filename, uint32_t *label_count) {
    FILE *f = fopen(filename, "rb");
    if (f == NULL) return NULL;
    IDX_Labels_Header header;
    if (fread(&header, sizeof(header), 1, f) != 1) { (void)fclose(f); return NULL; }
    if (header.magic != CorrectEndiannessFromBig(0x00000801)) { (void)fclose(f); return NULL; };
    header.label_count = CorrectEndiannessFromBig(header.label_count);
    char *labels = malloc(header.label_count); // Since each label is exactly a single byte
    if (labels == NULL) { (void)fclose(f); return NULL; };
    //(void)setvbuf(f, NULL, _IONBF, 0); // Turn off `fread()` buffering for performance; failure is acceptable
    if (fread(labels, sizeof(char), header.label_count, f) != header.label_count) { free(labels); (void)fclose(f); return NULL; }
    (void)fclose(f);
    if (label_count != NULL) *label_count = header.label_count;
    return labels;
}

// from `helpers.c`
// returns a double to the power of a long
extern double lpow(double a, long b);

// Returns 0 on success, 1 on failure
// Only fails if something has gone catastrophically wrong (e.g. malloc failure or irreparably invalidly formatted config file)
//
// NOTE: must be manually changed when GetConfigContext is changed
//       assumes fields in `context` are set to `0` or `NULL` or `'\0'` already
int GetConfig(const char *config_filename, GetConfigContext *context) {
    FILE *configfile = fopen(config_filename, "r"); // load filenames if set, otherwise get input later during runtime
    if (configfile == NULL) {
        return 0;
    }
    // gets chars individually to prevent newline being included
    // doubles use index as a counter for how many digits past a `'.'` (1 means the first decimal place)
    int c;
    long index = 0; // type is long because that's what `fseek()` returns so it seems natural for storing indices of characters in a file
    // learningRate
    while ((c = fgetc(configfile)) != EOF && c != '\n') {
        if (c == '.') {
            index = 1;
        } else if (c >= '0' && c <= '9') {
            *context->learningRate_set = true;
            if (index) {
                *context->learningRate += (double)(c - '0') * lpow(10, -index);
                index++;
            } else {
                *context->learningRate *= 10.0;
                *context->learningRate += (double)(c - '0');
            }
        }
    }
    if (c == EOF) return 0;
    index = 0;
    // learningRateMultiplier
    while ((c = fgetc(configfile)) != EOF && c != '\n') {
        if (c == '.') {
            index = 1;
        } else if (c >= '0' && c <= '9') {
            *context->learningRateMultiplier_set = true;
            if (index) {
                *context->learningRateMultiplier += (double)(c - '0') * lpow(10, -index);
                index++;
            } else {
                *context->learningRateMultiplier *= 10.0;
                *context->learningRateMultiplier += (double)(c - '0');
            }
        }
    }
    if (c == EOF) return 0;
    // layersCount
    while ((c = fgetc(configfile)) != EOF && c != '\n') {
        if (c >= '0' && c <= '9') {
            if ((SIZE_MAX - (c - '0')) / 10 < *context->layersCount) continue;
            *context->layersCount_set = true;
            *context->layersCount *= 10;
            *context->layersCount += c - '0';
        }
    }
    if (c == EOF && *context->layersCount <= 0) return 0;
    // layerLengths
    if (*context->layersCount) {
        *context->layerLengths = calloc((*context->layersCount), sizeof(size_t));
        if ((*context->layerLengths) == NULL) {
            return 1;
        }
        for (size_t i = 0; i < *context->layersCount; i++) {
            // layerLengths[i]
            while ((c = fgetc(configfile)) != EOF && c != '\n') {
                if (c >= '0' && c <= '9') {
                    if ((SIZE_MAX - (c - '0')) / 10 < (*context->layerLengths)[i]) continue;
                    (*context->layerLengths)[i] *= 10;
                    (*context->layerLengths)[i] += c - '0';
                }
            }
            if (c == EOF && i < *context->layersCount - 1) return 1; // irreparably invalid formatting
        }
    }
    if (c == EOF) return 0;
    // `fields` is the variable to be filled by the contents of the nth line
    char *fields[4] = { context->training_images_filename, context->training_labels_filename, context->testing_images_filename, context->testing_labels_filename };
    index = 0;
    size_t currField = 0;
    while ((c = fgetc(configfile)) != EOF && currField < sizeof(fields) / sizeof(fields[0])) {
        if (c == '\n') {
            fields[currField][index] = '\0';
            currField++;
            index = 0;
            continue;
        }
        if (index < MAX_PATH - 1) {
            fields[currField][index] = c;
            index++;
        }
    }
    fields[currField][index] = '\0';
    fclose(configfile);

    return 0;
}