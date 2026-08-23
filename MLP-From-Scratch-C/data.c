#include <time.h>
#include <unistd.h>
#include "data.h"

int assign_label(double x1, double x2) {
    // Class 1
    if ((x1 - 0.5) * (x1 - 0.5) + (x2 - 0.5) * (x2 - 0.5) < 0.2 && x1 > 0.5)
        return 1;
    if ((x1 + 0.5) * (x1 + 0.5) + (x2 + 0.5) * (x2 + 0.5) < 0.2 && x1 > -0.5)
        return 1;
    if ((x1 - 0.5) * (x1 - 0.5) + (x2 + 0.5) * (x2 + 0.5) < 0.2 && x1 > 0.5)
        return 1;
    if ((x1 + 0.5) * (x1 + 0.5) + (x2 - 0.5) * (x2 - 0.5) < 0.2 && x1 > -0.5)
        return 1;

    // Class 2
    if ((x1 - 0.5) * (x1 - 0.5) + (x2 - 0.5) * (x2 - 0.5) < 0.2 && x1 < 0.5)
        return 2;
    if ((x1 + 0.5) * (x1 + 0.5) + (x2 + 0.5) * (x2 + 0.5) < 0.2 && x1 < -0.5)
        return 2;
    if ((x1 - 0.5) * (x1 - 0.5) + (x2 + 0.5) * (x2 + 0.5) < 0.2 && x1 < 0.5)
        return 2;
    if ((x1 + 0.5) * (x1 + 0.5) + (x2 - 0.5) * (x2 - 0.5) < 0.2 && x1 < -0.5)
        return 2;

    // Class 3
    if (x1 > 0)
        return 3;

    // Class 4
    if (x1 < 0)
        return 4;

    // Default: no label
    return -1;
}

// Function to create and label training data
void create_train_data(struct Data trainData[], int numSamples) {
    for (int i = 0; i < numSamples; i++) {
        double x1 = -1 + 2 * ((double)rand() / RAND_MAX);
        double x2 = -1 + 2 * ((double)rand() / RAND_MAX);

        int label = assign_label(x1, x2);

        trainData[i] = (struct Data){x1, x2, label};
    }
}

// Function to create test data with initial label -1
void create_test_data(struct Data testData[], int numSamples) {
    for (int i = 0; i < numSamples; i++) {
        double x1, x2;
        if (i < 150) {
            x1 = 0.8 + 0.4 * ((double)rand() / RAND_MAX);
            x2 = 0.8 + 0.4 * ((double)rand() / RAND_MAX);
        } else if (i < 300) {
            x1 = 0.5 * ((double)rand() / RAND_MAX);
            x2 = 0.5 * ((double)rand() / RAND_MAX);
        } else if (i < 450) {
            x1 = 1.5 + 0.5 * ((double)rand() / RAND_MAX);
            x2 = 0.5 * ((double)rand() / RAND_MAX);
        } else if (i < 600) {
            x1 = 0.5 * ((double)rand() / RAND_MAX);
            x2 = 1.5 + 0.5 * ((double)rand() / RAND_MAX);
        } else if (i < 750) {
            x1 = 1.5 + 0.5 * ((double)rand() / RAND_MAX);
            x2 = 1.5 + 0.5 * ((double)rand() / RAND_MAX);
        } else if (i < 825) {
            x1 = 0.4 * ((double)rand() / RAND_MAX);
            x2 = 0.8 + 0.4 * ((double)rand() / RAND_MAX);
        } else if (i < 900) {
            x1 = 1.6 + 0.4 * ((double)rand() / RAND_MAX);
            x2 = 0.8 + 0.4 * ((double)rand() / RAND_MAX);
        } else if (i < 975) {
            x1 = 0.8 + 0.4 * ((double)rand() / RAND_MAX);
            x2 = 0.3 + 0.4 * ((double)rand() / RAND_MAX);
        } else if (i < 1050) {
            x1 = 0.8 + 0.4 * ((double)rand() / RAND_MAX);
            x2 = 1.3 + 0.4 * ((double)rand() / RAND_MAX);
        } else {
            x1 = 2.0 * ((double)rand() / RAND_MAX);
            x2 = 2.0 * ((double)rand() / RAND_MAX);
        }
        int label = assign_label(x1, x2);
        testData[i] = (struct Data){x1, x2, label};
    }
}

// Function to save data to a file
void save_data(const char *filename, struct Data data[], int numSamples) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        fprintf(stderr, "Error opening file for writing.\n");
        exit(1);
    }

    for (int i = 0; i < numSamples; i++) {
        fprintf(file, "%.6f %.6f %d\n", data[i].x1, data[i].x2, data[i].label);
    }

    fclose(file);
}


// Function to one-hot encode class labels
void one_hot_encode(Data data[], int numSamples) {
    for (int i = 0; i < numSamples; i++) {
        int label = data[i].label;

        // Only one-hot encode if the label is valid
        if (label >= 1 && label <= K) {
            for (int j = 0; j < K; j++) {
                if (j + 1 == label) {
                    data[i].oneHotEncodedLabels[j] = 1;
                } else {
                   data[i].oneHotEncodedLabels[j] = 0;
                }
            }
        }
    }
}

void create_and_save_data(){
    srand(time(NULL));

    struct Data trainData[NUM_TRAIN_SAMPLES];
    struct Data testData[NUM_TEST_SAMPLES];

    create_train_data(trainData, NUM_TRAIN_SAMPLES);
    create_test_data(testData, NUM_TEST_SAMPLES);

    save_data("train_data.txt", trainData, NUM_TRAIN_SAMPLES);
    save_data("test_data.txt", testData, NUM_TEST_SAMPLES);

    printf("Training and test data generated and saved to 'train_data.txt' and 'test_data.txt'.\n");

}

// Function to read data from file and save into variables
void load_data(const char *filename, struct Data data[], int numSamples) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file for reading.\n");
        exit(1);
    }

    for (int i = 0; i < numSamples; i++) {
        if (fscanf(file, "%lf %lf %d", &data[i].x1, &data[i].x2, &data[i].label) != 3) {
            fprintf(stderr, "Error reading data from file.\n");
            exit(1);
        }

        // Read until the end of the line
        char line[256];
        fgets(line, sizeof(line), file);
    }

    fclose(file);
}