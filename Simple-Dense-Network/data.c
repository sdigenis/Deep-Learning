#include <time.h>
#include <unistd.h>
#include "data.h"

int mlp_assign_label(double x1, double x2) {
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

// Function to create mlp data
void create_mlp_data(MLP_Data data[], int num_samples) {
    for (int i = 0; i < num_samples; i++) {
        double x1 = -1 + 2 * ((double)rand() / RAND_MAX);
        double x2 = -1 + 2 * ((double)rand() / RAND_MAX);

        int label = mlp_assign_label(x1, x2);

        data[i] = (MLP_Data){x1, x2, label};
    }
}

// Function to create kmeans data
void create_kmeans_data(kmeans_Data data[], int num_samples) {
    for (int i = 0; i < num_samples; i++) {
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
        data[i] = (kmeans_Data){x1, x2};
    }
}

// Function to save data to a file
void mlp_save_data(const char *filename, MLP_Data data[], int num_samples) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        fprintf(stderr, "Error opening file for writing.\n");
        exit(1);
    }

    for (int i = 0; i < num_samples; i++) {
        fprintf(file, "%.6f %.6f %d\n", data[i].x1, data[i].x2, data[i].label);
    }

    fclose(file);
}

void kmeans_save_data(const char *filename, kmeans_Data data[], int num_samples) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        fprintf(stderr, "Error opening file for writing.\n");
        exit(1);
    }

    for (int i = 0; i < num_samples; i++) {
        fprintf(file, "%.6f %.6f\n", data[i].x1, data[i].x2);
    }

    fclose(file);
}


// Function to one-hot encode class labels
void one_hot_encode(MLP_Data data[], int num_samples) {
    for (int i = 0; i < num_samples; i++) {
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

void mlp_create_and_save_data(){
    srand(time(NULL));

    MLP_Data data[NUM_MLP_SAMPLES];

    create_mlp_data(data, NUM_MLP_SAMPLES);

    mlp_save_data("mlp_data.txt", data, NUM_MLP_SAMPLES);

    printf("MLP data generated and saved to 'mlp_data.txt' .\n");

}

void kmeans_create_and_save_data(){
    srand(time(NULL));

    kmeans_Data kmeans_data[NUM_KMEANS_SAMPLES];

    create_kmeans_data(kmeans_data, NUM_KMEANS_SAMPLES);

    kmeans_save_data("kmeans_data.txt", kmeans_data, NUM_KMEANS_SAMPLES);

    printf("Kmeans data generated and saved to 'kmeans_data.txt'.\n");

}

void mlp_load_data(const char *filename, MLP_Data data[], int num_samples) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file for reading.\n");
        exit(1);
    }

    for (int i = 0; i < num_samples; i++) {
        if (fscanf(file, "%lf %lf %d", &data[i].x1, &data[i].x2, &data[i].label) != 3) {
            fprintf(stderr, "Error reading data from file.\n");
            exit(1);
        }

        // Read until the end of the line
        char line[256];
        if (fgets(line, sizeof(line), file) == NULL) {
            fprintf(stderr, "Error reading data from file.\n");
            exit(1);
        }
    }

    fclose(file);
}

void kmeans_load_data(const char *filename, kmeans_Data data[], int num_samples) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file for reading.\n");
        exit(1);
    }
    
    for (int i = 0; i < num_samples; i++) {
        if (fscanf(file, "%lf %lf", &data[i].x1, &data[i].x2) != 2) {
            fprintf(stderr, "Error reading data from file.\n");
            exit(1);
        }

        // Read until the end of the line
        char line[256];
        if (fgets(line, sizeof(line), file) == NULL) {
            fprintf(stderr, "Error reading data from file.\n");
            exit(1);
        }
    }

    fclose(file);
}