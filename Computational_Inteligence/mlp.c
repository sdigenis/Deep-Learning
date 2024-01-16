#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "data.c"
#include "nn_architecture.c"

int main(int argc, char *argv[]){

    printf("\nMLP Implementation\n\n");
    // Load Data
    if (access("mlp_data.txt", F_OK) == -1) {
        printf("Data not Found !\n");
        mlp_create_and_save_data();
    }

    MLP_Data data[NUM_MLP_SAMPLES];

    mlp_load_data("mlp_data.txt", data, NUM_MLP_SAMPLES);

    printf("Data Loaded !\n\n");
    /* 1 --> [1, 0, 0, 0]
    2 --> [0, 1, 0, 0]
    3 --> [0, 0, 1, 0]
    4 --> [0, 0, 0, 1] */
    one_hot_encode(data, NUM_MLP_SAMPLES);

    /* for(int i = 0; i < 5; i++){
        printf("Label: ");
        printf("%d ", trainData[i].label);
        printf(" One hot encoding: [");
        for (int j = 0; j < K; ++j) {
            printf("%d ", trainData[i].oneHotEncodedLabels[j]);
        }
        printf("]\n");
        printf("\n");
    } */

    printf("One Hot Encoding Done !\n\n");

    // Split Data
    int train_samples = (int) NUM_MLP_SAMPLES * TRAIN_PERCENTAGE;
    int test_samples = NUM_MLP_SAMPLES - train_samples;

    float x_train[train_samples * 2];
    float y_train[train_samples * K];

    float x_test[test_samples * 2];
    float y_test[test_samples * K];

    for(int i = 0; i < train_samples; i++){
        x_train[i * 2] = data[i].x1;
        x_train[i * 2 + 1] = data[i].x2;
        for(int j = 0; j < K; j++){
            y_train[i * K + j] = data[i].oneHotEncodedLabels[j];        }
    }
    int cnt = 0;
    for(int i = train_samples; i < NUM_MLP_SAMPLES; i++){
        x_test[cnt * 2] = data[i].x1;
        x_test[cnt * 2 + 1] = data[i].x2;
        for(int j = 0; j < K; j++){
            y_test[cnt * K + j] = data[i].oneHotEncodedLabels[j];
        }
        cnt++;
    }

    printf("Data Split Done !\n");
    printf("Train Samples: %d\n", train_samples);
    printf("Test Samples: %d\n\n", test_samples);

    // MLP Part

    mlp.learning_rate = LEARNING_RATE;
    mlp.epochs = NUM_MAX_EPOCHS;

    // Train Part

    initializeMLP(&mlp);
    printf("MLP Weights Initialized !\n\n");

    gradient_descent(x_train, y_train, train_samples);

    printf("Training Done !\n\n");

    // Test Part

    int correct = 0;
    for(int i = 0; i < test_samples; i++){
        forward_pass(x_test, d, y_test, K);
        int max_index = 0;
        // printf("Predicted: ");
        // for(int j = 0; j < K; j++){
        //     printf("%f ", mlp.output[j]);
        // }
        // printf("\n");
        // printf("Actual: ");
        // for(int j = 0; j < K; j++){
        //     printf("%f ", y_test[i * K + j]);
        // }
        // printf("\n");
        // sleep(1);
        for(int j = 0; j < K; j++){
            if(mlp.output[j] > mlp.output[max_index]){
                max_index = j;
            }
        }
        if(y_test[i * K + max_index] == 1){
            correct++;
            // printf("Correct !\n\n");
        }
    }

    printf("Testing Done !\n\n");
    printf("Correct values predicted: %d\n", correct);
    printf("Total values tested: %d \n", test_samples);
    printf("Accuracy: %f\n", (float)correct / test_samples);

    return 0;
}