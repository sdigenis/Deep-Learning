#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "data.c"
#include "nn_architecture.c"

int main(){
    // Load Data
    if (access("train_data.txt", F_OK) == -1) {
        printf("Train and Test Data not Found !\n");
        create_and_save_data();
    }

    Data train_data[NUM_TRAIN_SAMPLES];
    Data test_data[NUM_TEST_SAMPLES];

    load_data("train_data.txt", train_data, NUM_TRAIN_SAMPLES);
    load_data("test_data.txt", test_data, NUM_TEST_SAMPLES);

    printf("Train and Test Data Loaded !\n");

    one_hot_encode(train_data, NUM_TRAIN_SAMPLES);
    one_hot_encode(test_data, NUM_TEST_SAMPLES);

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

    float x_train[NUM_TRAIN_SAMPLES * 2];
    float y_train[NUM_TRAIN_SAMPLES * K];

    for(int i = 0; i < NUM_TRAIN_SAMPLES; i++){
        x_train[i * 2] = train_data[i].x1;
        x_train[i * 2 + 1] = train_data[i].x2;
        for(int j = 0; j < K; j++){
            y_train[i * K + j] = train_data[i].oneHotEncodedLabels[j];        }
    }

    // MLP Part

    mlp.learning_rate = LEARNING_RATE;
    mlp.epochs = 700;

    // Train Part

    initializeMLP(&mlp);
    printf("MLP Weights Initialized !\n\n");

    gradient_descent(x_train, y_train, NUM_TRAIN_SAMPLES);

    printf("Training Done !\n\n");

    // Test Part

    float x_test[NUM_TEST_SAMPLES * 2];
    float y_test[NUM_TEST_SAMPLES * K];

    for(int i = 0; i < NUM_TEST_SAMPLES; i++){
        x_test[i * 2] = test_data[i].x1;
        x_test[i * 2 + 1] = test_data[i].x2;
        for(int j = 0; j < K; j++){
            y_test[i * K + j] = test_data[i].oneHotEncodedLabels[j];
        }
    }

    int correct = 0;
    for(int i = 0; i < NUM_TEST_SAMPLES; i++){
        forward_pass(x_test, d, y_test, K);
        int max_index = 0;
        for(int j = 0; j < K; j++){
            if(mlp.output[j] > mlp.output[max_index]){
                max_index = j;
            }
        }
        if(y_test[i * K + max_index] == 1){
            correct++;
        }
    }

    printf("Accuracy: %f\n", (float)correct / NUM_TEST_SAMPLES);

    return 0;
}