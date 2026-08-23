#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include "nn_architecture.h"

float activate(float x) {
    if (ACTIVATION_FUNCTION == TANH) {
        return tanh(x);
    } 
    else if (ACTIVATION_FUNCTION == RELU) {
        return fmax(0.0, x);
    }
    else if (ACTIVATION_FUNCTION == LOGISTIC) {
        return 1.0 / (1.0 + exp(-x));
    } 
    else {
        printf("Invalid activation function.\n");
        exit(-1);
    }
}

float derivative(float x) {
    if (ACTIVATION_FUNCTION == TANH) {
        return 1.0 - tanh(x) * tanh(x);
    } 
    else if (ACTIVATION_FUNCTION == RELU) {
        return (x >= 0.0) ? 1.0 : 0.0;
    } 
    else if (ACTIVATION_FUNCTION == LOGISTIC) {
        return x * (1.0 - x);
    }
    else {
        printf("Invalid activation function.\n");
        exit(-1);
    }
}

void softmax(float *input, int length) {
    double max = input[0];

    // Find the maximum value in the input array
    for (int i = 0; i < length; ++i) {
        if (input[i] > max) {
            max = input[i];
        }
    }

    // Compute the exponentials and the sum of exponentials
    float sum = 0.0;

    // Normalize the values by dividing by the sum
    for (int i = 0; i < length; ++i) {
        input[i] /= sum;
    }
}

// Function to initialize MLP parameters randomly
void initializeMLP(MLP *mlp) {
    srand(time(NULL));

    for (int i = 0; i < d; i++) {
        for (int j = 0; j < H1; j++) {
            mlp->weights1[i][j] = ((double)rand() / RAND_MAX);
        }
        mlp->bias1[i] = ((double)rand() / RAND_MAX);
    }

    for (int i = 0; i < H1; i++) {
        for (int j = 0; j < H2; j++) {
            mlp->weights2[i][j] = ((double)rand() / RAND_MAX);
        }
        mlp->bias2[i] = ((double)rand() / RAND_MAX);
    }

    for (int i = 0; i < H2; i++) {
        for (int j = 0; j < H3; j++) {
            mlp->weights3[i][j] = ((double)rand() / RAND_MAX);
        }
        mlp->bias3[i] = ((double)rand() / RAND_MAX);
    }

    for (int i = 0; i < H3; i++) {
        for (int j = 0; j < K; j++) {
            mlp->weights4[i][j] = ((double)rand() / RAND_MAX);
        }
        mlp->bias4[i] = ((double)rand() / RAND_MAX);
    }
}

void forward_pass(float *x, int dimension, float *y, int K_dimension) {

    // Input layer 1
    for (int i = 0; i < H1; ++i) {
        mlp.hidden1[i] = 0.0;
        for (int j = 0; j < dimension; ++j) {
            mlp.hidden1[i] += x[j] * mlp.weights1[j][i];
        }
        mlp.hidden1[i] = activate(mlp.hidden1[i]) + mlp.bias1[i];
    }

    // Hidden layer 2
    for (int i = 0; i < H2; ++i) {
        mlp.hidden2[i] = 0.0;
        for (int j = 0; j < H1; ++j) {
            mlp.hidden2[i] += mlp.hidden1[j] * mlp.weights2[j][i];
        }
        mlp.hidden2[i] = activate(mlp.hidden2[i]) + mlp.bias2[i];
    }

    // Hidden layer 3
    for (int i = 0; i < H3; ++i) {
        mlp.hidden3[i] = 0.0;
        for (int j = 0; j < H2; ++j) {
            mlp.hidden3[i] += mlp.hidden2[j] * mlp.weights3[j][i];
        }
        mlp.hidden3[i] = activate(mlp.hidden3[i]) + mlp.bias3[i];
    }

    // Output layer
    for (int i = 0; i < K_dimension; ++i) {
        mlp.output[i] = 0.0;
        for (int j = 0; j < H3; ++j) {
            mlp.output[i] += mlp.hidden3[j] * mlp.weights4[j][i];
        }
        mlp.output[i] += mlp.bias4[i];
    }

    // check if input is nan
    for (int i = 0; i < K_dimension; i++) {
        if (isinf(mlp.output[i])) {
            printf("1. Forward pass mlp.output is inf\n");
            printf("MLP.output is :\n");
            for (int cnt = 0; cnt < K_dimension; cnt++) {
                printf("%lf ", mlp.output[cnt]);
            }
            printf("\n");
            sleep(1);
            break;
        }
    }
    for (int i = 0; i < K_dimension; i++) {
        if (isinf(mlp.output[i])) {
            printf("Weights: \n");
            printf("Weights are :\n");
            for (int cnt = 0; cnt < H3; cnt++) {
                for (int cnt2 = 0; cnt2 < K_dimension; cnt2++) {
                    printf("%lf ", mlp.weights4[cnt][cnt2]);
                }
                printf("\n");
            }
            printf("\n");
            sleep(1);
            break;
        }
    }
    
    softmax(mlp.output, K_dimension);
}


void backprop(float *x, int d_dimension, float *t, int K_dimension) {

    // Compute error in output layer - layer 4
    double delta_output[K];
    for (int i = 0; i < K; i++) {
        delta_output[i] = mlp.output[i] - t[i];
    }

    // Update weights and biases in output layer - layer 4
    for (int i = 0; i < H3; i++) {
        for (int j = 0; j < K; j++) {
            mlp.weights4[i][j] -= mlp.learning_rate * delta_output[j] * mlp.hidden3[i];
        }
        mlp.bias4[i] -= mlp.learning_rate * delta_output[i];
    }

    // Compute error in hidden layer 3
    double delta_hidden3[H3];
    for (int i = 0; i < H3; i++) {
        delta_hidden3[i] = 0.0;
        for (int j = 0; j < K; j++) {
            delta_hidden3[i] += delta_output[j] * mlp.weights4[i][j];
        }
        delta_hidden3[i] *= derivative(mlp.hidden3[i]);
    }

    // Update weights and biases in hidden layer 3
    for (int i = 0; i < H2; i++) {
        for (int j = 0; j < H3; j++) {
            mlp.weights3[i][j] -= mlp.learning_rate * delta_hidden3[j] * mlp.hidden2[i];
        }
        mlp.bias3[i] -= mlp.learning_rate * delta_hidden3[i];
    }

    // Compute error in hidden layer 2
    double delta_hidden2[H2];
    for (int i = 0; i < H2; i++) {
        delta_hidden2[i] = 0.0;
        for (int j = 0; j < H3; j++) {
            delta_hidden2[i] += delta_hidden3[j] * mlp.weights3[i][j];
        }
        delta_hidden2[i] *= derivative(mlp.hidden2[i]);
    }

    // Update weights and biases in hidden layer 2
    for (int i = 0; i < H1; i++) {
        for (int j = 0; j < H2; j++) {
            mlp.weights2[i][j] -= mlp.learning_rate * delta_hidden2[j] * mlp.hidden1[i];
        }
        mlp.bias1[i] -= mlp.learning_rate * delta_hidden2[i];
    }

    // Compute error in input layer - layer 1
    double delta_hidden1[H1];
    for (int i = 0; i < H1; i++) {
        delta_hidden1[i] = 0.0;
        for (int j = 0; j < H2; j++) {
            delta_hidden1[i] += delta_hidden2[j] * mlp.weights2[i][j];
        }
        delta_hidden1[i] *= derivative(mlp.hidden1[i]);
    }

    // Update weights and biases in input layer - layer 1
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < H1; j++) {
            mlp.weights1[i][j] -= LEARNING_RATE * delta_hidden1[j] * x[i];
        }
        mlp.bias1[i] -= mlp.learning_rate * delta_hidden1[i];
    }
}

// Gradient descent function
void gradient_descent(float *x_train, float *y_train, int num_samples) {
    
    int num_batches = num_samples / B;

    if(B == 0 || num_samples % B != 0){
        printf("Batch size is not a factor of number of samples.\n");
        printf("Please change the batch size or number of samples.\n");
        exit(-1);
    }
    int threshold_reached = 0;
    double previous_error = 0.0;
    double total_error = 0.0;

    printf("Number of samples: %d\n", num_samples);
    printf("Number of batches: %d\n\n", num_batches);
    for (int epoch = 0; epoch < mlp.epochs; epoch++) {
        if (threshold_reached) {
            break;
        }
        
        for (int batch = 0; batch < num_batches; batch++) {
            int start_idx = batch * B;
            int end_idx = start_idx + B;

            for (int i = start_idx; i < end_idx; i++) {
                // Forward pass
                float *y;
                y = (float*)malloc(K * sizeof(float));
                forward_pass(&x_train[i * d], d, y, K);
                
                // Backpropagation
                backprop(&x_train[i * d], d, &y_train[i * K], K);
                // threshold_reached = 1;

                // print actual and predicted values
                // printf("Actual: [");
                // for (int j = 0; j < K; ++j) {
                //     printf("%lf ", y_train[i * K + j]);
                // }
                // printf("]\n");
                // printf("Predicted: [");
                // for (int j = 0; j < K; ++j) {
                //     printf("%lf ", mlp.output[j]);
                // }
                // printf("]\n\n");

                //check if predicted values is nan
                for (int j = 0; j < K; j++) {
                    if (isnan(mlp.output[j])) {
                        printf("Index: %d\n", i);
                        printf("Predicted value is :\n");
                        for (int cnt = 0; cnt < K; cnt++) {
                            printf("%lf ", mlp.output[cnt]);
                        }
                        printf("\n");
                        sleep(1);
                        break;
                    }
                }
                
                
                for (int j = 0; j < K; ++j) {
                    total_error += -y_train[i * K + j] * log(mlp.output[j]);
                }
                free(y);
            }
        }
        //total_error /= num_batches;
        printf("Epoch %d, Training Error: %lf\n", epoch + 1, total_error);
        if (fabs(total_error - previous_error) < TERMINATION_THRESHOLD) {
            printf("Terminating at epoch %d\n", epoch + 1);
            printf("Termination Threshold: %lf Reached\n", TERMINATION_THRESHOLD);
            printf("Training Error: %lf\n", total_error);
            printf("Previous Training Error: %lf\n", previous_error);
            printf("Difference: %lf\n", total_error - previous_error);
            threshold_reached = 1;
            break;
        }
        sleep(1);
        previous_error = total_error;
        total_error = 0.0;
    }
}