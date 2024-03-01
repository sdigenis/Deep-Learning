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
    else if (ACTIVATION_FUNCTION == LOGISTIC) {
        return 1.0 / (1.0 + exp(-x));
    } 
    else if (ACTIVATION_FUNCTION == RELU) {
        return fmax(0.0, x);
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
    else if (ACTIVATION_FUNCTION == LOGISTIC) {
        return (1.0 / (1.0 + exp(-x))) * (1.0 - (1.0 / (1.0 + exp(-x))));
    }
    else if (ACTIVATION_FUNCTION == RELU) {
        return (x >= 0.0) ? 1.0 : 0.0;
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
    for (int i = 0; i < length; ++i) {
        input[i] = exp(input[i] - max);  // Subtracting max for numerical stability
        sum += input[i];
    }

    // Normalize the values by dividing by the sum
    for (int i = 0; i < length; ++i) {
        input[i] /= sum;
    }
}

// Function to initialize MLP parameters randomly
void initializeMLP(MLP *mlp) {
    srand(time(NULL));

    if(ACTIVATION_FUNCTION == TANH || ACTIVATION_FUNCTION == LOGISTIC){
        // glorot / xavier initialization
        float limit1 = sqrt(6.0 / (d + H1));
        float limit2 = sqrt(6.0 / (H1 + H2));
        float limit3 = sqrt(6.0 / (H2 + H3));
        float limit4 = sqrt(6.0 / (H3 + K));

        for (int i = 0; i < H1; i++) {
            for (int j = 0; j < d; j++) {
                mlp->weights1[j][i] = ((double)rand() / RAND_MAX) * 2 * limit1 - limit1;
            }
            mlp->bias1[i] = 0.0;
        }
        for (int i = 0; i < H2; i++) {
            for (int j = 0; j < H1; j++) {
                mlp->weights2[j][i] = ((double)rand() / RAND_MAX) * 2 * limit2 - limit2;
            }
            mlp->bias2[i] = 0.0;
        }
        for (int i = 0; i < H3; i++) {
            for (int j = 0; j < H2; j++) {
                mlp->weights3[j][i] = ((double)rand() / RAND_MAX) * 2 * limit3 - limit3;
            }
            mlp->bias3[i] = 0.0;
        }
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < H3; j++) {
                mlp->weights4[j][i] = ((double)rand() / RAND_MAX) * 2 * limit4 - limit4;
            }
            mlp->bias4[i] = 0.0;
        }
    }
    else{
        // He initialization 
        float variance1 = sqrt(2.0 / H1);
        float variance2 = sqrt(2.0 / H2);
        float variance3 = sqrt(2.0 / H3);
        float variance4 = sqrt(2.0 / K);

        for (int i = 0; i < H1; i++) {
            for (int j = 0; j < d; j++) {
                mlp->weights1[j][i] = variance1 * ((double)rand() / RAND_MAX);
            }
            mlp->bias1[i] = 0.0;
        }
        for (int i = 0; i < H2; i++) {
            for (int j = 0; j < H3; j++) {
                mlp->weights2[j][i] = variance2 * ((double)rand() / RAND_MAX);
            }
            mlp->bias2[i] = 0.0;
        }
        for (int i = 0; i < H3; i++) {
            for (int j = 0; j < H2; j++) {
                mlp->weights3[j][i] = variance3 * ((double)rand() / RAND_MAX);
            }
            mlp->bias3[i] = 0.0;
        }
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < H3; j++) {
                mlp->weights4[j][i] = variance4 * ((double)rand() / RAND_MAX);
            }
            mlp->bias4[i] = 0.0;
        }
    }
}
float calculate_norm(double *gradients, int size) {
    float norm = 0.0f;
    for (int i = 0; i < size; i++) {
        norm += gradients[i] * gradients[i];
    }
    return sqrt(norm);
}

// Clip gradients to a specified threshold
void clip_gradients(double *gradients, int size, float max_norm) {
    float current_norm = calculate_norm(gradients, size);

    // Check if the norm exceeds the threshold
    if (current_norm > max_norm) {
        // Scale the gradients to ensure the norm is below the threshold
        float scale_factor = max_norm / current_norm;
        for (int i = 0; i < size; i++) {
            gradients[i] *= scale_factor;
        }
    }
}

void forward_pass(float *x, int dimension, float *y, int K_dimension) {

    // Input layer 1
    for (int i = 0; i < H1; i++) {
        mlp.hidden1[i] = 0.0;
        for (int j = 0; j < dimension; j++) {
            mlp.hidden1[i] += x[j] * mlp.weights1[j][i];
        }
        mlp.hidden1[i] = activate(mlp.hidden1[i] + mlp.bias1[i]);
    }

    // check if hidden1 is nan
    for (int i = 0; i < H1; i++) {
        if (isnan(mlp.hidden1[i])) {
            printf("1. Forward pass hidden1 is nan\n");
            printf("MLP.hidden1 is :\n");
            for (int cnt = 0; cnt < H1; cnt++) {
                printf("%lf ", mlp.hidden1[cnt]);
            }
            printf("\n\n");

            printf("Bias1 are :\n");
            for (int cnt = 0; cnt < H1; cnt++) {
                printf("%lf ", mlp.bias1[cnt]);
            }
            printf("\n\n");
            sleep(1);

            printf("Weights: \n");
            printf("Weights1 are :\n");
            for (int cnt = 0; cnt < dimension; cnt++) {
                for (int cnt2 = 0; cnt2 < H1; cnt2++) {
                    printf("%lf ", mlp.weights1[cnt][cnt2]);
                }
                printf("\n");
            }
            printf("\n\n");
            sleep(1);
            break;
        }
    }

    // Hidden layer 2
    for (int i = 0; i < H2; i++) {
        mlp.hidden2[i] = 0.0;
        for (int j = 0; j < H1; j++) {
            mlp.hidden2[i] += mlp.hidden1[j] * mlp.weights2[j][i];
        }
        mlp.hidden2[i] = activate(mlp.hidden2[i] + mlp.bias2[i]);
    }

    // check if hidden2 is nan
    for (int i = 0; i < H2; i++) {
        if (isnan(mlp.hidden2[i])) {
            printf("2. Forward pass hidden2 is nan\n");
            printf("MLP.hidden2 is :\n");
            for (int cnt = 0; cnt < H2; cnt++) {
                printf("%lf ", mlp.hidden2[cnt]);
            }
            printf("\n\n");
            sleep(1);

            printf("Weights: \n");
            printf("Weights2 are :\n");
            for (int cnt = 0; cnt < H2; cnt++) {
                for (int cnt2 = 0; cnt2 < H1; cnt2++) {
                    printf("%lf ", mlp.weights2[cnt2][cnt]);
                }
                printf("\n");
            }
            printf("\n\n");
            sleep(1);
            break;
        }
    }

    

    // Hidden layer 3
    for (int i = 0; i < H3; i++) {
        mlp.hidden3[i] = 0.0;
        for (int j = 0; j < H2; j++) {
            mlp.hidden3[i] += mlp.hidden2[j] * mlp.weights3[j][i];
        }
        mlp.hidden3[i] = activate(mlp.hidden3[i] + mlp.bias3[i]);
    }

    // check if hidden3 is nan
    for (int i = 0; i < H3; i++) {
        if (isnan(mlp.hidden3[i])) {
            printf("3. Forward pass hidden3 is nan\n");
            printf("MLP.hidden3 is :\n");
            for (int cnt = 0; cnt < H3; cnt++) {
                printf("%lf ", mlp.hidden3[cnt]);
            }
            printf("\n\n");
            sleep(1);

            printf("Weights: \n");
            printf("Weights3 are :\n");
            for (int cnt = 0; cnt < H2; cnt++) {
                for (int cnt2 = 0; cnt2 < H3; cnt2++) {
                    printf("%lf ", mlp.weights3[cnt][cnt2]);
                }
                printf("\n");
            }
            printf("\n\n");
            sleep(1);
            break;
        }
    }

    // Output layer
    for (int i = 0; i < K_dimension; i++) {
        mlp.output[i] = 0.0;
        for (int j = 0; j < H3; j++) {
            mlp.output[i] += mlp.hidden3[j] * mlp.weights4[j][i];
        }
        mlp.output[i] += mlp.output[i] + mlp.bias4[i];
    }

    // check if output is nan
    for (int i = 0; i < K; i++) {
        if (isnan(mlp.output[i])) {
            printf("4. Forward pass output is nan\n");
            printf("MLP.output is :\n");
            for (int cnt = 0; cnt < K; cnt++) {
                printf("%lf ", mlp.output[cnt]);
            }
            printf("\n\n");
            sleep(1);

            printf("Weights: \n");
            printf("Weights4 are :\n");
            for (int cnt = 0; cnt < H3; cnt++) {
                for (int cnt2 = 0; cnt2 < K_dimension; cnt2++) {
                    printf("%lf ", mlp.weights4[cnt][cnt2]);
                }
                printf("\n");
            }
            printf("\n\n");
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
    clip_gradients(delta_output, K, 1.0);

    // Update weights and biases in output layer - layer 4
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < H3; j++) {
            mlp.weights4[j][i] -= mlp.learning_rate * delta_output[i] * mlp.hidden3[j];
        }
        mlp.bias4[i] -= mlp.learning_rate * delta_output[i];
    }
    
    int exit = 0;
    // check if weights4 are nan
    for (int i = 0; i < H3; i++) {
        if(exit){
            break;
        }
        for(int j = 0; j < K; j++){
            if(isnan(mlp.weights4[i][j])) {
                printf("4. Backprop weights4 is nan\n");
                // Additional print statements for debugging
                printf("mlp_output: %lf\n", mlp.output[j]);
                printf("Delta_output: %lf\n", delta_output[j]);
                printf("mlp.hidden3[%d]: %lf\n", i, mlp.hidden3[i]);
                printf("mlp.weights4[%d][%d]: %lf\n", i, j, mlp.weights4[i][j]);
                printf("\n");
                sleep(1);
                exit = 1;
            }
        }
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
    clip_gradients(delta_hidden3, H3, 1.0);

    // Update weights and biases in hidden layer 3
    for (int i = 0; i < H3; i++) {
        for (int j = 0; j < H2; j++) {
            mlp.weights3[j][i] -= mlp.learning_rate * delta_hidden3[i] * mlp.hidden2[j];
        }
        mlp.bias3[i] -= mlp.learning_rate * delta_hidden3[i];
    }

    exit = 0;
    // check if weights3 are nan
    for (int i = 0; i < H2; i++) {
        for(int j = 0; j < H3; j++){
            if(isnan(mlp.weights3[i][j])) {
                printf("3. Backprop weights3 is nan\n");
                // Additional print statements for debugging
                printf("Delta_hidden3: %lf\n", delta_hidden3[j]);
                printf("mlp.hidden2[%d]: %lf\n", i, mlp.hidden2[i]);
                printf("mlp.weights3[%d][%d]: %lf\n", i, j, mlp.weights3[i][j]);
                printf("\n");
                sleep(1);
                exit = 1;
            }
        }
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
    clip_gradients(delta_hidden2, H2, 1.0);

    // Update weights and biases in hidden layer 2
    for (int i = 0; i < H2; i++) {
        for (int j = 0; j < H1; j++) {
            mlp.weights2[j][i] -= mlp.learning_rate * delta_hidden2[i] * mlp.hidden1[j];
        }
        mlp.bias2[i] -= mlp.learning_rate * delta_hidden2[i];
    }

    exit = 0;
    // check if weights2 are nan
    for (int i = 0; i < H1; i++) {
        for(int j = 0; j < H2; j++){
            if(isnan(mlp.weights2[i][j])) {
                printf("2. Backprop weights2 is nan\n");
                // Additional print statements for debugging
                printf("Delta_hidden2: %lf\n", delta_hidden2[j]);
                printf("mlp.hidden1[%d]: %lf\n", i, mlp.hidden1[i]);
                printf("mlp.weights2[%d][%d]: %lf\n", i, j, mlp.weights2[i][j]);
                printf("\n");
                sleep(1);
                exit = 1;
            }
        }
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
    clip_gradients(delta_hidden1, H1, 1.0);

    // Update weights and biases in input layer - layer 1
    for (int i = 0; i < H1; i++) {
        for (int j = 0; j < d; j++) {
            mlp.weights1[j][i] -= LEARNING_RATE * delta_hidden1[i] * x[j];
        }
        mlp.bias1[i] -= mlp.learning_rate * delta_hidden1[i];
    }
    exit = 0;
    // check if weights1 are nan
    for (int i = 0; i < d_dimension; i++) {
        for(int j = 0; j < H1; j++){
            if(isnan(mlp.weights1[i][j])) {
                printf("1. Backprop weights1 is nan\n");
                // Additional print statements for debugging
                printf("Delta_hidden1: %lf\n", delta_hidden1[j]);
                printf("x[%d]: %lf\n", i, x[i]);
                printf("mlp.weights1[%d][%d]: %lf\n", i, j, mlp.weights1[i][j]);
                printf("\n");
                sleep(1);
                exit = 1;
            }
        }
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

    printf("Number of batches: %d\n\n", num_batches);
    for (int epoch = 0; epoch < mlp.epochs; epoch++) {
        if (threshold_reached) {
            break;
        }
        
        for (int batch = 0; batch < num_batches; batch++) {
            int start_idx = batch * B;
            int end_idx = start_idx + B;
            float *y;
            y = (float*)malloc(K * sizeof(float) * B);
            
            for (int i = start_idx; i < end_idx; i++) {
                // Forward pass
                
                
                // printf("i: %d\n", i); 
                forward_pass(&x_train[i * d], d, y, K);

                /* for (int cnt = 0; cnt < H1; cnt++) {
                    if (isnan(mlp.hidden1[cnt])) {
                        printf("After Forward pass hidden1 is nan\n");
                        printf("MLP.hidden1 is :\n");
                        for (int cnt2 = 0; cnt2 < H1; cnt2++) {
                            printf("%lf ", mlp.hidden1[cnt2]);
                        }
                        printf("\n\n");
                        sleep(1);
                        break;
                    }
                } */
                
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
                    total_error += -y_train[i * K + j] * log(mlp.output[j] + 1e-15);
                }
                free(y);
                if(isnan(total_error)){
                    printf("Total error is nan\n");
                    printf("Index: %d\n", i);
                    printf("y_train is :\n");
                    for (int cnt = 0; cnt < K; cnt++) {
                        printf("%lf ", y_train[i * K + cnt]);
                    }
                    printf("\n");
                    printf("mlp.output is :\n");
                    for (int cnt = 0; cnt < K; cnt++) {
                        printf("%lf ", mlp.output[cnt]);
                    }
                    printf("\n");
                    sleep(1);
                    break;
                }
            }
        }
        total_error /= num_batches;
        printf("Epoch %d, Training Error: %lf\n", epoch + 1, total_error);
        if (fabs(total_error - previous_error) < TERMINATION_THRESHOLD && epoch >= NUM_MAX_EPOCHS) {
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