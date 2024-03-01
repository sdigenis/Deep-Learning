#define d 2
#define K 4
#define B 10
#define H1 32
#define H2 16
#define H3 8
#define LEARNING_RATE 0.0001
#define TANH 1
#define LOGISTIC 2
#define RELU 3
#define ACTIVATION_FUNCTION RELU
#define TERMINATION_THRESHOLD 0.00001  
#define TRAIN_PERCENTAGE (0.8)
#define NUM_MAX_EPOCHS 1000

typedef struct {
    float weights1[d][H1];
    float weights2[H1][H2];
    float weights3[H2][H3];
    float weights4[H3][K];

    float bias1[H1];
    float bias2[H2];
    float bias3[H3];
    float bias4[K];

    float hidden1[H1];
    float hidden2[H2];
    float hidden3[H3];
    float output[K];

    double learning_rate;
    int epochs;
    
} MLP;

MLP mlp;

float activate(float x);
float derivative(float x);
void softmax(float *input, int length);
void initializeMLP(MLP *mlp);
float calculate_norm(double *gradients, int size);
void clip_gradients(double *gradients, int size, float max_norm);
void forward_pass(float *x, int d_dimension, float *y, int K_dimension);
void backprop(float *x, int d_dimension, float *t, int K_dimension);
void gradient_descent(float *x_train, float *y_train, int num_samples);