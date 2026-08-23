#define d 4
#define K 4
#define B 1
#define H1 16
#define H2 8
#define H3 4
#define LEARNING_RATE 0.001
#define TANH 1
#define LOGISTIC 2
#define RELU 3
#define ACTIVATION_FUNCTION TANH
#define TERMINATION_THRESHOLD 0.0001  

typedef struct {
    double weights1[d][H1];
    double weights2[H1][H2];
    double weights3[H2][H3];
    double weights4[H3][K];

    double bias1[H1];
    double bias2[H2];
    double bias3[H3];
    double bias4[K];

    float hidden1[H1];
    float hidden2[H2];
    float hidden3[H3];
    float output[B * K];

    double learning_rate;
    int epochs;
    
} MLP;

MLP mlp;

float activate(float x);
float derivative(float x);
void softmax(float *input, int length);
void initializeMLP(MLP *mlp);
void forward_pass(float *x, int d_dimension, float *y, int K_dimension);
void backprop(float *x, int d_dimension, float *t, int K_dimension);
void gradient_descent(float *x_train, float *y_train, int num_samples);

