#define NUM_TRAIN_SAMPLES 8000
#define NUM_TEST_SAMPLES 1200
#define K 4
#define MAX_LINE_LENGTH 100

struct Data {
    double x1;
    double x2;
    int label;
    int oneHotEncodedLabels[K];
};
typedef struct Data Data;

int assign_label(double x1, double x2);
void create_train_data(struct Data trainData[], int numSamples);
void create_test_data(struct Data testData[], int numSamples);
void save_data(const char *filename, struct Data data[], int numSamples);
void one_hot_encode(Data data[], int numSamples);
void create_and_save_data();
void load_data(const char *filename, struct Data data[], int numSamples);
