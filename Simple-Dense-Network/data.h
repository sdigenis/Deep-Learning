#define NUM_MLP_SAMPLES 8000
#define NUM_KMEANS_SAMPLES 1200
#define MAX_LINE_LENGTH 100
#define K 4

struct MLP_Data {
    double x1;
    double x2;
    int label;
    int oneHotEncodedLabels[K];
};
typedef struct MLP_Data MLP_Data;

struct KMEANS_Data {
    double x1;
    double x2;
};
typedef struct KMEANS_Data kmeans_Data;

int mlp_assign_label(double x1, double x2);
void one_hot_encode(MLP_Data data[], int num_samples);

void create_mlp_data(MLP_Data trainData[], int numSamples);
void create_kmeans_data(kmeans_Data testData[], int numSamples);


void mlp_save_data(const char *filename, struct MLP_Data data[], int num_samples);
void kmeans_save_data(const char *filename, kmeans_Data data[], int num_samples);

void mlp_create_and_save_data();
void kmeans_create_and_save_data();

void mlp_load_data(const char *filename, MLP_Data data[], int num_samples);
void kmeans_load_data(const char *filename, kmeans_Data data[], int num_samples);