#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "data.c"
#include <time.h>

#define M 9
#define ITERATIONS 1500

struct Cluster {
    double centroid_x1;
    double centroid_x2;
    int count;
    double sum_distance;
};

void clusters_initialization(struct Cluster clusters[M], kmeans_Data data[NUM_KMEANS_SAMPLES]) {
    srand(time(NULL));
    for (int i = 0; i < M; i++) {
        int index = rand() % NUM_KMEANS_SAMPLES;
        clusters[i].centroid_x1 = data[index].x1;
        clusters[i].centroid_x2 = data[index].x2;
        clusters[i].count = 0;
        clusters[i].sum_distance = 0.0;
    }
}

// Function to assign each data point to the nearest cluster
void cluster_assignment(struct Cluster clusters[M], kmeans_Data data[NUM_KMEANS_SAMPLES]) {
    for (int i = 0; i < NUM_KMEANS_SAMPLES; i++) {
        double minDistance = INFINITY;
        int minIndex = 0;

        for (int j = 0; j < M; j++) {
            double distance = sqrt(pow(data[i].x1 - clusters[j].centroid_x1, 2) + 
                                   pow(data[i].x2 - clusters[j].centroid_x2, 2));

            if (distance < minDistance) {
                minDistance = distance;
                minIndex = j;
            }
        }

        // Assign data point to the nearest cluster
        clusters[minIndex].count++;
        clusters[minIndex].sum_distance += minDistance;
        clusters[minIndex].centroid_x1 = (clusters[minIndex].centroid_x1 * (clusters[minIndex].count - 1) + data[i].x1) / clusters[minIndex].count;
        clusters[minIndex].centroid_x2 = (clusters[minIndex].centroid_x2 * (clusters[minIndex].count - 1) + data[i].x2) / clusters[minIndex].count;
    }
}

void print_clusters(struct Cluster clusters[M]) {
    for (int i = 0; i < M; i++) {
        printf("Cluster %d Centroid: (%lf, %lf)\n", i + 1, clusters[i].centroid_x1, clusters[i].centroid_x2);
    }
}


int main() {

     // Load Data
    if (access("kmeans_data.txt", F_OK) == -1) {
        printf("Data not Found !\n");
        kmeans_create_and_save_data();
    }

    kmeans_Data data[NUM_KMEANS_SAMPLES];

    kmeans_load_data("kmeans_data.txt", data, NUM_KMEANS_SAMPLES);

    printf("Data Loaded !\n\n");

    struct Cluster clusters[M];
    clusters_initialization(clusters, data);

    for (int iteration = 0; iteration < ITERATIONS; iteration++) {
        for (int i = 0; i < M; i++) {
            clusters[i].count = 0;
            clusters[i].sum_distance = 0.0;
        }
        cluster_assignment(clusters, data);
    }

    // Print final cluster centroids
    printf("Final Cluster Centroids:\n");
    print_clusters(clusters);

    // Calculate and print the error of clustering
    double total_error = 0.0;
    for (int i = 0; i < M; i++) {
        printf("Cluster %d Error: %lf\n", i + 1, clusters[i].sum_distance);
        total_error += clusters[i].sum_distance;
    }
    printf("Total Error of Clustering: %lf\n", total_error);

    return 0;
}