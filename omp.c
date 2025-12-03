#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define NUM_POINTS 100000
#define NUM_CLUSTERS 3
#define MAX_ITERATIONS 100
#define DIMENSIONS 20

typedef struct {
    double coordinates[DIMENSIONS];
    int cluster_id;
} Point;

typedef struct {
    double centroid[DIMENSIONS];
    int point_count;
} Cluster;

// Function to calculate Euclidean distance
double euclidean_distance(double *point1, double *point2, int dims) {
    double sum = 0.0;
    for (int i = 0; i < dims; i++) {
        double diff = point1[i] - point2[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

// Initialize points with random data
void initialize_points(Point *points, int num_points) {
    srand(42); // Fixed seed for reproducibility
    for (int i = 0; i < num_points; i++) {
        for (int j = 0; j < DIMENSIONS; j++) {
            points[i].coordinates[j] = (double)rand() / RAND_MAX * 100.0;
        }
        points[i].cluster_id = -1;
    }
}

// Initialize clusters with random points as initial centroids
void initialize_clusters(Cluster *clusters, Point *points, int num_clusters) {
    for (int i = 0; i < num_clusters; i++) {
        int random_point = rand() % NUM_POINTS;
        for (int j = 0; j < DIMENSIONS; j++) {
            clusters[i].centroid[j] = points[random_point].coordinates[j];
        }
        clusters[i].point_count = 0;
    }
}

// Parallel K-means algorithm using OpenMP
void kmeans_parallel(Point *points, Cluster *clusters, int num_points, int num_clusters) {
    int changed;
    int iterations = 0;
    
    do {
        changed = 0;
        
        // Assignment step: Assign each point to nearest cluster (PARALLEL)
        #pragma omp parallel for schedule(dynamic, 100) reduction(||:changed)
        for (int i = 0; i < num_points; i++) {
            double min_distance = 1e9;
            int closest_cluster = -1;
            
            for (int j = 0; j < num_clusters; j++) {
                double dist = euclidean_distance(points[i].coordinates,
                                                clusters[j].centroid,
                                                DIMENSIONS);
                if (dist < min_distance) {
                    min_distance = dist;
                    closest_cluster = j;
                }
            }
            
            if (points[i].cluster_id != closest_cluster) {
                points[i].cluster_id = closest_cluster;
                changed = 1;
            }
        }
        
        // Update step: Recalculate cluster centroids (PARALLEL)
        for (int i = 0; i < num_clusters; i++) {
            double sum[DIMENSIONS] = {0};
            int count = 0;
            
            #pragma omp parallel
            {
                double local_sum[DIMENSIONS] = {0};
                int local_count = 0;
                
                #pragma omp for nowait
                for (int j = 0; j < num_points; j++) {
                    if (points[j].cluster_id == i) {
                        for (int k = 0; k < DIMENSIONS; k++) {
                            local_sum[k] += points[j].coordinates[k];
                        }
                        local_count++;
                    }
                }
                
                #pragma omp critical
                {
                    for (int k = 0; k < DIMENSIONS; k++) {
                        sum[k] += local_sum[k];
                    }
                    count += local_count;
                }
            }
            
            if (count > 0) {
                for (int k = 0; k < DIMENSIONS; k++) {
                    clusters[i].centroid[k] = sum[k] / count;
                }
                clusters[i].point_count = count;
            }
        }
        
        iterations++;
    } while (changed && iterations < MAX_ITERATIONS);
    
    printf("K-means completed in %d iterations\n", iterations);
}

// Print results
void print_results(Point *points, Cluster *clusters, int num_points, int num_clusters) {
    printf("\nFinal cluster centroids:\n");
    for (int i = 0; i < num_clusters; i++) {
        printf("Cluster %d: (", i);
        for (int j = 0; j < DIMENSIONS; j++) {
            printf("%.2f", clusters[i].centroid[j]);
            if (j < DIMENSIONS - 1) printf(", ");
        }
        printf(") - Points: %d\n", clusters[i].point_count);
    }
}

int main(int argc, char *argv[]) {
    int num_threads = 4; // Default
    
    if (argc > 1) {
        num_threads = atoi(argv[1]);
    }
    
    omp_set_num_threads(num_threads);
    
    printf("Running K-means with OpenMP using %d threads\n", num_threads);
    
    Point *points = malloc(NUM_POINTS * sizeof(Point));
    Cluster *clusters = malloc(NUM_CLUSTERS * sizeof(Cluster));
    
    initialize_points(points, NUM_POINTS);
    initialize_clusters(clusters, points, NUM_CLUSTERS);
    
    double start = omp_get_wtime();
    kmeans_parallel(points, clusters, NUM_POINTS, NUM_CLUSTERS);
    double end = omp_get_wtime();
    
    double time_taken = end - start;
    printf("Execution time: %.4f seconds\n", time_taken);
    
    print_results(points, clusters, NUM_POINTS, NUM_CLUSTERS);
    
    free(points);
    free(clusters);
    
    return 0;
}

/* 
Compilation: gcc -fopenmp -o kmeans_openmp kmeans_openmp.c -lm
Execution: ./kmeans_openmp [num_threads]
Example: ./kmeans_openmp 8
*/