#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

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

// Initialize points with random data (only on rank 0)
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

// Parallel K-means algorithm using MPI
void kmeans_mpi(Point *local_points, Cluster *clusters, int local_num_points, 
                int num_clusters, int rank, int size) {
    int changed, global_changed;
    int iterations = 0;
    
    do {
        changed = 0;
        
        // Broadcast current centroids to all processes
        for (int i = 0; i < num_clusters; i++) {
            MPI_Bcast(clusters[i].centroid, DIMENSIONS, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
        
        // Assignment step: Each process assigns its local points
        for (int i = 0; i < local_num_points; i++) {
            double min_distance = 1e9;
            int closest_cluster = -1;
            
            for (int j = 0; j < num_clusters; j++) {
                double dist = euclidean_distance(local_points[i].coordinates,
                                                clusters[j].centroid,
                                                DIMENSIONS);
                if (dist < min_distance) {
                    min_distance = dist;
                    closest_cluster = j;
                }
            }
            
            if (local_points[i].cluster_id != closest_cluster) {
                local_points[i].cluster_id = closest_cluster;
                changed = 1;
            }
        }
        
        // Check if any process had changes
        MPI_Allreduce(&changed, &global_changed, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        
        // Update step: Calculate local sums and counts
        double local_sums[NUM_CLUSTERS][DIMENSIONS] = {0};
        int local_counts[NUM_CLUSTERS] = {0};
        
        for (int i = 0; i < local_num_points; i++) {
            int cluster_id = local_points[i].cluster_id;
            for (int k = 0; k < DIMENSIONS; k++) {
                local_sums[cluster_id][k] += local_points[i].coordinates[k];
            }
            local_counts[cluster_id]++;
        }
        
        // Reduce sums and counts to rank 0
        double global_sums[NUM_CLUSTERS][DIMENSIONS] = {0};
        int global_counts[NUM_CLUSTERS] = {0};
        
        MPI_Reduce(local_sums, global_sums, NUM_CLUSTERS * DIMENSIONS, 
                   MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(local_counts, global_counts, NUM_CLUSTERS, 
                   MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        
        // Rank 0 updates centroids
        if (rank == 0) {
            for (int i = 0; i < num_clusters; i++) {
                if (global_counts[i] > 0) {
                    for (int k = 0; k < DIMENSIONS; k++) {
                        clusters[i].centroid[k] = global_sums[i][k] / global_counts[i];
                    }
                    clusters[i].point_count = global_counts[i];
                }
            }
        }
        
        iterations++;
        changed = global_changed;
        
    } while (changed && iterations < MAX_ITERATIONS);
    
    if (rank == 0) {
        printf("K-means completed in %d iterations\n", iterations);
    }
}

// Print results (only on rank 0)
void print_results(Cluster *clusters, int num_clusters) {
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
    int rank, size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    Point *all_points = NULL;
    Cluster *clusters = malloc(NUM_CLUSTERS * sizeof(Cluster));
    
    // Calculate points per process
    int points_per_proc = NUM_POINTS / size;
    int remainder = NUM_POINTS % size;
    int local_num_points = points_per_proc + (rank < remainder ? 1 : 0);
    
    Point *local_points = malloc(local_num_points * sizeof(Point));
    
    // Rank 0 initializes all data
    if (rank == 0) {
        printf("Running K-means with MPI using %d processes\n", size);
        all_points = malloc(NUM_POINTS * sizeof(Point));
        initialize_points(all_points, NUM_POINTS);
        initialize_clusters(clusters, all_points, NUM_CLUSTERS);
    }
    
    // Prepare for scatterv (variable counts)
    int *sendcounts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));
    
    int offset = 0;
    for (int i = 0; i < size; i++) {
        sendcounts[i] = (points_per_proc + (i < remainder ? 1 : 0)) * sizeof(Point);
        displs[i] = offset;
        offset += sendcounts[i];
    }
    
    double start_time = MPI_Wtime();
    
    // Scatter points to all processes
    MPI_Scatterv(all_points, sendcounts, displs, MPI_BYTE,
                 local_points, local_num_points * sizeof(Point), MPI_BYTE,
                 0, MPI_COMM_WORLD);
    
    // Run parallel k-means
    kmeans_mpi(local_points, clusters, local_num_points, NUM_CLUSTERS, rank, size);
    
    double end_time = MPI_Wtime();
    
    if (rank == 0) {
        double time_taken = end_time - start_time;
        printf("Execution time: %.4f seconds\n", time_taken);
        print_results(clusters, NUM_CLUSTERS);
        free(all_points);
    }
    
    free(local_points);
    free(clusters);
    free(sendcounts);
    free(displs);
    
    MPI_Finalize();
    return 0;
}

/*
Compilation: mpicc -o kmeans_mpi kmeans_mpi.c -lm
Execution: mpirun -np [num_processes] ./kmeans_mpi
Example: mpirun -np 4 ./kmeans_mpi
*/