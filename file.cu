#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

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

// Custom atomicAdd for double (for older GPU compatibility)
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

// CUDA kernel for assignment step
__global__ void assign_clusters_kernel(Point *points, double *centroids, 
                                       int *changed, int num_points, int num_clusters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_points) {
        double min_distance = 1e9;
        int closest_cluster = -1;
        
        // Find nearest centroid
        for (int j = 0; j < num_clusters; j++) {
            double sum = 0.0;
            for (int k = 0; k < DIMENSIONS; k++) {
                double diff = points[idx].coordinates[k] - centroids[j * DIMENSIONS + k];
                sum += diff * diff;
            }
            double dist = sqrt(sum);
            
            if (dist < min_distance) {
                min_distance = dist;
                closest_cluster = j;
            }
        }
        
        // Update cluster assignment
        if (points[idx].cluster_id != closest_cluster) {
            points[idx].cluster_id = closest_cluster;
            atomicAdd(changed, 1);
        }
    }
}

// CUDA kernel for computing partial sums per block
__global__ void compute_partial_sums_kernel(Point *points, double *partial_sums,
                                            int *partial_counts, int num_points,
                                            int num_clusters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int block_id = blockIdx.x;
    
    // Shared memory for this block
    extern __shared__ char shared_memory[];
    double *block_sums = (double*)shared_memory;
    int *block_counts = (int*)&block_sums[num_clusters * DIMENSIONS];
    
    // Initialize shared memory
    for (int i = tid; i < num_clusters * DIMENSIONS; i += blockDim.x) {
        block_sums[i] = 0.0;
    }
    for (int i = tid; i < num_clusters; i += blockDim.x) {
        block_counts[i] = 0;
    }
    __syncthreads();
    
    // Each thread processes one point
    if (idx < num_points) {
        int cluster_id = points[idx].cluster_id;
        
        // Add point coordinates to cluster sum
        for (int k = 0; k < DIMENSIONS; k++) {
            atomicAdd(&block_sums[cluster_id * DIMENSIONS + k], 
                     points[idx].coordinates[k]);
        }
        atomicAdd(&block_counts[cluster_id], 1);
    }
    __syncthreads();
    
    // Write block results to global memory
    for (int i = tid; i < num_clusters * DIMENSIONS; i += blockDim.x) {
        partial_sums[block_id * num_clusters * DIMENSIONS + i] = block_sums[i];
    }
    for (int i = tid; i < num_clusters; i += blockDim.x) {
        partial_counts[block_id * num_clusters + i] = block_counts[i];
    }
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

// Check CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Parallel K-means algorithm using CUDA
void kmeans_cuda(Point *points, Cluster *clusters, int num_points, 
                 int num_clusters, int threads_per_block) {
    // Device pointers
    Point *d_points;
    double *d_centroids;
    int *d_changed;
    double *d_partial_sums;
    int *d_partial_counts;
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_points, num_points * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&d_centroids, num_clusters * DIMENSIONS * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_changed, sizeof(int)));
    
    // Calculate grid dimensions
    int blocks = (num_points + threads_per_block - 1) / threads_per_block;
    
    // Allocate memory for partial results
    CUDA_CHECK(cudaMalloc(&d_partial_sums, blocks * num_clusters * DIMENSIONS * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_partial_counts, blocks * num_clusters * sizeof(int)));
    
    // Host memory for partial results
    double *h_partial_sums = (double*)malloc(blocks * num_clusters * DIMENSIONS * sizeof(double));
    int *h_partial_counts = (int*)malloc(blocks * num_clusters * sizeof(int));
    
    // Copy initial data to device
    CUDA_CHECK(cudaMemcpy(d_points, points, num_points * sizeof(Point), cudaMemcpyHostToDevice));
    
    int iterations = 0;
    int changed = 1;
    
    // Shared memory size for reduction kernel
    size_t shared_mem_size = (num_clusters * DIMENSIONS * sizeof(double)) + 
                            (num_clusters * sizeof(int));
    
    while (changed && iterations < MAX_ITERATIONS) {
        changed = 0;
        
        // Copy centroids to device
        double *centroid_array = (double*)malloc(num_clusters * DIMENSIONS * sizeof(double));
        for (int i = 0; i < num_clusters; i++) {
            for (int j = 0; j < DIMENSIONS; j++) {
                centroid_array[i * DIMENSIONS + j] = clusters[i].centroid[j];
            }
        }
        CUDA_CHECK(cudaMemcpy(d_centroids, centroid_array, 
                   num_clusters * DIMENSIONS * sizeof(double), cudaMemcpyHostToDevice));
        free(centroid_array);
        
        // Assignment step
        CUDA_CHECK(cudaMemcpy(d_changed, &changed, sizeof(int), cudaMemcpyHostToDevice));
        assign_clusters_kernel<<<blocks, threads_per_block>>>(d_points, d_centroids, 
                                                               d_changed, num_points, num_clusters);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
        
        // Update step - compute partial sums
        compute_partial_sums_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
            d_points, d_partial_sums, d_partial_counts, num_points, num_clusters);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Copy partial results back to host
        CUDA_CHECK(cudaMemcpy(h_partial_sums, d_partial_sums, 
                   blocks * num_clusters * DIMENSIONS * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_partial_counts, d_partial_counts, 
                   blocks * num_clusters * sizeof(int), cudaMemcpyDeviceToHost));
        
        // Compute final centroids on CPU
        for (int i = 0; i < num_clusters; i++) {
            double sum[DIMENSIONS] = {0};
            int count = 0;
            
            for (int b = 0; b < blocks; b++) {
                for (int k = 0; k < DIMENSIONS; k++) {
                    sum[k] += h_partial_sums[b * num_clusters * DIMENSIONS + i * DIMENSIONS + k];
                }
                count += h_partial_counts[b * num_clusters + i];
            }
            
            if (count > 0) {
                for (int k = 0; k < DIMENSIONS; k++) {
                    clusters[i].centroid[k] = sum[k] / count;
                }
                clusters[i].point_count = count;
            }
        }
        
        iterations++;
    }
    
    printf("K-means completed in %d iterations\n", iterations);
    
    // Copy final results back
    CUDA_CHECK(cudaMemcpy(points, d_points, num_points * sizeof(Point), cudaMemcpyDeviceToHost));
    
    // Free device memory
    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_changed);
    cudaFree(d_partial_sums);
    cudaFree(d_partial_counts);
    
    // Free host memory
    free(h_partial_sums);
    free(h_partial_counts);
}

// Print results
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
    int threads_per_block = 256; // Default
    
    if (argc > 1) {
        threads_per_block = atoi(argv[1]);
    }
    
    // Print GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Running K-means with CUDA using %d threads per block\n\n", threads_per_block);
    
    Point *points = (Point*)malloc(NUM_POINTS * sizeof(Point));
    Cluster *clusters = (Cluster*)malloc(NUM_CLUSTERS * sizeof(Cluster));
    
    initialize_points(points, NUM_POINTS);
    initialize_clusters(clusters, points, NUM_CLUSTERS);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    kmeans_cuda(points, clusters, NUM_POINTS, NUM_CLUSTERS, threads_per_block);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("Execution time: %.4f seconds\n", milliseconds / 1000.0);
    
    print_results(clusters, NUM_CLUSTERS);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(points);
    free(clusters);
    
    return 0;
}

/*
Compilation: nvcc -o kmeans_cuda kmeans_cuda.cu -lm
Or with specific architecture: nvcc -arch=sm_75 -o kmeans_cuda kmeans_cuda.cu -lm
Execution: ./kmeans_cuda [threads_per_block]
Examples: 
  ./kmeans_cuda 128
  ./kmeans_cuda 256
  ./kmeans_cuda 512
*/