# K-means Clustering: Parallel Implementations

## Repository Structure (actual)

```
.
├── assumptions.txt
├── cuda.sh
├── file.cu                # CUDA kernel / CUDA source
├── kmeans_cuda            # (binary or folder) CUDA build output
├── kmeans_mpi            
├── kmeans_openmp         
├── mpi.c                  # MPI implementation source
├── mpi.sh
├── omp.c                  # OpenMP implementation source
├── omp.sh
├── README.md
├── run_tests.sh
├── serial_code.c
├── benchmarking/
│   ├── cuda.sh
│   ├── mpi.sh
│   ├── omp.sh
│   └── run_tests.sh
├── parallel-implementations/
├── recodings/
├── results_cuda/
├── results_mpi/
├── results_openmp/
└── screenshots/           # Add images here (commit to show on GitHub)
```

## Build & Run (examples)

Below are simple commands you can run from PowerShell (Windows) or a Unix shell adapted as needed.

OpenMP (compile `omp.c`):

```powershell
gcc -fopenmp -O3 -o kmeans_openmp.exe omp.c -lm
./kmeans_openmp.exe 8
```

MPI (compile `mpi.c`):

```powershell
mpicc -O3 -o kmeans_mpi.exe mpi.c -lm
mpirun -np 4 .\kmeans_mpi.exe
```

CUDA (compile `file.cu`):

```powershell
# Adjust -arch according to your GPU; example below uses sm_86
nvcc -arch=sm_86 -O3 -o kmeans_cuda.exe file.cu -lm
.\kmeans_cuda.exe 256
```

Scripts such as `cuda.sh`, `mpi.sh`, and `omp.sh` are provided in the root and in `benchmarking/` to automate runs — make sure scripts have executable permissions on Unix systems.
