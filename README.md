# K-means Clustering: Parallel Implementations

This repository contains multiple parallel implementations of the K-means clustering algorithm and supporting scripts for benchmarking and results. The README has been updated to reflect the actual project layout in this workspace and to show how to include screenshots/images so they render on GitHub.

**Quick notes**:
- **Files in this repo:** compile/run the source files present in the root (see structure below).
- **To display images on GitHub:** add image files to the `screenshots/` folder and commit them; reference them with relative paths in Markdown (examples below).

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

## Screenshots & Images (show on GitHub)

Place image files (PNG, JPG, GIF) into the `screenshots/` folder and commit them. Then reference them using relative paths so GitHub renders them in the README. Examples:

```markdown
![Clustering result example](screenshots/clustering_example.png)
![Performance plot](screenshots/perf_plot.png)
```

If you don't yet have images, add a placeholder or create and commit them later — GitHub will show them automatically when pushed to the repository.

## Results & Benchmarks

Timing and plotting artifacts are stored under `results_cuda/`, `results_mpi/`, and `results_openmp/`.

Use the benchmarking scripts in `benchmarking/` to reproduce experiments. Example (PowerShell):

```powershell
cd benchmarking
./run_tests.sh
```

## Suggested Next Steps

- Add real screenshots into `screenshots/` (e.g., `clustering_example.png`, `perf_plot.png`) and commit them to enable inline display in this README.
- Run `run_tests.sh` or the scripts in `benchmarking/` to generate up-to-date results.
- If you want, I can commit these README changes and/or add example placeholder images.

---
If you'd like, I can now (1) add placeholder screenshots, (2) commit these changes, or (3) run the benchmarking scripts and attach result summaries — tell me which you prefer.
