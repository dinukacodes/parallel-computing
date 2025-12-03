#!/bin/bash

# OpenMP
for t in 1 2 4 8 16; do
    echo "Testing with $t threads"
    ./kmeans_openmp $t
done

# MPI
for p in 1 2 4 8 16; do
    echo "Testing with $p processes"
    mpirun -np $p ./kmeans_mpi
done

# CUDA
for b in 128 256 512 1024; do
    echo "Testing with $b threads per block"
    ./kmeans_cuda $b
done
