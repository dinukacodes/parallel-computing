#!/bin/bash
# benchmark_cuda.sh
# Comprehensive CUDA K-means benchmarking script

echo "========================================="
echo "CUDA K-means Clustering Benchmark"
echo "========================================="
echo ""

# Create results directory
mkdir -p results_cuda
RESULTS_FILE="results_cuda/cuda_results.csv"
LOG_FILE="results_cuda/cuda_benchmark.log"

# Initialize CSV file
echo "ThreadsPerBlock,Run,Iterations,Time_seconds" > $RESULTS_FILE

# Block size configurations to test
BLOCK_SIZES=(64 128 256 512 768 1024)

# Number of runs per configuration (for averaging)
NUM_RUNS=5

echo "Starting CUDA benchmark..." | tee $LOG_FILE
echo "Running $NUM_RUNS trials for each configuration" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# Check if executable exists
if [ ! -f "./kmeans_cuda" ]; then
    echo "ERROR: kmeans_cuda executable not found!"
    echo "Please compile first: nvcc -o kmeans_cuda kmeans_cuda.cu -lm"
    exit 1
fi

# Get GPU information
echo "GPU Information:" | tee -a $LOG_FILE
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# Run benchmarks
for blocks in "${BLOCK_SIZES[@]}"
do
    echo "=========================================" | tee -a $LOG_FILE
    echo "Testing with $blocks threads per block" | tee -a $LOG_FILE
    echo "=========================================" | tee -a $LOG_FILE
    
    for run in $(seq 1 $NUM_RUNS)
    do
        echo "  Run $run/$NUM_RUNS..." | tee -a $LOG_FILE
        
        # Run the program and capture output
        OUTPUT=$(./kmeans_cuda $blocks 2>&1)
        
        # Check if run was successful
        if [ $? -ne 0 ]; then
            echo "    ERROR: Run failed!" | tee -a $LOG_FILE
            echo "$OUTPUT" | tee -a $LOG_FILE
            continue
        fi
        
        # Extract execution time and iterations
        TIME=$(echo "$OUTPUT" | grep "Execution time:" | awk '{print $3}')
        ITERATIONS=$(echo "$OUTPUT" | grep "K-means completed" | awk '{print $4}')
        
        # Save to CSV
        echo "$blocks,$run,$ITERATIONS,$TIME" >> $RESULTS_FILE
        
        echo "    Time: ${TIME}s, Iterations: $ITERATIONS" | tee -a $LOG_FILE
        
        # Small delay between runs
        sleep 0.5
    done
    
    # Calculate average time for this configuration
    AVG_TIME=$(awk -F',' -v b="$blocks" '$1==b {sum+=$4; count++} END {if(count>0) printf "%.4f", sum/count; else print "N/A"}' $RESULTS_FILE)
    echo "  Average time: ${AVG_TIME}s" | tee -a $LOG_FILE
    echo "" | tee -a $LOG_FILE
done

echo "=========================================" | tee -a $LOG_FILE
echo "Benchmark Complete!" | tee -a $LOG_FILE
echo "=========================================" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# Generate summary
echo "Summary of Results:" | tee -a $LOG_FILE
echo "-------------------" | tee -a $LOG_FILE
printf "%-18s %-15s %-20s\n" "Threads/Block" "Avg Time (s)" "Relative Performance" | tee -a $LOG_FILE

# Find best configuration
BEST_TIME=$(awk -F',' 'NR>1 {sum[$1]+=$4; count[$1]++} END {min=999999; for(b in sum) {avg=sum[b]/count[b]; if(avg<min) min=avg} print min}' $RESULTS_FILE)

for blocks in "${BLOCK_SIZES[@]}"
do
    AVG_TIME=$(awk -F',' -v b="$blocks" '$1==b {sum+=$4; count++} END {if(count>0) printf "%.4f", sum/count; else print "N/A"}' $RESULTS_FILE)
    
    if [ "$AVG_TIME" != "N/A" ]; then
        RELATIVE=$(echo "scale=2; $BEST_TIME / $AVG_TIME * 100" | bc)
        MARKER=""
        if [ "$AVG_TIME" == "$BEST_TIME" ]; then
            MARKER=" ⭐ BEST"
        fi
        printf "%-18s %-15s %-20s\n" "$blocks" "$AVG_TIME" "${RELATIVE}%${MARKER}" | tee -a $LOG_FILE
    else
        printf "%-18s %-15s %-20s\n" "$blocks" "FAILED" "N/A" | tee -a $LOG_FILE
    fi
done

echo "" | tee -a $LOG_FILE

# Compare with serial baseline (if available)
if [ -f "results_openmp/openmp_results.csv" ]; then
    SERIAL_TIME=$(awk -F',' '$1==1 {sum+=$4; count++} END {if(count>0) printf "%.4f", sum/count}' results_openmp/openmp_results.csv)
    if [ -n "$SERIAL_TIME" ]; then
        echo "Comparison with Serial CPU:" | tee -a $LOG_FILE
        echo "---------------------------" | tee -a $LOG_FILE
        SPEEDUP=$(echo "scale=2; $SERIAL_TIME / $BEST_TIME" | bc)
        echo "Serial CPU time: ${SERIAL_TIME}s" | tee -a $LOG_FILE
        echo "Best CUDA time: ${BEST_TIME}s" | tee -a $LOG_FILE
        echo "GPU Speedup: ${SPEEDUP}x faster than CPU" | tee -a $LOG_FILE
        echo "" | tee -a $LOG_FILE
    fi
fi

echo "Results saved to:" | tee -a $LOG_FILE
echo "  - CSV data: $RESULTS_FILE" | tee -a $LOG_FILE
echo "  - Log file: $LOG_FILE" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# Create a simple gnuplot script for visualization (optional)
cat > results_cuda/plot_cuda.gnu << 'EOF'
set terminal png size 1200,800
set output 'results_cuda/cuda_performance.png'
set multiplot layout 2,2 title "CUDA K-means Performance Analysis"

# Plot 1: Execution Time vs Block Size
set title "Execution Time vs Threads per Block"
set xlabel "Threads per Block"
set ylabel "Time (seconds)"
set grid
set key right top
plot 'results_cuda/cuda_summary.dat' using 1:2 with linespoints linewidth 2 pointtype 7 pointsize 1.5 title "Execution Time"

# Plot 2: Performance Bar Chart
set title "Performance Comparison"
set xlabel "Threads per Block"
set ylabel "Time (seconds)"
set style fill solid
set grid
set key off
plot 'results_cuda/cuda_summary.dat' using 1:2:xtic(1) with boxes linecolor rgb "#2563eb" title "Time"

# Plot 3: Relative Performance
set title "Relative Performance (100% = Best)"
set xlabel "Threads per Block"
set ylabel "Performance (%)"
set grid
set yrange [0:110]
set key right bottom
plot 'results_cuda/cuda_summary.dat' using 1:3 with linespoints linewidth 2 pointtype 7 pointsize 1.5 title "Relative Performance"

# Plot 4: Throughput
set title "Throughput (Points/Second)"
set xlabel "Threads per Block"
set ylabel "Points Processed per Second"
set grid
set key left top
set format y "%.0s%c"
plot 'results_cuda/cuda_summary.dat' using 1:(100000/$2) with linespoints linewidth 2 pointtype 7 pointsize 1.5 title "Throughput"

unset multiplot
EOF

# Generate summary data file for plotting
echo "# ThreadsPerBlock AvgTime RelativePerformance" > results_cuda/cuda_summary.dat
for blocks in "${BLOCK_SIZES[@]}"
do
    AVG_TIME=$(awk -F',' -v b="$blocks" '$1==b {sum+=$4; count++} END {if(count>0) printf "%.4f", sum/count}' $RESULTS_FILE)
    
    if [ -n "$AVG_TIME" ]; then
        RELATIVE=$(echo "scale=2; $BEST_TIME / $AVG_TIME * 100" | bc)
        echo "$blocks $AVG_TIME $RELATIVE" >> results_cuda/cuda_summary.dat
    fi
done

echo "To generate graphs (requires gnuplot):"
echo "  gnuplot results_cuda/plot_cuda.gnu"
echo ""

# Display final summary table
if [ -f results_cuda/cuda_summary.dat ]; then
    echo "========================================="
    echo "Quick Reference Table:"
    echo "========================================="
    column -t results_cuda/cuda_summary.dat
fi

# Additional GPU statistics
echo ""
echo "========================================="
echo "GPU Utilization Statistics:"
echo "========================================="
echo "Note: Run 'nvidia-smi dmon' in another terminal during benchmark for real-time monitoring"
echo ""

echo "Benchmark completed successfully!"