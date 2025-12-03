#!/bin/bash
# benchmark_openmp.sh
# Comprehensive OpenMP K-means benchmarking script

echo "========================================="
echo "OpenMP K-means Clustering Benchmark"
echo "========================================="
echo ""

# Create results directory
mkdir -p results_openmp
RESULTS_FILE="results_openmp/openmp_results.csv"
LOG_FILE="results_openmp/openmp_benchmark.log"

# Initialize CSV file
echo "Threads,Run,Iterations,Time_seconds" > $RESULTS_FILE

# Thread configurations to test
THREAD_COUNTS=(1 2 4 8 16)

# Number of runs per configuration (for averaging)
NUM_RUNS=5

echo "Starting OpenMP benchmark..." | tee $LOG_FILE
echo "Running $NUM_RUNS trials for each configuration" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# Check if executable exists
if [ ! -f "./kmeans_openmp" ]; then
    echo "ERROR: kmeans_openmp executable not found!"
    echo "Please compile first: gcc -fopenmp -o kmeans_openmp kmeans_openmp.c -lm"
    exit 1
fi

# Run benchmarks
for threads in "${THREAD_COUNTS[@]}"
do
    echo "=========================================" | tee -a $LOG_FILE
    echo "Testing with $threads thread(s)" | tee -a $LOG_FILE
    echo "=========================================" | tee -a $LOG_FILE
    
    for run in $(seq 1 $NUM_RUNS)
    do
        echo "  Run $run/$NUM_RUNS..." | tee -a $LOG_FILE
        
        # Run the program and capture output
        OUTPUT=$(./kmeans_openmp $threads 2>&1)
        
        # Extract execution time and iterations
        TIME=$(echo "$OUTPUT" | grep "Execution time:" | awk '{print $3}')
        ITERATIONS=$(echo "$OUTPUT" | grep "K-means completed" | awk '{print $4}')
        
        # Save to CSV
        echo "$threads,$run,$ITERATIONS,$TIME" >> $RESULTS_FILE
        
        echo "    Time: ${TIME}s, Iterations: $ITERATIONS" | tee -a $LOG_FILE
        
        # Small delay between runs
        sleep 0.5
    done
    
    # Calculate average time for this configuration
    AVG_TIME=$(awk -F',' -v t="$threads" '$1==t {sum+=$4; count++} END {printf "%.4f", sum/count}' $RESULTS_FILE)
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
printf "%-10s %-15s %-15s\n" "Threads" "Avg Time (s)" "Speedup" | tee -a $LOG_FILE

# Calculate baseline (1 thread)
BASELINE=$(awk -F',' '$1==1 {sum+=$4; count++} END {printf "%.4f", sum/count}' $RESULTS_FILE)

for threads in "${THREAD_COUNTS[@]}"
do
    AVG_TIME=$(awk -F',' -v t="$threads" '$1==t {sum+=$4; count++} END {printf "%.4f", sum/count}' $RESULTS_FILE)
    SPEEDUP=$(echo "scale=2; $BASELINE / $AVG_TIME" | bc)
    printf "%-10s %-15s %-15s\n" "$threads" "$AVG_TIME" "${SPEEDUP}x" | tee -a $LOG_FILE
done

echo "" | tee -a $LOG_FILE
echo "Results saved to:" | tee -a $LOG_FILE
echo "  - CSV data: $RESULTS_FILE" | tee -a $LOG_FILE
echo "  - Log file: $LOG_FILE" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# Create a simple gnuplot script for visualization (optional)
cat > results_openmp/plot_openmp.gnu << 'EOF'
set terminal png size 1200,800
set output 'results_openmp/openmp_performance.png'
set multiplot layout 2,2 title "OpenMP K-means Performance Analysis"

# Plot 1: Execution Time
set title "Execution Time vs Threads"
set xlabel "Number of Threads"
set ylabel "Time (seconds)"
set grid
set key left top
plot 'results_openmp/openmp_summary.dat' using 1:2 with linespoints linewidth 2 pointtype 7 pointsize 1.5 title "Execution Time"

# Plot 2: Speedup
set title "Speedup vs Threads"
set xlabel "Number of Threads"
set ylabel "Speedup"
set grid
set key left top
plot 'results_openmp/openmp_summary.dat' using 1:3 with linespoints linewidth 2 pointtype 7 pointsize 1.5 title "Actual Speedup", \
     x with lines linetype 2 linewidth 2 title "Ideal Speedup"

# Plot 3: Efficiency
set title "Parallel Efficiency"
set xlabel "Number of Threads"
set ylabel "Efficiency (%)"
set grid
set key right top
plot 'results_openmp/openmp_summary.dat' using 1:($3/$1*100) with linespoints linewidth 2 pointtype 7 pointsize 1.5 title "Efficiency"

unset multiplot
EOF

# Generate summary data file for plotting
echo "# Threads AvgTime Speedup" > results_openmp/openmp_summary.dat
for threads in "${THREAD_COUNTS[@]}"
do
    AVG_TIME=$(awk -F',' -v t="$threads" '$1==t {sum+=$4; count++} END {printf "%.4f", sum/count}' $RESULTS_FILE)
    SPEEDUP=$(echo "scale=4; $BASELINE / $AVG_TIME" | bc)
    echo "$threads $AVG_TIME $SPEEDUP" >> results_openmp/openmp_summary.dat
done

echo "To generate graphs (requires gnuplot):"
echo "  gnuplot results_openmp/plot_openmp.gnu"
echo ""

# Display final summary table
echo "========================================="
echo "Quick Reference Table:"
echo "========================================="
column -t results_openmp/openmp_summary.dat

echo ""
echo "Benchmark completed successfully!"