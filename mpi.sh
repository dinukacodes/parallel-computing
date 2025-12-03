#!/bin/bash
# benchmark_mpi.sh
# Comprehensive MPI K-means benchmarking script

echo "========================================="
echo "MPI K-means Clustering Benchmark"
echo "========================================="
echo ""

# Create results directory
mkdir -p results_mpi
RESULTS_FILE="results_mpi/mpi_results.csv"
LOG_FILE="results_mpi/mpi_benchmark.log"

# Initialize CSV file
echo "Processes,Run,Iterations,Time_seconds" > $RESULTS_FILE

# Process configurations to test
PROCESS_COUNTS=(1 2 4)

# Number of runs per configuration (for averaging)
NUM_RUNS=5

echo "Starting MPI benchmark..." | tee $LOG_FILE
echo "Running $NUM_RUNS trials for each configuration" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# Check if executable exists
if [ ! -f "./kmeans_mpi" ]; then
    echo "ERROR: kmeans_mpi executable not found!"
    echo "Please compile first: mpicc -o kmeans_mpi kmeans_mpi.c -lm"
    exit 1
fi

# Detect number of available CPU cores
NUM_CORES=$(nproc)
echo "Detected $NUM_CORES CPU cores" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# Run benchmarks
for procs in "${PROCESS_COUNTS[@]}"
do
    echo "=========================================" | tee -a $LOG_FILE
    echo "Testing with $procs process(es)" | tee -a $LOG_FILE
    echo "=========================================" | tee -a $LOG_FILE
    
    # Check if we need oversubscribe
    OVERSUBSCRIBE=""
    if [ $procs -gt $NUM_CORES ]; then
        OVERSUBSCRIBE="--oversubscribe"
        echo "  Note: Using --oversubscribe (requesting $procs processes on $NUM_CORES cores)" | tee -a $LOG_FILE
    fi
    
    for run in $(seq 1 $NUM_RUNS)
    do
        echo "  Run $run/$NUM_RUNS..." | tee -a $LOG_FILE
        
        # Run the program and capture output
        OUTPUT=$(mpirun $OVERSUBSCRIBE -np $procs ./kmeans_mpi 2>&1)
        
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
        echo "$procs,$run,$ITERATIONS,$TIME" >> $RESULTS_FILE
        
        echo "    Time: ${TIME}s, Iterations: $ITERATIONS" | tee -a $LOG_FILE
        
        # Small delay between runs
        sleep 0.5
    done
    
    # Calculate average time for this configuration
    AVG_TIME=$(awk -F',' -v p="$procs" '$1==p {sum+=$4; count++} END {if(count>0) printf "%.4f", sum/count; else print "N/A"}' $RESULTS_FILE)
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
printf "%-12s %-15s %-15s %-15s\n" "Processes" "Avg Time (s)" "Speedup" "Efficiency" | tee -a $LOG_FILE

# Calculate baseline (1 process)
BASELINE=$(awk -F',' '$1==1 {sum+=$4; count++} END {if(count>0) printf "%.4f", sum/count; else print "0"}' $RESULTS_FILE)

if [ "$BASELINE" == "0" ]; then
    echo "ERROR: No baseline data available!" | tee -a $LOG_FILE
    exit 1
fi

for procs in "${PROCESS_COUNTS[@]}"
do
    AVG_TIME=$(awk -F',' -v p="$procs" '$1==p {sum+=$4; count++} END {if(count>0) printf "%.4f", sum/count; else print "N/A"}' $RESULTS_FILE)
    
    if [ "$AVG_TIME" != "N/A" ]; then
        SPEEDUP=$(echo "scale=2; $BASELINE / $AVG_TIME" | bc)
        EFFICIENCY=$(echo "scale=2; ($BASELINE / $AVG_TIME) / $procs * 100" | bc)
        printf "%-12s %-15s %-15s %-15s\n" "$procs" "$AVG_TIME" "${SPEEDUP}x" "${EFFICIENCY}%" | tee -a $LOG_FILE
    else
        printf "%-12s %-15s %-15s %-15s\n" "$procs" "FAILED" "N/A" "N/A" | tee -a $LOG_FILE
    fi
done

echo "" | tee -a $LOG_FILE
echo "Results saved to:" | tee -a $LOG_FILE
echo "  - CSV data: $RESULTS_FILE" | tee -a $LOG_FILE
echo "  - Log file: $LOG_FILE" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# Create a simple gnuplot script for visualization (optional)
cat > results_mpi/plot_mpi.gnu << 'EOF'
set terminal png size 1200,800
set output 'results_mpi/mpi_performance.png'
set multiplot layout 2,2 title "MPI K-means Performance Analysis"

# Plot 1: Execution Time
set title "Execution Time vs Processes"
set xlabel "Number of Processes"
set ylabel "Time (seconds)"
set grid
set key left top
plot 'results_mpi/mpi_summary.dat' using 1:2 with linespoints linewidth 2 pointtype 7 pointsize 1.5 title "Execution Time"

# Plot 2: Speedup
set title "Speedup vs Processes"
set xlabel "Number of Processes"
set ylabel "Speedup"
set grid
set key left top
plot 'results_mpi/mpi_summary.dat' using 1:3 with linespoints linewidth 2 pointtype 7 pointsize 1.5 title "Actual Speedup", \
     x with lines linetype 2 linewidth 2 title "Ideal Speedup"

# Plot 3: Efficiency
set title "Parallel Efficiency"
set xlabel "Number of Processes"
set ylabel "Efficiency (%)"
set grid
set key right top
plot 'results_mpi/mpi_summary.dat' using 1:4 with linespoints linewidth 2 pointtype 7 pointsize 1.5 title "Efficiency"

# Plot 4: Scalability
set title "Strong Scaling"
set xlabel "Number of Processes"
set ylabel "Normalized Performance"
set grid
set key left top
plot 'results_mpi/mpi_summary.dat' using 1:(1/$2) with linespoints linewidth 2 pointtype 7 pointsize 1.5 title "Performance (1/Time)"

unset multiplot
EOF

# Generate summary data file for plotting
echo "# Processes AvgTime Speedup Efficiency" > results_mpi/mpi_summary.dat
for procs in "${PROCESS_COUNTS[@]}"
do
    AVG_TIME=$(awk -F',' -v p="$procs" '$1==p {sum+=$4; count++} END {if(count>0) printf "%.4f", sum/count}' $RESULTS_FILE)
    
    if [ -n "$AVG_TIME" ]; then
        SPEEDUP=$(echo "scale=4; $BASELINE / $AVG_TIME" | bc)
        EFFICIENCY=$(echo "scale=2; ($BASELINE / $AVG_TIME) / $procs * 100" | bc)
        echo "$procs $AVG_TIME $SPEEDUP $EFFICIENCY" >> results_mpi/mpi_summary.dat
    fi
done

echo "To generate graphs (requires gnuplot):"
echo "  gnuplot results_mpi/plot_mpi.gnu"
echo ""

# Display final summary table
if [ -f results_mpi/mpi_summary.dat ]; then
    echo "========================================="
    echo "Quick Reference Table:"
    echo "========================================="
    column -t results_mpi/mpi_summary.dat
fi

echo ""
echo "Benchmark completed successfully!"