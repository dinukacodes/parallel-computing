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
