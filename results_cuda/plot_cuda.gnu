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
