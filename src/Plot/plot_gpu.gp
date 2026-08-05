set datafile separator ","
set key autotitle columnhead

file = ARG1

set terminal pngcairo size 1400,900
set output file.".png"

set multiplot layout 3,1 title "GPU Statistics"

set title "GPU Utilization"
plot file using "time_sec":"utilization.gpu [%]" with lines lw 2, \
     file using "time_sec":"utilization.memory [%]" with lines lw 2

set title "Memory Usage"
plot file using "time_sec":"memory.used [MiB]" with lines lw 2

set title "Power Consumption"
plot file using "time_sec":"power.draw [W]" with lines lw 2

unset multiplot