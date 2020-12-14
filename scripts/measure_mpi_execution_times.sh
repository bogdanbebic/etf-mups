#!/bin/bash

# ----------
# Usage:
# ./measure_execution_times.sh <execution_times_file> <prog> <prog_parallel> <prog_arg1> <prog_arg2> ... <prog_argn>
#
# Runs <prog> with different MPI process counts
# and times their execution. Output of <prog>
# is saved to log files with format <prog>-proc-count-<n>.log
# where <n> is the MPI process count. Generates a bar chart
# for every program with execution time with respect to thread count
# The bar chart is generated by `generate_bar_chart.py` and is saved
# in format <prog>-thread-exec-times-bar.png
# ----------

oldIFS=$IFS
IFS=", "
num_threads_list="1,2,4"

outfile=$1
# Shift all the parameters down by one
shift
program=$1
# Shift all the parameters down by one
shift
program_parallel=$1
# Shift all the parameters down by one
shift
args=$@

log_file="$program.log"
real_time=`{ time $program $args >$log_file; } 2>&1 | grep real | cut -f2`

real_times_parallel=
for num_threads in $num_threads_list;
do
    log_file="$program_parallel-proc-count-$num_threads.log"
    real_time_parallel=`{ time mpirun -n $num_threads $program_parallel $args >$log_file; } 2>&1 | grep real | cut -f2`
    real_times_parallel="$real_times_parallel$real_time_parallel,"
done 

echo "$program $program_parallel $args" >> $outfile
echo "$real_time $real_times_parallel $num_threads_list" >> $outfile

IFS=$oldIFS