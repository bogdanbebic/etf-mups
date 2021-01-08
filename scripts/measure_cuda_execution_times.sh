#!/bin/bash

# ----------
# Usage:
# ./measure_execution_times.sh <execution_times_file> <prog> <prog_parallel> <prog_arg1> <prog_arg2> ... <prog_argn>
#
# ----------

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

log_file="$program_parallel-parallel.log"
real_time_parallel=`{ time $program_parallel $args >$log_file; } 2>&1 | grep real | cut -f2`

echo "$program $program_parallel $args" >> $outfile
echo "$real_time $real_time_parallel" >> $outfile
