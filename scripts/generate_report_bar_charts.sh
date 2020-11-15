root_dir=$(realpath `git rev-parse --git-dir`/..)
python3_venv_dir=${root_dir}/venv

# activate python venv
source $python3_venv_dir/bin/activate

runfile=$1
prog_parallel=$2

oldIFS=$IFS
IFS=$'\n'

execution_times_file="$prog_parallel-exec-times.txt"

for line in `cat $runfile`
do
    
    prog=`echo $line | cut -d' ' -f1`
    args=`echo $line | cut -d' ' -f2-`
    prog_dir=`dirname $runfile`
    prog="$prog_dir/$prog"
    $root_dir/scripts/measure_execution_times.sh $execution_times_file $prog $prog_parallel $args
done

IFS=$oldIFS

python3 $root_dir/scripts/plot_speedups.py $execution_times_file

# deactivate python venv
deactivate
