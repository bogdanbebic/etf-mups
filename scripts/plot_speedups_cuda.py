import matplotlib.pyplot as plt
import argparse


def parse_exec_time(exec_time_str):
    minutes_seconds_pair = exec_time_str.split('m')
    minutes = float(minutes_seconds_pair[0])
    seconds = float(minutes_seconds_pair[1].strip('s'))
    return 60 * minutes + seconds


def get_execution_times(execution_times_str):
    exec_time_str, exec_time_parallel_str = execution_times_str.split()
    return {
        'exec_time' : parse_exec_time(exec_time_str),
        'exec_time_parallel' : parse_exec_time(exec_time_parallel_str)
    }


def main():
    parser = argparse.ArgumentParser(description='Plot speedup times from file')
    parser.add_argument('input_file_times', help='File which contains execution times to be plotted')
    arguments = parser.parse_args()
    with open(arguments.input_file_times) as in_file:
        times = in_file.read().split('\n')

    program_names_args = [times[i] for i in range(0, len(times) - 1, 2)]
    execution_times_str = [times[i] for i in range(1, len(times), 2)]

    width = 0.15
    width_all = width * (len(execution_times_str) - 1)
    plt.figure(1)
    for i, (prog_names_args_elem, execution_times_str_elem) in enumerate(zip(program_names_args, execution_times_str)):
        prog, prog_parallel, *args = prog_names_args_elem.split()
        dct = get_execution_times(execution_times_str_elem)
        speedup = dct['exec_time'] / dct['exec_time_parallel']
        plt.bar(- width_all / 2 + i * width, speedup, width, label=f'args={args}')
        plt.xticks([], [])

    plt.title(f'Speedup from "{prog}" to "{prog_parallel}"')
    plt.ylabel('speedup')
    plt.legend()
    plt.grid()
    plt.savefig(f'{prog_parallel}-speedups.png')


if __name__ == "__main__":
    main()
