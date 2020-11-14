import matplotlib.pyplot as plt
import sys


def execution_times_from_str_list(execution_times_str_list):
    execution_times = []
    for execution_times_str in execution_times_str_list:
        minutes_seconds_pair = execution_times_str.split('m')
        minutes = float(minutes_seconds_pair[0])
        seconds = float(minutes_seconds_pair[1].strip('s'))
        execution_times.append(60 * minutes + seconds)
    return execution_times


def main():
    '''Expected cmd args:
    execution times: list of arguments in format provided by the `time` utility
    thread count: list of argumets
    program name: name of the program for which we are generating the bar chart
    '''
    execution_times_str = sys.argv[1:len(sys.argv) // 2]

    thread_cnt = sys.argv[len(sys.argv) // 2:-1]

    program_name = sys.argv[-1]

    execution_times = execution_times_from_str_list(execution_times_str)

    plt.figure(1)
    plt.bar(thread_cnt, execution_times)
    plt.title(program_name)
    plt.xlabel('number of threads')
    plt.ylabel('execution time [s]')
    plt.grid()
    plt.savefig(program_name + '-thread-exec-times-bar.png')
    pass


if __name__ == "__main__":
    main()
