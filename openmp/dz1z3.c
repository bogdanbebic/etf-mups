#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

void Usage(char *prog_name);

#define ACCURACY 0.01

/*
 * tasks
 */

double sequential_solution(int argc, char *argv[])
{
    long long n, i;
    double factor;
    double sum = 0.0;

    if (argc != 2)
        Usage(argv[0]);
    n = strtoll(argv[1], NULL, 10);
    if (n < 1)
        Usage(argv[0]);

    printf("Before for loop, factor = %f.\n", factor);
    for (i = 0; i < n; i++)
    {
        factor = (i % 2 == 0) ? 1.0 : -1.0;
        sum += factor / (2 * i + 1);
    }
    printf("After for loop, factor = %f.\n", factor);

    sum = 4.0 * sum;
    printf("With n = %lld terms\n", n);
    printf("   Our estimate of pi = %.14f\n", sum);
    printf("   Ref estimate of pi = %.14f\n", 4.0 * atan(1.0));
    return sum;
}

double parallel_solution(int argc, char *argv[])
{
    long long n, i;
    double factor;
    double sum = 0.0;

    if (argc != 2)
        Usage(argv[0]);
    n = strtoll(argv[1], NULL, 10);
    if (n < 1)
        Usage(argv[0]);

    printf("Before for loop, factor = %f.\n", factor);
    int it;
    int num_tasks = 8;
    double sums[num_tasks];
    double last_factor;
#pragma omp parallel shared(it, num_tasks, sums)
{
#pragma omp single
{
    long long int start_i, end_i;
    for (it = 0; it < num_tasks; it++)
    {
        // init start_i, end_i
        start_i = it * n / num_tasks;
        end_i = (it + 1) * n / num_tasks;
#pragma omp task firstprivate(it, start_i, end_i) private(i, factor) shared(sums) shared(last_factor)
{
    for (i = start_i; i < end_i; i++)
    {
        factor = (i % 2 == 0) ? 1.0 : -1.0;
        sums[it] += factor / (2 * i + 1);
    }
    if (it == num_tasks - 1)
        last_factor = factor;
} // task
    }
} // single
} // parallel
    factor = last_factor;
    for (it = 0; it < num_tasks; it++)
    {
        sum += sums[it];
    }
    printf("After for loop, factor = %f.\n", factor);

    sum = 4.0 * sum;
    printf("With n = %lld terms\n", n);
    printf("   Our estimate of pi = %.14f\n", sum);
    printf("   Ref estimate of pi = %.14f\n", 4.0 * atan(1.0));
    return sum;
}

int main(int argc, char *argv[])
{
    printf("---------------------Sequential execution---------------------\n");
    double start_time_seq = omp_get_wtime();
    double sum_seq = sequential_solution(argc, argv);
    double end_time_seq = omp_get_wtime();

    printf("----------------------Parallel execution----------------------\n");
    double start_time_parallel = omp_get_wtime();
    double sum_parallel = parallel_solution(argc, argv);
    double end_time_parallel = omp_get_wtime();

    printf("\nSequential elapsed time: %lfs\n", end_time_seq - start_time_seq);
    printf("Parallel elapsed time: %lfs\n", end_time_parallel - start_time_parallel);

    if (fabs(sum_seq - sum_parallel) < ACCURACY)
        printf("Test PASSED\n");
    else
        printf("Test FAILED\n");

    return 0;
}

void Usage(char *prog_name)
{
    fprintf(stderr, "usage: %s <thread_count> <n>\n", prog_name);
    fprintf(stderr, "   n is the number of terms and should be >= 1\n");
    exit(0);
}
