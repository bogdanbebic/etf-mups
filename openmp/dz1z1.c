#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

void Usage(char *prog_name);

/*
 * RUCNO
 */

int main(int argc, char *argv[])
{
    long long n, i, thread_cnt;
    double factor;
    double sum = 0.0;

    // revert back to argc != 2
    // after testing
    if (argc != 3)
        Usage(argv[0]);
    n = strtoll(argv[1], NULL, 10);
    thread_cnt = strtoll(argv[2], NULL, 10);

    if (n < 1)
        Usage(argv[0]);

    printf("Before for loop, factor = %f.\n", factor);

    double last_iteration_factor;
    omp_set_num_threads(thread_cnt);

#pragma omp parallel reduction(+: sum) private(factor, i) shared(last_iteration_factor)
{
    long long chunk = n / omp_get_num_threads();
    long long start_i = omp_get_thread_num() * chunk; 
    long long end_i = start_i + chunk;
    int is_last_chunk = 0;

    if (omp_get_thread_num() + 1 == omp_get_num_threads())
    {
        end_i += n % omp_get_num_threads();
        is_last_chunk = 1;
    }
    // for debug purposes
    // printf("id: %2d, start: %10lld, end: %11lld, chunk: %10lld\n", omp_get_thread_num(), start_i, end_i, chunk);

    for (i = start_i; i < end_i; i++)
    {
        factor = (i % 2 == 0) ? 1.0 : -1.0;
        sum += factor / (2 * i + 1);
    }

    // substitute for lastprivate(factor)
    if (is_last_chunk) last_iteration_factor = factor;
} // parallel

    factor = last_iteration_factor;
    printf("After for loop, factor = %f.\n", factor);

    sum = 4.0 * sum;
    printf("With n = %lld terms\n", n);
    printf("   Our estimate of pi = %.14f\n", sum);
    printf("   Ref estimate of pi = %.14f\n", 4.0 * atan(1.0));
    return 0;
}

void Usage(char *prog_name)
{
    fprintf(stderr, "usage: %s <thread_count> <n>\n", prog_name);
    fprintf(stderr, "   n is the number of terms and should be >= 1\n");
    exit(0);
}
