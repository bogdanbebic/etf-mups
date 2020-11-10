#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

void Usage(char *prog_name);

/*
 * tasks
 */

int main(int argc, char *argv[])
{
    long long n, i, j;
    double factor;
    double sum = 0.0;

    if (argc != 2)
        Usage(argv[0]);
    n = strtoll(argv[1], NULL, 10);
    if (n < 1)
        Usage(argv[0]);

    printf("Before for loop, factor = %f.\n", factor);
#pragma omp parallel shared(sum)
{
#pragma omp single
{
    for (j = 0; j < 8; j++)
    {
#pragma omp task firstprivate(i) private(factor)
{
        for (i = j; i < n; i+=8)
        {
            factor = (i % 2 == 0) ? 1.0 : -1.0;
#pragma omp atomic
            sum += factor / (2 * i + 1);
        }
} // task
    }
} // single
} // parallel
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
