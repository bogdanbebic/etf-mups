#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void Usage(char *prog_name);

int main(int argc, char *argv[])
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
    return 0;
}

void Usage(char *prog_name)
{
    fprintf(stderr, "usage: %s <n>\n", prog_name);
    fprintf(stderr, "   n is the number of terms and should be >= 1\n");
    exit(0);
}
