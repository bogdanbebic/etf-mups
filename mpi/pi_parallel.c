#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

enum Constants
{
    MANAGER = 0,
};

void Usage(char *prog_name);

int main(int argc, char *argv[])
{
    long long n, i;
    double factor;
    double sum = 0.0;

    int rank, size;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == MANAGER)
    {
        if (argc != 2)
            Usage(argv[0]);
        n = strtoll(argv[1], NULL, 10);
    }

    MPI_Bcast(&n, 1, MPI_LONG_LONG_INT, MANAGER, MPI_COMM_WORLD);

    if (n < 1)
        Usage(argv[0]);

    if (rank == MANAGER)
        printf("Before for loop, factor = %f.\n", factor);

    long long int chunk = n / size;
    long long int i_start = rank * chunk;
    long long int i_end = i_start + chunk;
    if (rank + 1 == size)
        i_end += n % size;

    for (i = i_start; i < i_end; i++)
    {
        factor = (i % 2 == 0) ? 1.0 : -1.0;
        sum += factor / (2 * i + 1);
    }

    MPI_Bcast(&factor, 1, MPI_DOUBLE, size - 1, MPI_COMM_WORLD);

    double sum_reduced = 0.0;
    MPI_Reduce(&sum, &sum_reduced, 1, MPI_DOUBLE, MPI_SUM, MANAGER, MPI_COMM_WORLD);

    if (rank == MANAGER)
    {
        printf("After for loop, factor = %f.\n", factor);
        sum_reduced = 4.0 * sum_reduced;
        printf("With n = %lld terms\n", n);
        printf("   Our estimate of pi = %.14f\n", sum_reduced);
        printf("   Ref estimate of pi = %.14f\n", 4.0 * atan(1.0));
    }

    MPI_Finalize();

    return 0;
}

void Usage(char *prog_name)
{
    fprintf(stderr, "usage: %s <n>\n", prog_name);
    fprintf(stderr, "   n is the number of terms and should be >= 1\n");
    exit(0);
}
