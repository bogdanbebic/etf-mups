#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

void Usage(char *prog_name);

#define ACCURACY 0.01

enum Constants
{
    MANAGER = 0,
};
int rank, size;

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

    return sum_reduced;
}


int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double sum_seq, start_time_seq, end_time_seq;

    if (rank == MANAGER) 
    {
        printf("---------------------Sequential execution---------------------\n");
        start_time_seq = MPI_Wtime();
        sum_seq = sequential_solution(argc, argv);
        end_time_seq = MPI_Wtime();
    }

    double start_time_parallel, end_time_parallel, sum_parallel;

    if (rank == MANAGER) 
    {
        printf("----------------------Parallel execution----------------------\n");
        start_time_parallel = MPI_Wtime();
    }
    if (rank == MANAGER)
        sum_parallel = parallel_solution(argc, argv);
    else
        parallel_solution(argc, argv);

    if (rank == MANAGER)
    {
        end_time_parallel = MPI_Wtime();

        printf("\nSequential elapsed time: %lfs\n", end_time_seq - start_time_seq);
        printf("Parallel elapsed time: %lfs\n", end_time_parallel - start_time_parallel);
        if (fabs(sum_seq - sum_parallel) < ACCURACY)
            printf("Test PASSED\n");
        else
            printf("Test FAILED\n");
    }

    MPI_Finalize();

    return 0;
}

void Usage(char *prog_name)
{
    fprintf(stderr, "usage: %s <thread_count> <n>\n", prog_name);
    fprintf(stderr, "   n is the number of terms and should be >= 1\n");
    exit(0);
}
