#define LIMIT -999
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#include <omp.h>

#define ACCURACY 0.01

#define N 100000
int i_seq = 0;
int i_par = 0;
int traceback_seq[N];
int traceback_par[N];

void usage(int argc, char **argv);
int maximum(int a, int b, int c)
{
    int k;
    if (a <= b)
        k = b;
    else
        k = a;
    if (k <= c)
        return (c);
    else
        return (k);
}

int blosum62[24][24] = {
    {4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0, -2, -1, 0, -4},
    {-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3, -1, 0, -1, -4},
    {-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3, 3, 0, -1, -4},
    {-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3, 4, 1, -1, -4},
    {0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
    {-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2, 0, 3, -1, -4},
    {-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1, -4},
    {0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3, -1, -2, -1, -4},
    {-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3, 0, 0, -1, -4},
    {-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3, -3, -3, -1, -4},
    {-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1, -4, -3, -1, -4},
    {-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2, 0, 1, -1, -4},
    {-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1, -3, -1, -1, -4},
    {-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1, -3, -3, -1, -4},
    {-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2, -2, -1, -2, -4},
    {1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2, 0, 0, 0, -4},
    {0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0, -1, -1, 0, -4},
    {-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3, -4, -3, -2, -4},
    {-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1, -3, -2, -1, -4},
    {0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4, -3, -2, -1, -4},
    {-2, -1, 3, 4, -3, 0, 1, -1, 0, -3, -4, 0, -3, -3, -2, 0, -1, -4, -3, -3, 4, 1, -1, -4},
    {-1, 0, 0, 1, -3, 3, 4, -2, 0, -3, -3, 1, -1, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1, -4},
    {0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, 0, 0, -2, -1, -1, -1, -1, -1, -4},
    {-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, 1}};

double gettime()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}

int * sequential_solution(int argc, char *argv[])
{
    int max_rows, max_cols, penalty, idx, index;
    int *input_itemsets, *output_itemsets, *referrence;
    int *matrix_cuda, *matrix_cuda_out, *referrence_cuda;
    int size;
    int omp_num_threads;

    if (argc == 3)
    {
        max_cols = max_rows = atoi(argv[1]);
        penalty = atoi(argv[2]);
    }
    else
    {
        usage(argc, argv);
    }

    max_rows = max_rows + 1;
    max_cols = max_cols + 1;
    referrence = (int *)malloc(max_rows * max_cols * sizeof(int));
    input_itemsets = (int *)malloc(max_rows * max_cols * sizeof(int));
    output_itemsets = (int *)malloc(max_rows * max_cols * sizeof(int));

    if (!input_itemsets)
        fprintf(stderr, "error: can not allocate memory");

    srand(time(NULL));

    for (int i = 0; i < max_cols; i++)
    {
        for (int j = 0; j < max_rows; j++)
        {
            input_itemsets[i * max_cols + j] = 0;
        }
    }

    printf("Start Needleman-Wunsch\n");

    for (int i = 1; i < max_rows; i++)
    {
        input_itemsets[i * max_cols] = rand() % 10 + 1;
    }
    for (int j = 1; j < max_cols; j++)
    {
        input_itemsets[j] = rand() % 10 + 1;
    }

    for (int i = 1; i < max_cols; i++)
    {
        for (int j = 1; j < max_rows; j++)
        {
            referrence[i * max_cols + j] = blosum62[input_itemsets[i * max_cols]][input_itemsets[j]];
        }
    }

    for (int i = 1; i < max_rows; i++)
        input_itemsets[i * max_cols] = -i * penalty;
    for (int j = 1; j < max_cols; j++)
        input_itemsets[j] = -j * penalty;

    printf("Processing top-left matrix\n");
    for (int i = 0; i < max_cols - 2; i++)
    {
        for (idx = 0; idx <= i; idx++)
        {
            index = (idx + 1) * max_cols + (i + 1 - idx);
            input_itemsets[index] = maximum(input_itemsets[index - 1 - max_cols] + referrence[index],
                                            input_itemsets[index - 1] - penalty,
                                            input_itemsets[index - max_cols] - penalty);
        }
    }
    printf("Processing bottom-right matrix\n");
    for (int i = max_cols - 4; i >= 0; i--)
    {
        for (idx = 0; idx <= i; idx++)
        {
            index = (max_cols - idx - 2) * max_cols + idx + max_cols - i - 2;
            input_itemsets[index] = maximum(input_itemsets[index - 1 - max_cols] + referrence[index],
                                            input_itemsets[index - 1] - penalty,
                                            input_itemsets[index - max_cols] - penalty);
        }
    }

#define TRACEBACK
#ifdef TRACEBACK

    FILE *fpo = fopen("result_seq.txt", "w");
    fprintf(fpo, "print traceback value:\n");

    for (int i = max_rows - 2, j = max_rows - 2; i >= 0, j >= 0;)
    {
        int nw, n, w, traceback;
        if (i == max_rows - 2 && j == max_rows - 2)
            fprintf(fpo, "%d ", input_itemsets[i * max_cols + j]);
        if (i == 0 && j == 0)
            break;
        if (i > 0 && j > 0)
        {
            nw = input_itemsets[(i - 1) * max_cols + j - 1];
            w = input_itemsets[i * max_cols + j - 1];
            n = input_itemsets[(i - 1) * max_cols + j];
        }
        else if (i == 0)
        {
            nw = n = LIMIT;
            w = input_itemsets[i * max_cols + j - 1];
        }
        else if (j == 0)
        {
            nw = w = LIMIT;
            n = input_itemsets[(i - 1) * max_cols + j];
        }
        else
        {
        }

        //traceback = maximum(nw, w, n);
        int new_nw, new_w, new_n;
        new_nw = nw + referrence[i * max_cols + j];
        new_w = w - penalty;
        new_n = n - penalty;

        traceback = maximum(new_nw, new_w, new_n);
        if (traceback == new_nw)
            traceback = nw;
        if (traceback == new_w)
            traceback = w;
        if (traceback == new_n)
            traceback = n;

        fprintf(fpo, "%d ", traceback);
        traceback_seq[i_seq++] = traceback;

        if (traceback == nw)
        {
            i--;
            j--;
            continue;
        }

        else if (traceback == w)
        {
            j--;
            continue;
        }

        else if (traceback == n)
        {
            i--;
            continue;
        }

        else
            ;
    }

    fclose(fpo);

#endif

    free(referrence);
    // free(input_itemsets);
    free(output_itemsets);
    
    return input_itemsets;
}

int * parallel_solution(int argc, char *argv[])
{
    int max_rows, max_cols, penalty, idx, index;
    int *input_itemsets, *output_itemsets, *referrence;
    int *matrix_cuda, *matrix_cuda_out, *referrence_cuda;
    int size;
    int omp_num_threads;

    if (argc == 3)
    {
        max_cols = max_rows = atoi(argv[1]);
        penalty = atoi(argv[2]);
    }
    else
    {
        usage(argc, argv);
    }

    max_rows = max_rows + 1;
    max_cols = max_cols + 1;
    referrence = (int *)malloc(max_rows * max_cols * sizeof(int));
    input_itemsets = (int *)malloc(max_rows * max_cols * sizeof(int));
    output_itemsets = (int *)malloc(max_rows * max_cols * sizeof(int));

    if (!input_itemsets)
        fprintf(stderr, "error: can not allocate memory");

    srand(time(NULL));

#pragma omp parallel for collapse(2)
    for (int i = 0; i < max_cols; i++)
    {
        for (int j = 0; j < max_rows; j++)
        {
            input_itemsets[i * max_cols + j] = 0;
        }
    }

    printf("Start Needleman-Wunsch\n");

#pragma omp parallel
    {
#pragma omp for nowait
        for (int i = 1; i < max_rows; i++)
        {
            input_itemsets[i * max_cols] = rand() % 10 + 1;
        }
#pragma omp for
        for (int j = 1; j < max_cols; j++)
        {
            input_itemsets[j] = rand() % 10 + 1;
        }
    }
#pragma omp parallel for collapse(2)
    for (int i = 1; i < max_cols; i++)
    {
        for (int j = 1; j < max_rows; j++)
        {
            referrence[i * max_cols + j] = blosum62[input_itemsets[i * max_cols]][input_itemsets[j]];
        }
    }

#pragma omp parallel
    {
#pragma omp for nowait
        for (int i = 1; i < max_rows; i++)
            input_itemsets[i * max_cols] = -i * penalty;
#pragma omp for
        for (int j = 1; j < max_cols; j++)
            input_itemsets[j] = -j * penalty;
    }
#pragma omp parallel
    {
#pragma omp single
        {
            printf("Processing top-left matrix\n");

#pragma omp task private(idx, index)
            {
                for (int i = 0; i < max_cols - 2; i++)
                {
                    for (idx = 0; idx <= i; idx++)
                    {
                        index = (idx + 1) * max_cols + (i + 1 - idx);
                        input_itemsets[index] = maximum(input_itemsets[index - 1 - max_cols] + referrence[index],
                                                        input_itemsets[index - 1] - penalty,
                                                        input_itemsets[index - max_cols] - penalty);
                    }
                }
            } // task
            printf("Processing bottom-right matrix\n");

#pragma omp task private(idx, index)
            {
                for (int i = max_cols - 4; i >= 0; i--)
                {
                    for (idx = 0; idx <= i; idx++)
                    {
                        index = (max_cols - idx - 2) * max_cols + idx + max_cols - i - 2;
                        input_itemsets[index] = maximum(input_itemsets[index - 1 - max_cols] + referrence[index],
                                                        input_itemsets[index - 1] - penalty,
                                                        input_itemsets[index - max_cols] - penalty);
                    }
                }
            } // task
        }     // single
    }         // parallel
#define TRACEBACK
#ifdef TRACEBACK

    FILE *fpo = fopen("result_par.txt", "w");
    fprintf(fpo, "print traceback value:\n");

    for (int i = max_rows - 2, j = max_rows - 2; i >= 0, j >= 0;)
    {
        int nw, n, w, traceback;
        if (i == max_rows - 2 && j == max_rows - 2)
            fprintf(fpo, "%d ", input_itemsets[i * max_cols + j]);
        if (i == 0 && j == 0)
            break;
        if (i > 0 && j > 0)
        {
            nw = input_itemsets[(i - 1) * max_cols + j - 1];
            w = input_itemsets[i * max_cols + j - 1];
            n = input_itemsets[(i - 1) * max_cols + j];
        }
        else if (i == 0)
        {
            nw = n = LIMIT;
            w = input_itemsets[i * max_cols + j - 1];
        }
        else if (j == 0)
        {
            nw = w = LIMIT;
            n = input_itemsets[(i - 1) * max_cols + j];
        }
        else
        {
        }

        //traceback = maximum(nw, w, n);
        int new_nw, new_w, new_n;
        new_nw = nw + referrence[i * max_cols + j];
        new_w = w - penalty;
        new_n = n - penalty;

        traceback = maximum(new_nw, new_w, new_n);
        if (traceback == new_nw)
            traceback = nw;
        if (traceback == new_w)
            traceback = w;
        if (traceback == new_n)
            traceback = n;

        fprintf(fpo, "%d ", traceback);
        traceback_par[i_par++] = traceback;

        if (traceback == nw)
        {
            i--;
            j--;
            continue;
        }

        else if (traceback == w)
        {
            j--;
            continue;
        }

        else if (traceback == n)
        {
            i--;
            continue;
        }

        else
            ;
    }

    fclose(fpo);

#endif

    free(referrence);
    // free(input_itemsets);
    free(output_itemsets);

    return input_itemsets;
}

int compare_arr(int * arr1, int * arr2, int n)
{
    for (int i = 0; i < n; i++)
        if (arr1[i] != arr2[i])
            return 0;

    return 1;
}

int main(int argc, char **argv)
{
    printf("---------------------Sequential execution---------------------\n");
    double start_time_seq = omp_get_wtime();
    int * ret_seq = sequential_solution(argc, argv);
    double end_time_seq = omp_get_wtime();

    printf("----------------------Parallel execution----------------------\n");
    double start_time_parallel = omp_get_wtime();
    int * ret_par = parallel_solution(argc, argv);
    double end_time_parallel = omp_get_wtime();

    printf("\nSequential elapsed time: %lfs\n", end_time_seq - start_time_seq);
    printf("Parallel elapsed time: %lfs\n", end_time_parallel - start_time_parallel);
    
    int max_rows, max_cols;
    if (argc == 3)
        max_cols = max_rows = atoi(argv[1]);

    max_rows++;
    max_cols++;

    if (compare_arr(traceback_seq, traceback_par, N))
        printf("Test PASSED\n");
    else
        printf("Test FAILED\n");

    free(ret_seq);
    free(ret_par);
    return EXIT_SUCCESS;
}

void usage(int argc, char **argv)
{
    fprintf(stderr, "Usage: %s <max_rows/max_cols> <penalty>\n", argv[0]);
    fprintf(stderr, "\t<dimension>      - x and y dimensions\n");
    fprintf(stderr, "\t<penalty>        - penalty(positive integer)\n");
    exit(1);
}
