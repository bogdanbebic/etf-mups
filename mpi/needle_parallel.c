#include <mpi.h>
// #define DEBUG
#ifdef DEBUG
#define LOG(fmt, ...)                                      \
    do                                                     \
    {                                                      \
        fprintf(stderr, "%s:%d:%s(): " fmt "\n", __FILE__, \
                __LINE__, __func__, __VA_ARGS__);          \
    } while (0)
#else
#define LOG(fmt, ...)
#endif

#define LIMIT -999
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

enum Tags
{
    TAG_RESULT = 1000,
    TAG_SIZE,
};

int send_buffer_size;
int send_buffer[45000];
int proc_size, proc_rank;

MPI_Datatype diagonal1, diagonal2;

void runTest(int argc, char **argv);
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

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &proc_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

    runTest(argc, argv);

    MPI_Finalize();

    return EXIT_SUCCESS;
}

void usage(int argc, char **argv)
{
    fprintf(stderr, "Usage: %s <max_rows/max_cols> <penalty>\n", argv[0]);
    fprintf(stderr, "\t<dimension>      - x and y dimensions\n");
    fprintf(stderr, "\t<penalty>        - penalty(positive integer)\n");
    exit(1);
}

void runTest(int argc, char **argv)
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

    // srand(time(NULL));
    srand(0);

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
        /*
        if (proc_rank == 0)
        {
            MPI_Status status;
            // TODO: something
            for (int proc_idx = 1; proc_idx < proc_size; proc_idx++)
                MPI_Send(send_buffer, count, MPI_INT, proc_idx, TAG_DATA, MPI_COMM_WORLD);

            for (int proc_idx = 1; proc_idx < proc_size; proc_idx++)
            {
                MPI_Recv(send_buffer, count, MPI_INT, 0, TAG_RESULT, MPI_COMM_WORLD, &status);
                // TODO: status -> index, save data
            }
        }
        else
        {
            MPI_Recv(send_buffer, count, MPI_INT, 0, TAG_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // TODO: something
            MPI_Send(send_buffer, count, MPI_INT, 0, TAG_RESULT, MPI_COMM_WORLD);
        }*/

        // MPI_Bcast(input_itemsets, max_rows * max_cols, MPI_INT, 0, MPI_COMM_WORLD);

        MPI_Type_vector(i + 1, 1, max_cols - 1, MPI_INT, &diagonal1);
        MPI_Type_commit(&diagonal1);
        MPI_Type_vector(i + 2, 1, max_cols - 1, MPI_INT, &diagonal2);
        MPI_Type_commit(&diagonal2);
        MPI_Bcast(input_itemsets + i, 1, diagonal1, 0, MPI_COMM_WORLD);
        MPI_Bcast(input_itemsets + i + 1, 1, diagonal2, 0, MPI_COMM_WORLD);


        int chunk = (i + 1) / proc_size;
        int idx_start = chunk * proc_rank;
        int idx_end = idx_start + chunk;
        if (proc_rank == proc_size - 1)
        {
            idx_end = i + 1;
        }

        int send_buffer_idx = 0;

        for (idx = idx_start; idx < idx_end; idx++)
        {
            index = (idx + 1) * max_cols + (i + 1 - idx);
            send_buffer[send_buffer_idx++] = index;
            input_itemsets[index] = maximum(input_itemsets[index - 1 - max_cols] + referrence[index],
                                            input_itemsets[index - 1] - penalty,
                                            input_itemsets[index - max_cols] - penalty);
            send_buffer[send_buffer_idx++] = input_itemsets[index];
        }

        if (proc_rank == 0 && proc_size > 1)
        {
            int proc_idx;
            for (proc_idx = 1; proc_idx < proc_size; proc_idx++)
            {
                MPI_Recv(&send_buffer_size, 1, MPI_INT, proc_idx, TAG_SIZE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(send_buffer, send_buffer_size, MPI_INT, proc_idx, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (int it = 0; it < send_buffer_size - 1; it += 2)
                {
                    input_itemsets[send_buffer[it]] = send_buffer[it + 1];
                }
            }
        }
        else if (proc_rank != 0)
        {
            MPI_Send(&send_buffer_idx, 1, MPI_INT, 0, TAG_SIZE, MPI_COMM_WORLD);
            MPI_Send(send_buffer, send_buffer_idx, MPI_INT, 0, TAG_RESULT, MPI_COMM_WORLD);
        }

        MPI_Type_free(&diagonal1);
        MPI_Type_free(&diagonal2);
    }

    printf("Processing bottom-right matrix\n");
    for (int i = max_cols - 4; i >= 0; i--)
    {
        MPI_Bcast(input_itemsets, max_rows * max_cols, MPI_INT, 0, MPI_COMM_WORLD);

        int chunk = (i + 1) / proc_size;
        int idx_start = chunk * proc_rank;
        int idx_end = idx_start + chunk;
        if (proc_rank == proc_size - 1)
        {
            idx_end = i + 1;
        }

        int send_buffer_idx = 0;

        for (idx = idx_start; idx < idx_end; idx++)
        {
            index = (max_cols - idx - 2) * max_cols + idx + max_cols - i - 2;
            send_buffer[send_buffer_idx++] = index;
            input_itemsets[index] = maximum(input_itemsets[index - 1 - max_cols] + referrence[index],
                                            input_itemsets[index - 1] - penalty,
                                            input_itemsets[index - max_cols] - penalty);
            send_buffer[send_buffer_idx++] = input_itemsets[index];
        }

        if (proc_rank == 0 && proc_size > 1)
        {
            int proc_idx;
            for (proc_idx = 1; proc_idx < proc_size; proc_idx++)
            {
                MPI_Recv(&send_buffer_size, 1, MPI_INT, proc_idx, TAG_SIZE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(send_buffer, send_buffer_size, MPI_INT, proc_idx, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (int it = 0; it < send_buffer_size - 1; it += 2)
                {
                    input_itemsets[send_buffer[it]] = send_buffer[it + 1];
                }
            }
        }
        else if (proc_rank != 0)
        {
            MPI_Send(&send_buffer_idx, 1, MPI_INT, 0, TAG_SIZE, MPI_COMM_WORLD);
            MPI_Send(send_buffer, send_buffer_idx, MPI_INT, 0, TAG_RESULT, MPI_COMM_WORLD);
        }
    }

if (proc_rank == 0)
{
    #define TRACEBACK
    #ifdef TRACEBACK

        FILE *fpo = fopen("result.txt", "w");
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
}
    free(referrence);
    free(input_itemsets);
    free(output_itemsets);
}
