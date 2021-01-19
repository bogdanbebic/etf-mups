#define LIMIT -999
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

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

__device__ int maximum_dev(int a, int b, int c)
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
    runTest(argc, argv);

    return EXIT_SUCCESS;
}

void usage(int argc, char **argv)
{
    fprintf(stderr, "Usage: %s <max_rows/max_cols> <penalty>\n", argv[0]);
    fprintf(stderr, "\t<dimension>      - x and y dimensions\n");
    fprintf(stderr, "\t<penalty>        - penalty(positive integer)\n");
    exit(1);
}

typedef int thread_idx_t;

__global__ void kernel_init_reference(int *reference, int *blosum62, int *input_itemsets, int num_rows, int num_cols)
{
    const thread_idx_t thread_idx_i = threadIdx.y + (thread_idx_t)blockIdx.y * blockDim.y;
    const thread_idx_t thread_idx_j = threadIdx.x + (thread_idx_t)blockIdx.x * blockDim.x;

    if (thread_idx_i < num_rows && thread_idx_j < num_cols)
        reference[thread_idx_i * num_cols + thread_idx_j] =
            blosum62
            [
                input_itemsets[thread_idx_i * num_cols]
            * 24 +
                input_itemsets[thread_idx_j]
            ];
}

__global__ void kernel_init_input_itemsets(int *input_itemsets, int num_cols, int penalty)
{
    const thread_idx_t thread_idx = threadIdx.x + (thread_idx_t)blockIdx.x * blockDim.x;

    if (thread_idx < num_cols)
    {
        input_itemsets[thread_idx * num_cols] = -thread_idx * penalty;
        input_itemsets[thread_idx] = -thread_idx * penalty;
    }
}

__global__ void kernel_top_left_processing(int *input_itemsets, int *reference, int num_cols, int i, int penalty) {
    __shared__ int s_input_itemsets_nw[1024];
    __shared__ int s_input_itemsets_w[1025];
    const thread_idx_t thread_idx = threadIdx.x + (thread_idx_t)blockIdx.x * blockDim.x;
    const thread_idx_t tid = threadIdx.x;

    const int index = (thread_idx + 1) * num_cols + (i + 1 - thread_idx);
    s_input_itemsets_nw[tid] = input_itemsets[index - 1 - num_cols] + reference[index];
    s_input_itemsets_w[tid + 1] = input_itemsets[index - 1];

    if (tid == 0)
        s_input_itemsets_w[0] = input_itemsets[index - num_cols];

    __syncthreads();

    if (thread_idx <= i)
    {
        input_itemsets[index] = maximum_dev(s_input_itemsets_nw[tid],
            s_input_itemsets_w[tid + 1] - penalty,
            s_input_itemsets_w[tid] - penalty);
    }
}

__global__ void kernel_bottom_right_processing(int *input_itemsets, int *reference, int num_cols, int i, int penalty) {
    __shared__ int s_input_itemsets_nw[1024];
    __shared__ int s_input_itemsets_n[1025];
    const thread_idx_t thread_idx = threadIdx.x + (thread_idx_t)blockIdx.x * blockDim.x;
    const thread_idx_t tid = threadIdx.x;
    const int index = (num_cols - thread_idx - 2) * num_cols + thread_idx + num_cols - i - 2;

    s_input_itemsets_nw[tid] = input_itemsets[index - 1 - num_cols] + reference[index];
    s_input_itemsets_n[tid + 1] = input_itemsets[index - num_cols];

    if (tid == 0)
        s_input_itemsets_n[0] = input_itemsets[index - 1];

    __syncthreads();

    if (thread_idx <= i) {
        input_itemsets[index] = maximum_dev(s_input_itemsets_nw[tid],
            s_input_itemsets_n[tid + 1] - penalty,
            s_input_itemsets_n[tid] - penalty);
    }
}

void runTest(int argc, char **argv)
{
    int max_rows, max_cols, penalty;
    int *input_itemsets, *reference;

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
    cudaMallocHost(&reference, max_rows * max_cols * sizeof(int));
    input_itemsets = (int *)calloc(max_rows * max_cols, sizeof(int));

    if (!input_itemsets)
        fprintf(stderr, "error: can not allocate memory");

    // srand(time(NULL));
    srand(0);

    printf("Start Needleman-Wunsch\n");

    for (int i = 1; i < max_rows; i++)
    {
        input_itemsets[i * max_cols] = rand() % 10 + 1;
    }
    for (int j = 1; j < max_cols; j++)
    {
        input_itemsets[j] = rand() % 10 + 1;
    }

    int *reference_dev;
    int *input_itemsets_dev;
    int *blosum62_dev;

    cudaStream_t reference_copy_stream;

    cudaStreamCreateWithFlags(&reference_copy_stream, cudaStreamNonBlocking);

    const size_t size = max_rows * max_cols * sizeof(int);

    cudaMalloc(&reference_dev, size);
    cudaMalloc(&input_itemsets_dev, size);
    cudaMalloc(&blosum62_dev, sizeof(blosum62));

    cudaMemcpy(input_itemsets_dev, input_itemsets, size, cudaMemcpyHostToDevice);
    cudaMemcpy(blosum62_dev, blosum62, sizeof(blosum62), cudaMemcpyHostToDevice);

    const dim3 block_size(32, 32, 1);
    const size_t grid_cols = (max_cols + block_size.x - 1) / block_size.x;
    const size_t grid_rows = (max_rows + block_size.y - 1) / block_size.y;
    const dim3 grid_size(grid_cols, grid_rows, 1);

    kernel_init_reference<<< grid_size, block_size >>>(reference_dev, blosum62_dev, input_itemsets_dev, max_rows, max_cols);

    cudaMemcpyAsync(reference, reference_dev, size, cudaMemcpyDeviceToHost, reference_copy_stream);
    kernel_init_input_itemsets<<< ceil(max_cols / 1024.0), 1024 >>>(input_itemsets_dev, max_cols, penalty);

    printf("Processing top-left matrix\n");
    for (int i = 0; i < max_cols - 2; i++)
    {
        kernel_top_left_processing<<<ceil((i + 1) / 1024.0), 1024>>>(input_itemsets_dev, reference_dev, max_cols, i, penalty);
    }

    printf("Processing bottom-right matrix\n");
    for (int i = max_cols - 4; i >= 0; i--)
    {
        kernel_bottom_right_processing<<<ceil((i + 1) / 1024.0), 1024>>>(input_itemsets_dev, reference_dev, max_cols, i, penalty);
    }

    cudaMemcpy(input_itemsets, input_itemsets_dev, size, cudaMemcpyDeviceToHost);

    cudaStreamDestroy(reference_copy_stream);

    cudaFree(reference_dev);
    cudaFree(input_itemsets_dev);
    cudaFree(blosum62_dev);

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

        //traceback = maximum(nw, w, n);
        int new_nw, new_w, new_n;
        new_nw = nw + reference[i * max_cols + j];
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
    }

    fclose(fpo);

#endif

    cudaFreeHost(reference);
    free(input_itemsets);
}
