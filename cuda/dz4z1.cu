#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void Usage(char *prog_name);

#define ACCURACY 0.01

typedef float float_type;

#define KERNEL_ITERATIONS 1000

__device__ void warp_reduce(volatile float_type *sdata, const unsigned int thread_id)
{
    sdata[thread_id] += sdata[thread_id + 32];
    sdata[thread_id] += sdata[thread_id + 16];
    sdata[thread_id] += sdata[thread_id + 8];
    sdata[thread_id] += sdata[thread_id + 4];
    sdata[thread_id] += sdata[thread_id + 2];
    sdata[thread_id] += sdata[thread_id + 1];
}

__global__ void reduce_pi(float_type *gdata)
{
    extern __shared__ float_type sdata[];

    const unsigned int thread_id = threadIdx.x;
    const unsigned long long int i = (((unsigned long long int)blockIdx.x) * blockDim.x + threadIdx.x) * KERNEL_ITERATIONS;

    float_type current_thread_factor = 0.0f;
    for (int it = 0; it < KERNEL_ITERATIONS; it++)
    {
        const float factor = ((i + it) & 1) ? -1.0f : 1.0f;
        current_thread_factor += factor / (((i + it) << 1) + 1);
    }

    sdata[thread_id] = current_thread_factor;

    __syncthreads();

    // reduction in shared memory
    for (unsigned int stride = blockDim.x >> 1; stride > 32; stride >>= 1)
    {
        if (thread_id < stride) {
            sdata[thread_id] += sdata[thread_id + stride];
        }
        __syncthreads();
    }

    if (thread_id < 32)
        warp_reduce(sdata, thread_id);

    // write result for this block to global memory
    if (thread_id == 0)
        gdata[blockIdx.x] = sdata[0];
}

double sequential_solution(int argc, char *argv[])
{
    long long n, i;
    double factor = 0.0;
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
    double factor = 0.0;
    double sum = 0.0;
    float_type *dev_sum;
    float_type *cpu_sum;

    if (argc != 2)
        Usage(argv[0]);
    n = strtoll(argv[1], NULL, 10);
    if (n < 1)
        Usage(argv[0]);

    printf("Before for loop, factor = %f.\n", factor);

    long long block_size = 1024;
    long long grid_size = ceil(((double)n / KERNEL_ITERATIONS) / block_size);

    cpu_sum = (float_type*)calloc(grid_size, sizeof(float_type));
    cudaMalloc(&dev_sum, grid_size * sizeof(float_type));

    reduce_pi<<< grid_size, block_size, block_size * sizeof(float_type) >>>(dev_sum);

    cudaMemcpy(cpu_sum, dev_sum, grid_size * sizeof(float_type), cudaMemcpyDeviceToHost);

    factor = ((n - 1) % 2 == 0) ? 1.0 : -1.0;
    for (i = 0; i < grid_size; i++)
        sum += cpu_sum[i];

    cudaFree(dev_sum);
    free(cpu_sum);

    printf("After for loop, factor = %f.\n", factor);

    sum = 4.0 * sum;
    printf("With n = %lld terms\n", n);
    printf("   Our estimate of pi = %.14f\n", sum);
    printf("   Ref estimate of pi = %.14f\n", 4.0 * atan(1.0));
    return sum;
}

int main(int argc, char *argv[])
{
    float elapsed_time_seq;
    cudaEvent_t start_time_seq, end_time_seq;
    cudaEventCreate(&start_time_seq);
    cudaEventCreate(&end_time_seq);
    float elapsed_time_parallel;
    cudaEvent_t start_time_parallel, end_time_parallel;
    cudaEventCreate(&start_time_parallel);
    cudaEventCreate(&end_time_parallel);

    printf("---------------------Sequential execution---------------------\n");
    cudaEventRecord(start_time_seq, 0);
    double sum_seq = sequential_solution(argc, argv);
    cudaEventRecord(end_time_seq, 0);
    cudaEventSynchronize(end_time_seq);
    cudaEventElapsedTime(&elapsed_time_seq, start_time_seq, end_time_seq);

    printf("----------------------Parallel execution----------------------\n");
    cudaEventRecord(start_time_parallel, 0);
    double sum_parallel = parallel_solution(argc, argv);
    cudaEventRecord(end_time_parallel, 0);
    cudaEventSynchronize(end_time_parallel);
    cudaEventElapsedTime(&elapsed_time_parallel, start_time_parallel, end_time_parallel);

    printf("\nSequential elapsed time: %fs\n", elapsed_time_seq / 1000.0);
    printf("Parallel elapsed time: %fs\n", elapsed_time_parallel / 1000.0);

    if (fabs(sum_seq - sum_parallel) < ACCURACY)
        printf("Test PASSED\n");
    else
        printf("Test FAILED\n");

    return 0;
}

void Usage(char *prog_name)
{
    fprintf(stderr, "usage: %s <n>\n", prog_name);
    fprintf(stderr, "   n is the number of terms and should be >= 1\n");
    exit(0);
}
