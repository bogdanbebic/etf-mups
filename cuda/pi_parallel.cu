#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__device__ void warp_reduce(volatile double *sdata, const unsigned long long int thread_id)
{
    sdata[thread_id] += sdata[thread_id + 32];
    sdata[thread_id] += sdata[thread_id + 16];
    sdata[thread_id] += sdata[thread_id + 8];
    sdata[thread_id] += sdata[thread_id + 4];
    sdata[thread_id] += sdata[thread_id + 2];
    sdata[thread_id] += sdata[thread_id + 1];
}

__global__ void reduce_pi(double *gdata)
{
    extern __shared__ double sdata[];

    const unsigned long long int thread_id = threadIdx.x;
    const unsigned long long int i = ((unsigned long long int)blockIdx.x) * blockDim.x + threadIdx.x;

    const double factor = (i & 1) ? -1.0 : 1.0;
    sdata[thread_id] = factor / ((i << 1) + 1);

    __syncthreads();

    // reduction in shared memory
    for (unsigned long long int stride = blockDim.x >> 1; stride > 32; stride >>= 1)
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

void Usage(char *prog_name);

int main(int argc, char *argv[])
{
    long long n, i;
    double factor = 0.0;
    double sum = 0.0;
    double *dev_sum;
    double *cpu_sum;

    if (argc != 2)
        Usage(argv[0]);
    n = strtoll(argv[1], NULL, 10);
    if (n < 1)
        Usage(argv[0]);

    printf("Before for loop, factor = %f.\n", factor);

    long long block_size = 1024;
    long long grid_size = ceil(((double)n) / block_size);

    cpu_sum = (double*)calloc(grid_size, sizeof(double));
    cudaMalloc(&dev_sum, grid_size * sizeof(double));

    reduce_pi<<< grid_size, block_size, block_size * sizeof(double) >>>(dev_sum);

    cudaMemcpy(cpu_sum, dev_sum, grid_size * sizeof(double), cudaMemcpyDeviceToHost);

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
    return 0;
}

void Usage(char *prog_name)
{
    fprintf(stderr, "usage: %s <thread_count> <n>\n", prog_name);
    fprintf(stderr, "   n is the number of terms and should be >= 1\n");
    exit(0);
}
