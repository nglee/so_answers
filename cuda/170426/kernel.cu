#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>

#define N 100000    // number of trials
#define MAX 10      // number of the maximum desired value

__global__ void init(unsigned int seed, curandState_t* states)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, /* the seed can be the same for each thread, here we pass the time from CPU */
                id,   /* the sequence number should be different for each core */
                0,    /* the offset is how much extra we advance in the sequence for each call, can be 0 */
                &states[id]);
}

__global__ void random_casting(curandState_t* states, unsigned int* numbers)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    numbers[id] = curand_uniform(&states[id]) * MAX;
}

__global__ void random_ceiling(curandState_t* states, unsigned int* numbers)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    numbers[id] = ceilf(curand_uniform(&states[id]) * MAX);
}

void histogram(unsigned int* h_nums)
{
    int hist[MAX + 1]; // from 0 to MAX, so (MAX + 1) elements needed
    for (int i = 0; i < MAX + 1; i++)
        hist[i] = 0;
    for (int i = 0; i < N; i++)
        hist[h_nums[i]]++;
    for (int i = 0; i < MAX + 1; i++)
        printf("%2d : %6d\n", i, hist[i]);
}

int main() {
    curandState_t* states;
    cudaMalloc((void**)&states, N * sizeof(curandState_t));

    // initialize the random states
    dim3 blkDim = 1000;
    dim3 grdDim = (N + blkDim.x - 1) / blkDim.x;
    init<<<grdDim, blkDim >>>(time(0), states);

    // allocate an array of unsigned ints on the CPU and GPU
    unsigned int h_nums[N];
    unsigned int* d_nums;
    cudaMalloc((void**)&d_nums, N * sizeof(unsigned int));

    // get random number with casting
    random_casting<<<grdDim, blkDim >>>(states, d_nums);
    cudaMemcpy(h_nums, d_nums, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    printf("Histogram for random numbers generated with casting\n");
    histogram(h_nums);

    // get random number with ceiling
    random_ceiling<<<grdDim, blkDim >>>(states, d_nums);
    cudaMemcpy(h_nums, d_nums, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    printf("Histogram for random numbers generated with ceiling\n");
    histogram(h_nums);

    cudaFree(states);
    cudaFree(d_nums);

    return 0;
}
