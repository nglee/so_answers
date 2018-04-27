#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void
setArray(int *d_arr, int arrSize)
{
    int t_id = blockIdx.x * blockDim.x + threadIdx.x;

    for (int idx = t_id; idx < arrSize; idx += gridDim.x * blockDim.x)
        d_arr[idx] = idx;
}

__global__ void
search(int *d_arr, int arrSize, int searchItem)
{
    int t_id = blockIdx.x * blockDim.x + threadIdx.x;

    for (int idx = t_id; idx < arrSize; idx += gridDim.x * blockDim.x)
        if (d_arr[idx] == searchItem)
            printf("Found\n");
}

#define NUMBER_OF_BLOCKS    4
#define THREADS_PER_BLOCK   1024

int main()
{
    // Allocate array
    const size_t memSize = 2L * 1024L * 1024L * 1024L; // 2GB
    const int arrSize = memSize / sizeof(int);
    int *d_arr;
    cudaMalloc(&d_arr, sizeof(int) * arrSize);

    // Set array elements
    setArray<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(d_arr, arrSize);

    // Measure kernel execution time using CUDA events
    // https://devblogs.nvidia.com/how-implement-performance-metrics-cuda-cc/
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    search<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(d_arr, arrSize, arrSize - 1);
    cudaEventRecord(stop);

    // Free allocated array
    cudaFree(d_arr);

    // Print execution time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << milliseconds << " ms" << std::endl;
}
