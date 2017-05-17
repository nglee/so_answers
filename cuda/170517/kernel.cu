#include <stdlib.h>
#include <stdio.h>

#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <math_functions.h>

void _CheckCudaError(const cudaError_t cudaError, const char* file, const int line)
{
	if (cudaError != cudaSuccess) {
		std::cout << "[CUDA ERROR] " << cudaGetErrorString(cudaError) << " (" << file << ":" << line << ")\n";
		exit(EXIT_FAILURE);
	}
}
#define CheckCudaError(call) _CheckCudaError((call), __FILE__, __LINE__)

#define N       32
#define blkSize 32
#define grdSize 1

__global__ void init(unsigned long long seed, curandState_t* states)
{
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed, id, 0, &states[id]);
}

__global__ void build(int* d_data, curandState_t* states)
{
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	d_data[id] = ceilf(curand_uniform(&states[id]) * N);
}

// Finds the maximum element within a warp and gives the maximum element to
// thread with lane id 0. Note that other elements do not get lost but their
// positions are shuffled.
__inline__ __device__ int warpMax(int data, unsigned int threadId)
{
	for (int mask = 16; mask > 0; mask /= 2) {
		int dual_data = __shfl_xor(data, mask, 32);
		if (threadId & mask)
			data = min(data, dual_data);
		else
			data = max(data, dual_data);
	}
	return data;
}

__global__ void selection32(int* d_data, int* d_data_sorted)
{
	unsigned int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int laneId = threadIdx.x % 32;

	int n = N;
	while(n-- > 0) { 
		// get the maximum element among d_data and put it in d_data_sorted[n]
		int data = d_data[threadId];
		data = warpMax(data, threadId);
		d_data[threadId] = data;

		// now maximum element is in d_data[0]
		if (laneId == 0) {
			d_data_sorted[n] = d_data[0];
			d_data[0] = INT_MIN; // this element is ignored from now on
		}
	}
}

int main()
{
	int* d_data;
	int* d_data_sorted;
	int* h_data;
	int* h_data_sorted;
	curandState_t* states;

	// allocate host and device memory
	CheckCudaError(cudaMalloc(&d_data, sizeof(int) * N));
	CheckCudaError(cudaMalloc(&d_data_sorted, sizeof(int) * N));
	h_data = (int*)malloc(sizeof(int) * N);
	h_data_sorted = (int*)malloc(sizeof(int) * N);
	CheckCudaError(cudaMalloc(&states, sizeof(curandState_t) * N));

	// build random data
	init<<<grdSize, blkSize>>>(time(0), states);
	build<<<grdSize, blkSize>>>(d_data, states);

	// print random data
	CheckCudaError(cudaMemcpy(h_data, d_data, sizeof(int) * N, cudaMemcpyDeviceToHost));
	for (int i = 0; i < N; i++)
		printf("%d ", h_data[i]);
	printf("\n");

	// selection-sort
	selection32<<<grdSize, blkSize>>>(d_data, d_data_sorted);

	// print sorted data
	CheckCudaError(cudaMemcpy(h_data_sorted, d_data_sorted, sizeof(int) * N, cudaMemcpyDeviceToHost));
	for (int i = 0; i < N; i++)
		printf("%d ", h_data_sorted[i]);
	printf("\n");

	// free allocated memory
	CheckCudaError(cudaFree(d_data));
	CheckCudaError(cudaFree(d_data_sorted));
	free(h_data);
	free(h_data_sorted);
	CheckCudaError(cudaFree(states));

    return 0;
}