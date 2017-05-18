#include <stdlib.h>
#include <stdio.h>

#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

void _CheckCudaError(const cudaError_t cudaError, const char* file, const int line)
{
	if (cudaError != cudaSuccess) {
		std::cout << "[CUDA ERROR] " << cudaGetErrorString(cudaError) << " (" << file << ":" << line << ")\n";
		exit(EXIT_FAILURE);
	}
}
#define CheckCudaError(call) _CheckCudaError((call), __FILE__, __LINE__)

__global__ void add(const int *a, const int *b, int *c)
{
	int tid = blockIdx.x;

	if (tid < gridDim.x)
		c[tid] = a[tid] + b[(gridDim.x - 1)- tid];
}

int main()
{
	int h_a[6] = { 1, 2, 3, 4, 5, 6 };
	int h_b[6] = { 10, 20, 30, 40, 50, 60 };
	int h_c[6];

	int* d_a;
	int* d_b;
	int* d_c;

	CheckCudaError(cudaMalloc(&d_a, 6 * sizeof(int)));
	CheckCudaError(cudaMalloc(&d_b, 6 * sizeof(int)));
	CheckCudaError(cudaMalloc(&d_c, 6 * sizeof(int)));

	CheckCudaError(cudaMemcpy(d_a, h_a, 6 * sizeof(int), cudaMemcpyHostToDevice));
	CheckCudaError(cudaMemcpy(d_b, h_b, 6 * sizeof(int), cudaMemcpyHostToDevice));

	add<<<6, 1>>>(d_a, d_b, d_c);
	CheckCudaError(cudaDeviceSynchronize());

	CheckCudaError(cudaMemcpy(h_c, d_c, 6 * sizeof(int), cudaMemcpyDeviceToHost));
	for (int i = 0; i < 6; i++)
		printf("%d ", h_c[i]);
	printf("\n");

	CheckCudaError(cudaFree(d_a));
	CheckCudaError(cudaFree(d_b));
	CheckCudaError(cudaFree(d_c));
}