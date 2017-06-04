#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

void _CheckCudaError(cudaError_t ret, char *file, int line)
{
	if (ret != cudaSuccess) {
		printf("%s - %s (%s:%d)\n", cudaGetErrorName(ret), cudaGetErrorString(ret), file, line);
		exit(EXIT_FAILURE);
	}
}
#define CheckCudaError(call)	_CheckCudaError((call), __FILE__, __LINE__)

__global__ void copyKernel(const unsigned char* a, unsigned char* b)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
	b[i] = a[i];
}

int main()
{
	// test up to 1M element
	const int arraySize = 1024 * 1024;
	const int memSize = arraySize * sizeof(unsigned char);

	unsigned char* a = (unsigned char*)malloc(memSize);
	unsigned char* b = (unsigned char*)malloc(memSize);

	for (int i = 0; i < arraySize; i++)
		a[i] = i%256;

	unsigned char* d_a;
	unsigned char* d_b;

	CheckCudaError(cudaMalloc(&d_a, memSize));
	CheckCudaError(cudaMalloc(&d_b, memSize));

	CheckCudaError(cudaMemcpy(d_a, a, memSize, cudaMemcpyHostToDevice));

	for (int i = 1; i <= 31; i++)
		copyKernel<<<1, i>>>(d_a, d_b);

	// test from 1 threads to 1K threads
	for (int i = 32; i <= 1024; i *= 2) {
		CheckCudaError(cudaMemset(d_b, 0, memSize));    // reset device memory of b to 0

		copyKernel<<<1, i>>>(d_a, d_b);

		CheckCudaError(cudaMemcpy(b, d_b, memSize, cudaMemcpyDeviceToHost));

		for (int j = 0; j < i; j++)
			if (b[j] != j%256) {                        // if something is wrong with kernel
				printf("i = %d, j = %d, b[j] = %d\n", i, j, b[j]);
				exit(EXIT_FAILURE);
			} else
				b[j] = 0;                               // reset host memory of b to 0
	} 

	// test from 1K threads to 1M threads
	for (int i = 1024; i <= 1024 * 1024; i *= 2) {
		CheckCudaError(cudaMemset(d_b, 0, memSize));    // reset device memory of b to 0

		copyKernel<<<1024, i/1024>>>(d_a, d_b);

		CheckCudaError(cudaMemcpy(b, d_b, memSize, cudaMemcpyDeviceToHost));

		for (int j = 0; j < i; j++)
			if (b[j] != j%256) {                        // if something is wrong with kernel
				printf("i = %d, j = %d, b[j] = %d\n", i, j, b[j]);
				exit(EXIT_FAILURE);
			} else
				b[j] = 0;                               // reset host memory of b to 0
	} 

	CheckCudaError(cudaDeviceReset());

    return 0;
}