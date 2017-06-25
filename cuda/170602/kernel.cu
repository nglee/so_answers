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
#define CheckCudaError(call)    _CheckCudaError((call), __FILE__, __LINE__)

struct s12 {
    int a; int b; int c;
};
struct s16 {
    int a; int b; int c; int d;
};
struct s20 {
    int a; int b; int c; int d; int e;
};
struct s24 {
    int a; int b; int c; int d; int e; int f;
};
struct s28 {
    int a; int b; int c; int d; int e; int f; int g;
};
struct s32 {
    int a; int b; int c; int d; int e; int f; int g; int h;
};

#define TESTSIZE    1024

template <typename T>
__global__ void test_kernel(T* d)
{
    __shared__ T s[TESTSIZE / 2];

    // copy first half of data to SMEM
    if (threadIdx.x < TESTSIZE / 2)
        s[threadIdx.x] = d[threadIdx.x];
    __syncthreads();

    // copy SMEM to second half
    if (threadIdx.x >= TESTSIZE / 2)
        d[threadIdx.x] = s[threadIdx.x % (TESTSIZE / 2)];
}

template <typename T>
__global__ void test_kernel_conflict(T* d)
{
    __shared__ T s[TESTSIZE];

    if (threadIdx.x < TESTSIZE / 2)
        s[threadIdx.x * 2] = d[threadIdx.x]; // bank conflict
    __syncthreads();
    if (threadIdx.x >= TESTSIZE / 2)
        d[threadIdx.x] = s[threadIdx.x * 2 % TESTSIZE]; // bank conflict
}

template <typename T>
void test()
{
    T h[TESTSIZE];
    memset(h, 0, sizeof(h));
    for (int i = 0; i < TESTSIZE / 2; i++)
        *(int*)&h[i] = TESTSIZE / 2 - i;

    T *d;
    CheckCudaError(cudaMalloc(&d, sizeof(h)));
    CheckCudaError(cudaMemcpy(d, h, sizeof(h), cudaMemcpyHostToDevice));

    test_kernel<<<1, TESTSIZE>>>(d);

    CheckCudaError(cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost));
    printf("first element : %d, last element : %d\n", *(int*)&h[0], *(int*)&h[TESTSIZE - 1]); // expected numbers are: TESTSIZE / 2,  1

    CheckCudaError(cudaMemcpy(d, h, sizeof(h), cudaMemcpyHostToDevice));

    test_kernel_conflict<<<1, TESTSIZE>>>(d);

    CheckCudaError(cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost));
    printf("first element : %d, last element : %d\n", *(int*)&h[0], *(int*)&h[TESTSIZE - 1]); // expected numbers are: TESTSIZE / 2,  1

    CheckCudaError(cudaFree(d));
}

int main()
{
    test<int>();                //  4B per element
    test<long long int>();      //  8B per element
    test<struct s12>();         // 12B per element
    test<struct s16>();         // 16B per element
    test<struct s20>();         // 20B per element
    test<struct s24>();         // 24B per element
    test<struct s28>();         // 28B per element
    test<struct s32>();         // 32B per element
}
