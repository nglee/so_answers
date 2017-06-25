#include <stdio.h>
#include <stdlib.h>

extern "C" {
    #include "timer.h"
}

__global__ void vecAdd(float* a, float* b, float* c) {
	/* Calculate index for this thread */
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	/* Compute the element of C */
 	c[i] = a[i] + b[i];
}

void compute_vec_add(int N, float *a, float* b, float *c)
{
    for (int i = 0; i < N; i++)
        c[i] = a[i] + b[i];
}

int main() {

	float *d_a, *d_b, *d_c;
	float *h_a, *h_b, *h_c, *h_temp;
	int i;
	int N = 1024 * 1024 * 512;

	struct stopwatch_t* timer = NULL;

	long double t_pcie_htd, t_pcie_dth, t_kernel, t_cpu;

	/* Setup timers */
	stopwatch_init();
	timer = stopwatch_create();

	h_a = (float *) malloc(sizeof(float) * N);
	h_b = (float *) malloc(sizeof(float) * N);
	h_c = (float *) malloc(sizeof(float) * N);

	for (i = 0; i < N; i++) {
		h_a[i] = (float) (rand() % 100) / 10.0;
		h_b[i] = (float) (rand() % 100) / 10.0;
		h_c[i] = 0.0;
	}

	cudaMalloc(&d_a, sizeof(float) * N);
	cudaMalloc(&d_b, sizeof(float) * N);
	cudaMalloc(&d_c, sizeof(float) * N);

	stopwatch_start(timer);
	cudaMemcpy(d_a, h_a, sizeof(float) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, sizeof(float) * N, cudaMemcpyHostToDevice);
	t_pcie_htd = stopwatch_stop(timer);
	fprintf(stderr, "Time to transfer data from host to device: %Lg secs\n",t_pcie_htd);

	dim3 GS(N / 256, 1, 1);
	dim3 BS(256, 1, 1);

	stopwatch_start(timer);
	vecAdd<<<GS, BS>>>(d_a, d_b, d_c);
	cudaThreadSynchronize();
	t_kernel = stopwatch_stop(timer);
	fprintf(stderr, "Time to execute GPU kernel: %Lg secs\n", t_kernel);

	stopwatch_start(timer);
	cudaMemcpy(h_c, d_c, sizeof(float) * N, cudaMemcpyDeviceToHost);
	t_pcie_dth = stopwatch_stop(timer);
	fprintf(stderr, "Time to transfer data from device to host: %Lg secs\n",t_pcie_dth);

	h_temp = (float *) malloc(sizeof(float) * N);

	stopwatch_start(timer);
	compute_vec_add(N, h_a, h_b, h_temp);
	t_cpu = stopwatch_stop(timer);
	fprintf(stderr, "Time to execute CPU program: %Lg secs\n", t_cpu);

	int cnt = 0;
	for (int i = 0; i < N; i++) {
		if (abs(h_temp[i] - h_c[i]) > 1e-5)
  			cnt++;
	}
	fprintf(stderr, "number of errors: %d out of %d\n", cnt, N);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	free(h_a);
	free(h_b);
	free(h_c);

	stopwatch_destroy(timer);

	if (cnt == 0) {
		printf("\n\nSuccess\n");
	}
}
