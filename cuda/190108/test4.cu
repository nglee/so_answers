#include <stdio.h>

__global__ void kernel4(int *data)
{
	__shared__ int data_s[32];

	size_t t_id = threadIdx.x;

	data_s[t_id] = data[t_id];

	int v = data_s[t_id];

	unsigned mask = 0xffffffff;     __syncwarp(mask);

	mask = __ballot_sync(0xffffffff, 1 <= t_id);
	if (1 <= t_id) {
		v += data_s[t_id - 1];  __syncwarp(mask);
		data_s[t_id] = v;       __syncwarp(mask);
	}
	mask = __ballot_sync(0xffffffff, 2 <= t_id);
	if (2 <= t_id) {
		v += data_s[t_id - 2];  __syncwarp(mask);
		data_s[t_id] = v;       __syncwarp(mask);
	}
	mask = __ballot_sync(0xffffffff, 4 <= t_id);
	if (4 <= t_id) {
		v += data_s[t_id - 4];  __syncwarp(mask);
		data_s[t_id] = v;       __syncwarp(mask);
	}
	mask = __ballot_sync(0xffffffff, 8 <= t_id);
	if (8 <= t_id) {
		v += data_s[t_id - 8];  __syncwarp(mask);
		data_s[t_id] = v;       __syncwarp(mask);
	}
	mask = __ballot_sync(0xffffffff, 16 <= t_id);
	if (16 <= t_id) {
		v += data_s[t_id - 16]; __syncwarp(mask);
		data_s[t_id] = v;
	}

	data[t_id] = data_s[t_id];
}

int main()
{
	int version;
	cudaRuntimeGetVersion(&version);
	if (version < 9000) {
		printf("Test not supported in this CUDA runtime version (%d)\n", version);
		exit(EXIT_SUCCESS);
	}

	int data[32];
	int result[32];

	int *data_d;
	cudaMalloc(&data_d, sizeof(data));

	for (int i = 0; i < 32; i++)
		data[i] = i;

	dim3 gridDim(1);
	dim3 blockDim(32);

	cudaMemcpy(data_d, data, sizeof(data), cudaMemcpyHostToDevice);
	kernel4<<<gridDim, blockDim>>>(data_d);
	cudaMemcpy(result, data_d, sizeof(data), cudaMemcpyDeviceToHost);

	printf("kernel4 : ");
	for (int i = 0; i < 32; i++)
		printf("%4i ", result[i]);
	printf("(device, shared memory with masked __syncwarp())\n");
}
