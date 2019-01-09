#include <stdio.h>

__global__ void kernel2(int *data)
{
	__shared__ volatile int data_s[32];

	size_t t_id = threadIdx.x;

	data_s[t_id] = data[t_id];

	if (1 <= t_id)
		data_s[t_id] += data_s[t_id - 1];
	if (2 <= t_id)
		data_s[t_id] += data_s[t_id - 2];
	if (4 <= t_id)
		data_s[t_id] += data_s[t_id - 4];
	if (8 <= t_id)
		data_s[t_id] += data_s[t_id - 8];
	if (16 <= t_id)
		data_s[t_id] += data_s[t_id - 16];

	data[t_id] = data_s[t_id];
}

int main()
{
	int data[32];
	int result[32];

	int *data_d;
	cudaMalloc(&data_d, sizeof(data));

	for (int i = 0; i < 32; i++)
		data[i] = i;

	dim3 gridDim(1);
	dim3 blockDim(32);

	cudaMemcpy(data_d, data, sizeof(data), cudaMemcpyHostToDevice);
	kernel2<<<gridDim, blockDim>>>(data_d);
	cudaMemcpy(result, data_d, sizeof(data), cudaMemcpyDeviceToHost);

	printf("kernel2 : ");
	for (int i = 0; i < 32; i++)
		printf("%4i ", result[i]);
	printf("(device, shared memory with volatile qualifier)\n");
}
