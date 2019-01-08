#include <stdio.h>
#include <string.h>

int main()
{
	int data[32];
	int dummy[32];

	for (int i = 0; i < 32; i++)
		data[i] = i;

	memcpy(dummy, data, sizeof(data));
	for (int i = 1; i < 32; i++)
		data[i] += dummy[i - 1];
	memcpy(dummy, data, sizeof(data));
	for (int i = 2; i < 32; i++)
		data[i] += dummy[i - 2];
	memcpy(dummy, data, sizeof(data));
	for (int i = 4; i < 32; i++)
		data[i] += dummy[i - 4];
	memcpy(dummy, data, sizeof(data));
	for (int i = 8; i < 32; i++)
		data[i] += dummy[i - 8];
	memcpy(dummy, data, sizeof(data));
	for (int i = 16; i < 32; i++)
		data[i] += dummy[i - 16];

	printf("kernel  : ");
	for (int i = 0; i < 32; i++)
		printf("%4i ", data[i]);
	printf("\n");
}
