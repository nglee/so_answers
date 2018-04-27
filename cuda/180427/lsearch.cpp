#include <iostream>
#include <chrono>
#include <stdlib.h>

void search(int * arr, int N, int x)
{
    for (int i = 0; i<N; i++)
    {
        if (arr[i] == x)
        {
            std::cout << "Found" << std::endl;
            break;
        }
     }
}

int main()
{
    // Allocate array
    const size_t memSize = 2L * 1024L * 1024L * 1024L; // 2GB
    const int arrSize = memSize / sizeof(int);
    int *arr = (int *)malloc(memSize);

    // Set array elements
    for (int i = 0; i < arrSize; i++)
        arr[i] = i;

    // Measure execution time using chrono library
    auto start = std::chrono::high_resolution_clock::now();
    search(arr, arrSize, arrSize - 1);
    auto end = std::chrono::high_resolution_clock::now();

    // Free allocated array
    free(arr);

    // Print execution time
    float milliseconds = 0;
    milliseconds = std::chrono::duration<float, std::milli>(end - start).count();
    std::cout << milliseconds << " ms" << std::endl;
}
