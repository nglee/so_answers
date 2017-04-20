/*
 * Compile      : nvcc main.cu -lX11
 *
 * Prerequisite : sudo apt install cimg-dev
 */
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "CImg.h"
#include <iostream>

using namespace std;
using namespace cimg_library;

__global__ void rgb2gray(unsigned char * d_src, unsigned char * d_dst, int width, int height)
{
	int pos_x = blockIdx.x * blockDim.x + threadIdx.x;
	int pos_y = blockIdx.y * blockDim.y + threadIdx.y;

	if (pos_x >= width || pos_y >= height)
	    return;

	unsigned char r = d_src[pos_y * width + pos_x];
	unsigned char g = d_src[(height + pos_y ) * width + pos_x];
	unsigned char b = d_src[(height * 2 + pos_y) * width + pos_x];

	unsigned int _gray = (unsigned int)((float)(r + g + b) / 3.0f + 0.5);
	unsigned char gray = _gray > 255 ? 255 : _gray;

	d_dst[pos_y * width + pos_x] = gray;
}


int main()
{
    //Load image
    CImg<unsigned char> src("lena.gif");
    int width = src.width();
    int height = src.height();
    unsigned long sizee = src.size();

    int sze = width * height;

    cout << sze << endl;

    //create pointer to image
    unsigned char *h_src = src.data();

    CImg<unsigned char> dst(width, height, 1, 1);
    unsigned char *h_dst = dst.data();

    unsigned char *d_src;
    unsigned char *d_dst;

    cout << sizee << endl;

    cudaMalloc((void**)&d_src, sizee);
    cudaMalloc((void**)&d_dst, width*height*sizeof(unsigned char));

    cudaMemcpy(d_src, h_src, sizee, cudaMemcpyHostToDevice);

    //launch the kernel
	dim3 blkDim (16, 16, 1);
	dim3 grdDim ((width + 15)/16, (height + 15)/16, 1);
	rgb2gray<<<grdDim, blkDim>>>(d_src, d_dst, width, height);

    //force the printf()s to flush
    cudaDeviceSynchronize();

    // copy back the result array to the CPU
    cudaMemcpy(h_dst, d_dst, width*height, cudaMemcpyDeviceToHost);

    cudaFree(d_src);
    cudaFree(d_dst);

    CImgDisplay main_disp(dst, "After Processing");
    while (!main_disp.is_closed())
        main_disp.wait();

    return 0;
}
