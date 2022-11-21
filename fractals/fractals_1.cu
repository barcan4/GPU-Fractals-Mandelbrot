#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "util.h"

__global__ void set_image(unsigned char* image, unsigned char* colormap) {

	int i = threadIdx.y + blockIdx.y * blockDim.y; // rand
	int j = threadIdx.x + blockIdx.x * blockDim.x; // coloana
	int max = MAX_ITERATION;

	if (j >= WIDTH || i >= HEIGHT) {
		return;
	}

	double c_re = (j - WIDTH / 2.0) * 4.0 / WIDTH;
	double c_im = (i - HEIGHT / 2.0) * 4.0 / WIDTH;
	double x = 0, y = 0, x_new;
	int iteration = 0;

	while (x * x + y * y <= 4.0 && iteration < max) {
		x_new = x * x - y * y + c_re;
		y = 2.0 * x * y + c_im;
		x = x_new;
		iteration++;
	}
	if (iteration > max) {
		iteration = max;
	}

	image[4 * i * WIDTH + 4 * j + 0] = colormap[iteration * 3 + 0];
	image[4 * i * WIDTH + 4 * j + 1] = colormap[iteration * 3 + 1];
	image[4 * i * WIDTH + 4 * j + 2] = colormap[iteration * 3 + 2];
	image[4 * i * WIDTH + 4 * j + 3] = 255;
}


cudaError_t generate_image(unsigned char* image, unsigned char* colormap) {
	unsigned char* dev_image;
	unsigned char* dev_colormap;
	//unsigned char* dev_new_image;
	//unsigned int* dev_width;
	//unsigned int* dev_height;
	//unsigned int* dev_max_iterations;
	cudaError_t cudaStatus;

	dim3 block(2, 2);
	dim3 grid(WIDTH / block.x, HEIGHT / block.y);

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_image, (WIDTH * HEIGHT * 4));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_colormap, ((MAX_ITERATION + 1) * 3));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_image, image, (WIDTH * HEIGHT * 4), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_colormap, colormap, ((MAX_ITERATION + 1) * 3), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	set_image << <grid, block >> > (dev_image, dev_colormap);



	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching set_image!\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaMemcpy(image, dev_image, (WIDTH * HEIGHT * 4), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


Error:
	//cudaFree(dev_max_iterations);
	//cudaFree(dev_height);
	//cudaFree(dev_width);
	cudaFree(dev_colormap);
	cudaFree(dev_image);

	return cudaStatus;
}

int main() {
	double times[REPEAT];
	struct timeb start_time, end;
	int r;
	char path[255];
	cudaError_t cudaStatus;

	unsigned char* colormap = (unsigned char*)malloc((MAX_ITERATION + 1) * 3);
	unsigned char* image = (unsigned char*)malloc(WIDTH * HEIGHT * 4);

	init_colormap(MAX_ITERATION, colormap);

	for (r = 0; r < REPEAT; r++) {
		memset(image, 0, WIDTH * HEIGHT + 4);

		ftime(&start_time);

		//start GPU implementation
		cudaStatus = generate_image(image, colormap);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addWithCuda failed!");
			return 1;
		}

		//end GPU implementation
		ftime(&end);
		times[r] = end.time - start_time.time + ((double)end.millitm - (double)start_time.millitm) / 1000.0;

		sprintf(path, IMAGE, "gpu", r);
		save_image(path, image, WIDTH, HEIGHT);
		progress("gpu", r, times[r]);
	}
	report("gpu", times);

	free(image);
	free(colormap);

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}