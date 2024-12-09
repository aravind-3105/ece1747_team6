#include <cuda_runtime.h>
#include <cstdio>

#define TILE_WIDTH 16
#define TILE_HEIGHT 16

// CUDA kernel for warping frames based on motion vectors using bilinear interpolation
__global__ void warpFrameKernel(const uchar3* input, uchar3* output, int width, int height, float motionX, float motionY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float srcX = x + motionX;
        float srcY = y + motionY;

        if (srcX >= 0.0f && srcX < static_cast<float>(width - 1) && srcY >= 0.0f && srcY < static_cast<float>(height - 1)) {
            int x0 = static_cast<int>(floorf(srcX));
            int y0 = static_cast<int>(floorf(srcY));
            float fracX = srcX - static_cast<float>(x0);
            float fracY = srcY - static_cast<float>(y0);

            uchar3 c00 = input[y0 * width + x0];
            uchar3 c10 = input[y0 * width + (x0 + 1)];
            uchar3 c01 = input[(y0 + 1) * width + x0];
            uchar3 c11 = input[(y0 + 1) * width + (x0 + 1)];

            uchar3 interpolated;
            interpolated.x = static_cast<unsigned char>(
                (1.0f - fracX) * (1.0f - fracY) * c00.x +
                fracX * (1.0f - fracY) * c10.x +
                (1.0f - fracX) * fracY * c01.x +
                fracX * fracY * c11.x
            );
            interpolated.y = static_cast<unsigned char>(
                (1.0f - fracX) * (1.0f - fracY) * c00.y +
                fracX * (1.0f - fracY) * c10.y +
                (1.0f - fracX) * fracY * c01.y +
                fracX * fracY * c11.y
            );
            interpolated.z = static_cast<unsigned char>(
                (1.0f - fracX) * (1.0f - fracY) * c00.z +
                fracX * (1.0f - fracY) * c10.z +
                (1.0f - fracX) * fracY * c01.z +
                fracX * fracY * c11.z
            );

            output[y * width + x] = interpolated;
        } else {
            output[y * width + x] = make_uchar3(0, 0, 0);
        }
    }
}

// Kernel launcher function
void warpFrameKernelLauncher(const uchar3* input, uchar3* output, int width, int height, float motionX, float motionY, cudaStream_t stream) {
    dim3 blockSize(TILE_WIDTH, TILE_HEIGHT);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Timing variables
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timing
    cudaEventRecord(start, stream);

    // Launch the kernel
    warpFrameKernel<<<gridSize, blockSize, 0, stream>>>(input, output, width, height, motionX, motionY);

    // Stop timing
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0.0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %.3f ms\n", milliseconds);

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch warpFrameKernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
