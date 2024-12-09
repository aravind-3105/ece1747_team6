#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>

#define CUDA_CHECK(call)                                                      \
    {                                                                         \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__      \
                      << " - " << cudaGetErrorString(err) << std::endl;       \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }

__constant__ float d_kernel[5][5];
#define BLOCK_SIZE 16
__global__ void GaussianBlurKernel(float* d_image, float* d_blurred, int width, int height) {
    // Shared memory for the tile
    __shared__ float tile[BLOCK_SIZE + 4][BLOCK_SIZE + 4];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;

    // Load shared memory with appropriate boundary handling
    int sharedX = tx + 2; // +2 for kernel offset
    int sharedY = ty + 2;
    if (x < width && y < height) {
        tile[sharedY][sharedX] = d_image[y * width + x];
    } else {
        tile[sharedY][sharedX] = 0.0f; // Zero-padding for out-of-bound pixels
    }

    // Load additional boundary pixels into shared memory
    if (tx < 2) {
        // Left border
        if (x >= 2) {
            tile[sharedY][tx] = d_image[y * width + (x - 2)];
        } else {
            tile[sharedY][tx] = 0.0f; // Padding
        }

        // Right border
        if (x + blockDim.x < width) {
            tile[sharedY][sharedX + blockDim.x] = d_image[y * width + (x + blockDim.x)];
        } else {
            tile[sharedY][sharedX + blockDim.x] = 0.0f;
        }
    }

    if (ty < 2) {
        // Top border
        if (y >= 2) {
            tile[ty][sharedX] = d_image[(y - 2) * width + x];
        } else {
            tile[ty][sharedX] = 0.0f;
        }

        // Bottom border
        if (y + blockDim.y < height) {
            tile[sharedY + blockDim.y][sharedX] = d_image[(y + blockDim.y) * width + x];
        } else {
            tile[sharedY + blockDim.y][sharedX] = 0.0f;
        }
    }

    // Wait for all threads to finish loading
    __syncthreads();

    // Perform the convolution only for valid pixels
    if (x < width && y < height) {
        float sum = 0.0f;
        for (int ky = 0; ky < 5; ++ky) {
            for (int kx = 0; kx < 5; ++kx) {
                sum += d_kernel[ky][kx] * tile[sharedY + ky - 2][sharedX + kx - 2];
            }
        }
        d_blurred[y * width + x] = sum;
    }
}


#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

// Move Sobel kernels to constant memory
__constant__ int Gx[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

__constant__ int Gy[3][3] = {
    { 1,  2,  1},
    { 0,  0,  0},
    {-1, -2, -1}
};

__global__ void SobelKernel(float* d_blurred, float* d_gradient, float* d_direction, int width, int height) {
    // Calculate global indices
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Calculate indices in shared memory (+1 for halo)
    int s_x = threadIdx.x + 1;
    int s_y = threadIdx.y + 1;

    // Declare shared memory with halo regions
    __shared__ float s_data[BLOCK_SIZE_Y + 2][BLOCK_SIZE_X + 2];

    // Load central data into shared memory
    if (x < width && y < height)
        s_data[s_y][s_x] = d_blurred[y * width + x];
    else
        s_data[s_y][s_x] = 0.0f;

    // Load halo regions
    // Left and right halos
    if (threadIdx.x == 0) {
        int x_left = x - 1;
        s_data[s_y][s_x - 1] = (x_left >= 0 && y < height) ? d_blurred[y * width + x_left] : 0.0f;
    }
    if (threadIdx.x == blockDim.x - 1) {
        int x_right = x + 1;
        s_data[s_y][s_x + 1] = (x_right < width && y < height) ? d_blurred[y * width + x_right] : 0.0f;
    }

    // Top and bottom halos
    if (threadIdx.y == 0) {
        int y_top = y - 1;
        s_data[s_y - 1][s_x] = (x < width && y_top >= 0) ? d_blurred[y_top * width + x] : 0.0f;
    }
    if (threadIdx.y == blockDim.y - 1) {
        int y_bottom = y + 1;
        s_data[s_y + 1][s_x] = (x < width && y_bottom < height) ? d_blurred[y_bottom * width + x] : 0.0f;
    }

    // Corner halos
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        int x_left = x - 1, y_top = y - 1;
        s_data[s_y - 1][s_x - 1] = (x_left >= 0 && y_top >= 0) ? d_blurred[y_top * width + x_left] : 0.0f;
    }
    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == 0) {
        int x_right = x + 1, y_top = y - 1;
        s_data[s_y - 1][s_x + 1] = (x_right < width && y_top >= 0) ? d_blurred[y_top * width + x_right] : 0.0f;
    }
    if (threadIdx.x == 0 && threadIdx.y == blockDim.y - 1) {
        int x_left = x - 1, y_bottom = y + 1;
        s_data[s_y + 1][s_x - 1] = (x_left >= 0 && y_bottom < height) ? d_blurred[y_bottom * width + x_left] : 0.0f;
    }
    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1) {
        int x_right = x + 1, y_bottom = y + 1;
        s_data[s_y + 1][s_x + 1] = (x_right < width && y_bottom < height) ? d_blurred[y_bottom * width + x_right] : 0.0f;
    }

    // Synchronize to ensure all shared memory is loaded
    __syncthreads();

    // Perform convolution if within image bounds
    if (x < width && y < height) {
        float sumX = 0.0f, sumY = 0.0f;

        // Unrolled convolution
        sumX += -1 * s_data[s_y - 1][s_x - 1];
        sumX +=  0 * s_data[s_y - 1][s_x];
        sumX +=  1 * s_data[s_y - 1][s_x + 1];
        sumX += -2 * s_data[s_y][s_x - 1];
        sumX +=  0 * s_data[s_y][s_x];
        sumX +=  2 * s_data[s_y][s_x + 1];
        sumX += -1 * s_data[s_y + 1][s_x - 1];
        sumX +=  0 * s_data[s_y + 1][s_x];
        sumX +=  1 * s_data[s_y + 1][s_x + 1];

        sumY +=  1 * s_data[s_y - 1][s_x - 1];
        sumY +=  2 * s_data[s_y - 1][s_x];
        sumY +=  1 * s_data[s_y - 1][s_x + 1];
        sumY +=  0 * s_data[s_y][s_x - 1];
        sumY +=  0 * s_data[s_y][s_x];
        sumY +=  0 * s_data[s_y][s_x + 1];
        sumY += -1 * s_data[s_y + 1][s_x - 1];
        sumY += -2 * s_data[s_y + 1][s_x];
        sumY += -1 * s_data[s_y + 1][s_x + 1];

        // Compute gradient magnitude and direction
        d_gradient[y * width + x] = sqrtf(sumX * sumX + sumY * sumY);
        d_direction[y * width + x] = atan2f(sumY, sumX);
    }
}


__global__ void NonMaxSuppressionKernel(float* d_gradient, float* d_direction, float* d_edges, int width, int height) {
    // Implement non-maximum suppression logic
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;
    float direction = d_direction[y * width + x];
    float magnitude = d_gradient[y * width + x];
    int dx1 = 0, dy1 = 0, dx2 = 0, dy2 = 0;

    // Map direction to nearest 0, 45, 90, or 135 degrees
    float angle = fmodf(direction + M_PI, M_PI) * (180.0f / M_PI);
    if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180)) {
        dx1 = 1; dy1 = 0; dx2 = -1; dy2 = 0;
    } else if (angle >= 22.5 && angle < 67.5) {
        dx1 = 1; dy1 = 1; dx2 = -1; dy2 = -1;
    } else if (angle >= 67.5 && angle < 112.5) {
        dx1 = 0; dy1 = 1; dx2 = 0; dy2 = -1;
    } else if (angle >= 112.5 && angle < 157.5) {
        dx1 = -1; dy1 = 1; dx2 = 1; dy2 = -1;
    }

    int neighbor1X = min(max(x + dx1, 0), width - 1);
    int neighbor1Y = min(max(y + dy1, 0), height - 1);
    int neighbor2X = min(max(x + dx2, 0), width - 1);
    int neighbor2Y = min(max(y + dy2, 0), height - 1);

    float neighbor1 = d_gradient[neighbor1Y * width + neighbor1X];
    float neighbor2 = d_gradient[neighbor2Y * width + neighbor2X];

    if (magnitude >= neighbor1 && magnitude >= neighbor2) {
        d_edges[y * width + x] = magnitude;
    } else {
        d_edges[y * width + x] = 0.0f;
    }
}

__global__ void DoubleThresholdKernel(float* d_edges, float highThreshold, float lowThreshold, int width, int height) {
    // Implement double thresholding logic
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float value = d_edges[y * width + x];
    if (value >= highThreshold) {
        d_edges[y * width + x] = 1.0f; // Strong edge
    } else if (value >= lowThreshold) {
        d_edges[y * width + x] = 0.5f; // Weak edge
    } else {
        d_edges[y * width + x] = 0.0f; // Non-edge
    }
}

__global__ void HysteresisKernel(float* d_edges, int width, int height) {
    // Implement edge tracking by hysteresis
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    if (d_edges[y * width + x] != 0.5f) return; // Process only weak edges

    bool connectedToStrong = false;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;

            int neighborX = min(max(x + dx, 0), width - 1);
            int neighborY = min(max(y + dy, 0), height - 1);
            if (d_edges[neighborY * width + neighborX] == 1.0f) {
                connectedToStrong = true;
                break;
            }
        }
        if (connectedToStrong) break;
    }

    d_edges[y * width + x] = connectedToStrong ? 1.0f : 0.0f;
}

// Host Function Declarations
void processSingleScale(float* d_image, float* d_output, int width, int height);
void combineEdgeMaps(std::vector<cv::Mat>& edgeMaps, cv::Mat& output);

void processCannySingleImage(const cv::Mat& inputImage, cv::Mat& outputEdges) {
    int width = inputImage.cols;
    int height = inputImage.rows;

    // Copy the Gaussian kernel to constant memory
    const float h_kernel[5][5] = {
        {1 / 273.0f, 4 / 273.0f, 7 / 273.0f, 4 / 273.0f, 1 / 273.0f},
        {4 / 273.0f,16 / 273.0f,26 / 273.0f,16 / 273.0f, 4 / 273.0f},
        {7 / 273.0f,26 / 273.0f,41 / 273.0f,26 / 273.0f, 7 / 273.0f},
        {4 / 273.0f,16 / 273.0f,26 / 273.0f,16 / 273.0f, 4 / 273.0f},
        {1 / 273.0f, 4 / 273.0f, 7 / 273.0f, 4 / 273.0f, 1 / 273.0f}
    };

    // Start timing for kernel copy
    auto kernelCopyStart = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpyToSymbol(d_kernel, h_kernel, sizeof(h_kernel)));
    auto kernelCopyEnd = std::chrono::high_resolution_clock::now();
    auto kernelCopyDuration = std::chrono::duration_cast<std::chrono::milliseconds>(kernelCopyEnd - kernelCopyStart).count();
    std::cout << "Kernel copy to constant memory time: " << kernelCopyDuration << " ms" << std::endl;

    // Allocate device memory
    float *d_image, *d_edges;
    auto deviceAllocStart = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMalloc(&d_image, width * height * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_edges, width * height * sizeof(float)));
    auto deviceAllocEnd = std::chrono::high_resolution_clock::now();
    auto deviceAllocDuration = std::chrono::duration_cast<std::chrono::milliseconds>(deviceAllocEnd - deviceAllocStart).count();
    std::cout << "Device memory allocation time: " << deviceAllocDuration << " ms" << std::endl;

    // Copy image to device
    float* h_image = (float*)malloc(width * height * sizeof(float));
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            h_image[i * width + j] = inputImage.at<uchar>(i, j) / 255.0f;
        }
    }
    auto copyToDeviceStart = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(d_image, h_image, width * height * sizeof(float), cudaMemcpyHostToDevice));
    auto copyToDeviceEnd = std::chrono::high_resolution_clock::now();
    auto copyToDeviceDuration = std::chrono::duration_cast<std::chrono::milliseconds>(copyToDeviceEnd - copyToDeviceStart).count();
    std::cout << "Image copy to device time: " << copyToDeviceDuration << " ms" << std::endl;

    // Process single scale (includes Gaussian smoothing)
    auto processingStart = std::chrono::high_resolution_clock::now();
    processSingleScale(d_image, d_edges, width, height);
    auto processingEnd = std::chrono::high_resolution_clock::now();
    auto processingDuration = std::chrono::duration_cast<std::chrono::microseconds>(processingEnd - processingStart).count();
    std::cout << "Single-scale processing time: " << processingDuration << " µs" << std::endl;

    // Copy result back to host
    float* h_edges = (float*)malloc(width * height * sizeof(float));
    auto copyToHostStart = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(h_edges, d_edges, width * height * sizeof(float), cudaMemcpyDeviceToHost));
    auto copyToHostEnd = std::chrono::high_resolution_clock::now();
    auto copyToHostDuration = std::chrono::duration_cast<std::chrono::milliseconds>(copyToHostEnd - copyToHostStart).count();
    std::cout << "Result copy to host time: " << copyToHostDuration << " ms" << std::endl;

    // Convert result to OpenCV format
    auto edgeConversionStart = std::chrono::high_resolution_clock::now();
    cv::Mat edgeMap(height, width, CV_32F, h_edges);
    edgeMap.convertTo(outputEdges, CV_8U, 255.0);
    auto edgeConversionEnd = std::chrono::high_resolution_clock::now();
    auto edgeConversionDuration = std::chrono::duration_cast<std::chrono::milliseconds>(edgeConversionEnd - edgeConversionStart).count();
    std::cout << "Edge map conversion time: " << edgeConversionDuration << " ms" << std::endl;

    // Free memory
    auto freeMemoryStart = std::chrono::high_resolution_clock::now();
    free(h_image);
    CUDA_CHECK(cudaFree(d_image));
    CUDA_CHECK(cudaFree(d_edges));
    auto freeMemoryEnd = std::chrono::high_resolution_clock::now();
    auto freeMemoryDuration = std::chrono::duration_cast<std::chrono::milliseconds>(freeMemoryEnd - freeMemoryStart).count();
    std::cout << "Memory deallocation time: " << freeMemoryDuration << " ms" << std::endl;
}



// Multi-Scale Canny Implementation
void multiScaleCanny(const cv::Mat& inputImage, std::vector<float> scales, cv::Mat& outputEdges) {
    int originalWidth = inputImage.cols;
    int originalHeight = inputImage.rows;

    std::vector<cv::Mat> edgeMaps;

    // For total time of multiScaleCanny
    auto totalStart = std::chrono::high_resolution_clock::now();
    const float h_kernel[5][5] = {
        {1 / 273.0f, 4 / 273.0f, 7 / 273.0f, 4 / 273.0f, 1 / 273.0f},
        {4 / 273.0f,16 / 273.0f,26 / 273.0f,16 / 273.0f, 4 / 273.0f},
        {7 / 273.0f,26 / 273.0f,41 / 273.0f,26 / 273.0f, 7 / 273.0f},
        {4 / 273.0f,16 / 273.0f,26 / 273.0f,16 / 273.0f, 4 / 273.0f},
        {1 / 273.0f, 4 / 273.0f, 7 / 273.0f, 4 / 273.0f, 1 / 273.0f}
    };

    // Copy the Gaussian kernel to constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(d_kernel, h_kernel, sizeof(h_kernel)));

    for (float scale : scales) {
        // Start timer for this scale
        auto scaleStart = std::chrono::high_resolution_clock::now();

        // Resize the image for the current scale
        auto resizeStart = std::chrono::high_resolution_clock::now();
        int scaledWidth = static_cast<int>(originalWidth * scale);
        int scaledHeight = static_cast<int>(originalHeight * scale);
        cv::Mat resizedImage;
        cv::resize(inputImage, resizedImage, cv::Size(scaledWidth, scaledHeight), 0, 0, cv::INTER_LINEAR);
        auto resizeEnd = std::chrono::high_resolution_clock::now();
        auto resizeDuration = std::chrono::duration_cast<std::chrono::milliseconds>(resizeEnd - resizeStart).count();

        // Copy the resized image to device memory
        auto hostAllocStart = std::chrono::high_resolution_clock::now();
        float* h_resizedImage = (float*)malloc(scaledWidth * scaledHeight * sizeof(float));
        for (int i = 0; i < resizedImage.rows; ++i) {
            for (int j = 0; j < resizedImage.cols; ++j) {
                h_resizedImage[i * scaledWidth + j] = resizedImage.at<uchar>(i, j) / 255.0f;
            }
        }
        auto hostAllocEnd = std::chrono::high_resolution_clock::now();
        auto hostAllocDuration = std::chrono::duration_cast<std::chrono::milliseconds>(hostAllocEnd - hostAllocStart).count();

        // Device memory allocation
        auto deviceAllocStart = std::chrono::high_resolution_clock::now();
        float *d_image, *d_edges;
        CUDA_CHECK(cudaMalloc(&d_image, scaledWidth * scaledHeight * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_edges, scaledWidth * scaledHeight * sizeof(float)));
        auto deviceAllocEnd = std::chrono::high_resolution_clock::now();
        auto deviceAllocDuration = std::chrono::duration_cast<std::chrono::milliseconds>(deviceAllocEnd - deviceAllocStart).count();

        // Copy to device
        auto copyToDeviceStart = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaMemcpy(d_image, h_resizedImage, scaledWidth * scaledHeight * sizeof(float), cudaMemcpyHostToDevice));
        auto copyToDeviceEnd = std::chrono::high_resolution_clock::now();
        auto copyToDeviceDuration = std::chrono::duration_cast<std::chrono::milliseconds>(copyToDeviceEnd - copyToDeviceStart).count();
        float transferTimeSec = copyToDeviceDuration / 1000.0f;
        float bandwidth = (scaledWidth * scaledHeight * sizeof(float)) / (1024.0 * 1024.0 * 1024.0) / transferTimeSec;
        std::cout << "  Copy to device bandwidth: " << bandwidth << " GB/s" << std::endl;

        // Process single scale
        auto processStart = std::chrono::high_resolution_clock::now();
        processSingleScale(d_image, d_edges, scaledWidth, scaledHeight);
        auto processEnd = std::chrono::high_resolution_clock::now();
        auto processDuration = std::chrono::duration_cast<std::chrono::milliseconds>(processEnd - processStart).count();

        // Copy the edge map back to host memory
        auto copyToHostStart = std::chrono::high_resolution_clock::now();
        float* h_edges = (float*)malloc(scaledWidth * scaledHeight * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h_edges, d_edges, scaledWidth * scaledHeight * sizeof(float), cudaMemcpyDeviceToHost));
        auto copyToHostEnd = std::chrono::high_resolution_clock::now();
        auto copyToHostDuration = std::chrono::duration_cast<std::chrono::milliseconds>(copyToHostEnd - copyToHostStart).count();
        transferTimeSec = copyToHostDuration / 1000.0f;
        bandwidth = (scaledWidth * scaledHeight * sizeof(float)) / (1024.0 * 1024.0 * 1024.0) / transferTimeSec;
        std::cout << "  Copy to host bandwidth: " << bandwidth << " GB/s" << std::endl;

        // Resize edge map back to original size
        auto edgeResizeStart = std::chrono::high_resolution_clock::now();
        cv::Mat edgeMap(scaledHeight, scaledWidth, CV_32F, h_edges);
        cv::resize(edgeMap, edgeMap, cv::Size(originalWidth, originalHeight), 0, 0, cv::INTER_LINEAR);
        edgeMaps.push_back(edgeMap);
        auto edgeResizeEnd = std::chrono::high_resolution_clock::now();
        auto edgeResizeDuration = std::chrono::duration_cast<std::chrono::milliseconds>(edgeResizeEnd - edgeResizeStart).count();

        // Free memory
        auto freeStart = std::chrono::high_resolution_clock::now();
        free(h_resizedImage);
        // Do not free h_edges here; it is managed by cv::Mat
        CUDA_CHECK(cudaFree(d_image));
        CUDA_CHECK(cudaFree(d_edges));
        auto freeEnd = std::chrono::high_resolution_clock::now();
        auto freeDuration = std::chrono::duration_cast<std::chrono::milliseconds>(freeEnd - freeStart).count();

        auto scaleEnd = std::chrono::high_resolution_clock::now();
        auto scaleDuration = std::chrono::duration_cast<std::chrono::milliseconds>(scaleEnd - scaleStart).count();

        // Print timings for this scale
        std::cout << "Scale: " << scale << " - Time: " << scaleDuration << " ms" << std::endl;
        std::cout << "  Resize image: " << resizeDuration << " ms" << std::endl;
        std::cout << "  Host allocation and copy: " << hostAllocDuration << " ms" << std::endl;
        std::cout << "  Device memory allocation: " << deviceAllocDuration << " ms" << std::endl;
        std::cout << "  Copy to device: " << copyToDeviceDuration << " ms" << std::endl;
        std::cout << "  Processing (kernels): " << processDuration << " ms" << std::endl;
        std::cout << "  Copy to host: " << copyToHostDuration << " ms" << std::endl;
        std::cout << "  Resize edge map: " << edgeResizeDuration << " ms" << std::endl;
        std::cout << "  Free memory: " << freeDuration << " ms" << std::endl;
    }

    // Combine edge maps from all scales
    auto combineStart = std::chrono::high_resolution_clock::now();
    combineEdgeMaps(edgeMaps, outputEdges);
    auto combineEnd = std::chrono::high_resolution_clock::now();
    auto combineDuration = std::chrono::duration_cast<std::chrono::milliseconds>(combineEnd - combineStart).count();

    auto totalEnd = std::chrono::high_resolution_clock::now();
    auto totalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(totalEnd - totalStart).count();

    std::cout << "Total multiScaleCanny time: " << totalDuration << " ms" << std::endl;
    std::cout << "  Combine edge maps: " << combineDuration << " ms" << std::endl;
}

void processSingleScale(float* d_image, float* d_output, int width, int height) {
    float *d_blurred, *d_gradient, *d_direction;
    CUDA_CHECK(cudaMalloc(&d_blurred, width * height * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gradient, width * height * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_direction, width * height * sizeof(float)));

    // Gaussian Blur
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Create CUDA events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float milliseconds = 0;

    // Gaussian Blur
    int minGridSize, blockSize1;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize1, GaussianBlurKernel, 0, 0));
    std::cout << "    GaussianBlurKernel occupancy: " << blockSize1 
            << " threads per block (min grid size: " << minGridSize << ")" << std::endl;
    CUDA_CHECK(cudaEventRecord(start));
    GaussianBlurKernel<<<gridSize, blockSize>>>(d_image, d_blurred, width, height);
    cudaFuncAttributes attributes;
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, GaussianBlurKernel));
    std::cout << "    GaussianBlurKernel shared memory usage: " 
            << attributes.sharedSizeBytes << " bytes" << std::endl;
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "    GaussianBlurKernel time: " << milliseconds << " ms" << std::endl;

    // Gradient Computation
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize1, SobelKernel, 0, 0));
    std::cout << "    SobelKernel occupancy: " << blockSize1 
            << " threads per block (min grid size: " << minGridSize << ")" << std::endl;
    CUDA_CHECK(cudaEventRecord(start));
    SobelKernel<<<gridSize, blockSize>>>(d_blurred, d_gradient, d_direction, width, height);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "    SobelKernel time: " << milliseconds << " ms" << std::endl;
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, SobelKernel));
    std::cout << "    SobelKernel shared memory usage: " 
          << attributes.sharedSizeBytes << " bytes" << std::endl;
    // Non-Maximum Suppression
    CUDA_CHECK(cudaEventRecord(start));
    NonMaxSuppressionKernel<<<gridSize, blockSize>>>(d_gradient, d_direction, d_output, width, height);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "    NonMaxSuppressionKernel time: " << milliseconds << " ms" << std::endl;

    // Double Thresholding
    CUDA_CHECK(cudaEventRecord(start));
    DoubleThresholdKernel<<<gridSize, blockSize>>>(d_output, 0.2f, 0.1f, width, height);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "    DoubleThresholdKernel time: " << milliseconds << " ms" << std::endl;

    // Edge Tracking by Hysteresis
    CUDA_CHECK(cudaEventRecord(start));
    HysteresisKernel<<<gridSize, blockSize>>>(d_output, width, height);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "    HysteresisKernel time: " << milliseconds << " ms" << std::endl;

    // Free intermediate buffers
    CUDA_CHECK(cudaFree(d_blurred));
    CUDA_CHECK(cudaFree(d_gradient));
    CUDA_CHECK(cudaFree(d_direction));

    // Destroy CUDA events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

void combineEdgeMaps(std::vector<cv::Mat>& edgeMaps, cv::Mat& output) {
    output = cv::Mat::zeros(edgeMaps[0].size(), CV_32F);

    for (const cv::Mat& edgeMap : edgeMaps) {
        cv::max(output, edgeMap, output);
    }

    // Convert back to 8-bit for display
    output.convertTo(output, CV_8U, 255.0);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <image_file> <mode> [scales]" << std::endl;
        std::cerr << "  <mode>: 'single' or 'multi'" << std::endl;
        std::cerr << "  [scales]: Comma-separated list of scales for 'multi' mode (e.g., 0.5,1.0,2.0)" << std::endl;
        return -1;
    }
    cudaFree(0);
    const char* inputImageFile = argv[1];
    std::string mode = argv[2];

    // Load input image
    cv::Mat inputImage = cv::imread(inputImageFile, cv::IMREAD_GRAYSCALE);
    if (inputImage.empty()) {
        std::cerr << "Error: Could not load image." << std::endl;
        return -1;
    }

    cv::Mat outputEdges;

    if (mode == "single") {
        // Perform Canny edge detection without multi-scale Gaussian
        std::cout << "Running single-scale Canny edge detection..." << std::endl;
        processCannySingleImage(inputImage, outputEdges);
    } else if (mode == "multi") {
        // Parse scales from command line (default: {0.5, 1.0, 2.0})
        std::vector<float> scales = {0.5f, 1.0f, 2.0f};
        if (argc == 4) {
            scales.clear();
            std::string scaleArg = argv[3];
            size_t pos = 0;
            while ((pos = scaleArg.find(',')) != std::string::npos) {
                scales.push_back(std::stof(scaleArg.substr(0, pos)));
                scaleArg.erase(0, pos + 1);
            }
            scales.push_back(std::stof(scaleArg)); // Add the last scale
        }

        std::cout << "Running multi-scale Canny edge detection with scales: ";
        for (float scale : scales) std::cout << scale << " ";
        std::cout << std::endl;

        multiScaleCanny(inputImage, scales, outputEdges);
    } else {
        std::cerr << "Error: Unknown mode '" << mode << "'. Use 'single' or 'multi'." << std::endl;
        return -1;
    }

    // Save and display the output
    cv::imwrite("canny_edges.png", outputEdges);
    // cv::imshow("Canny Edges", outputEdges);
    cv::waitKey(0);

    return 0;
}