#include "hog_descriptor.h"
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

#define HOG_NUM_BINS 16

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ \
                      << ": " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// CUDA kernel to compute gradients (Gx, Gy) and gradient magnitude and orientation
__global__ void computeGradientKernel(const float* d_image, float* d_gx, float* d_gy, float* d_magnitude, float* d_orientation, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    float gx = 0.0f;
    float gy = 0.0f;

    if (x > 0 && x < width - 1) {
        gx = d_image[y * width + (x + 1)] - d_image[y * width + (x - 1)];
    }
    if (y > 0 && y < height - 1) {
        gy = d_image[(y + 1) * width + x] - d_image[(y - 1) * width + x];
    }

    d_gx[idx] = gx;
    d_gy[idx] = gy;
    d_magnitude[idx] = sqrtf(gx * gx + gy * gy);
    d_orientation[idx] = atan2f(gy, gx) * 180.0f / M_PI;

    if (d_orientation[idx] < 0) {
        d_orientation[idx] += 180.0f;
    }
}

// CUDA kernel to compute HOG histograms for each cell
// __global__ void computeHOGKernel(const float* d_magnitude, const float* d_orientation, float* d_histogram, int width, int height, int cell_size, int num_bins) {
//     int cell_x = blockIdx.x * blockDim.x + threadIdx.x;
//     int cell_y = blockIdx.y * blockDim.y + threadIdx.y;

//     if (cell_x >= width / cell_size || cell_y >= height / cell_size) return;

//     int histogram_idx = (cell_y * (width / cell_size) + cell_x) * num_bins;

//     for (int bin = 0; bin < num_bins; ++bin) {
//         d_histogram[histogram_idx + bin] = 0.0f;
//     }

//     for (int y = 0; y < cell_size; ++y) {
//         for (int x = 0; x < cell_size; ++x) {
//             int pixel_x = cell_x * cell_size + x;
//             int pixel_y = cell_y * cell_size + y;

//             if (pixel_x >= width || pixel_y >= height) continue;

//             int pixel_idx = pixel_y * width + pixel_x;
//             float magnitude = d_magnitude[pixel_idx];
//             float orientation = d_orientation[pixel_idx];

//             int bin = static_cast<int>(orientation / (180.0f / num_bins)) % num_bins;
//             d_histogram[histogram_idx + bin] += magnitude;
//         }
//     }
// }
__global__ void computeHOGKernel(const float* d_magnitude, const float* d_orientation, float* d_histogram, 
                                       int width, int height, int cell_size, int num_bins) {
    __shared__ float shared_histogram[HOG_NUM_BINS];

    int cell_x = blockIdx.x * blockDim.x + threadIdx.x;
    int cell_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (threadIdx.x < HOG_NUM_BINS) {
        shared_histogram[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    if (cell_x < width / cell_size && cell_y < height / cell_size) {
        int histogram_idx = (cell_y * (width / cell_size) + cell_x) * num_bins;

        for (int y = 0; y < cell_size; ++y) {
            for (int x = 0; x < cell_size; ++x) {
                int pixel_x = cell_x * cell_size + x;
                int pixel_y = cell_y * cell_size + y;

                if (pixel_x >= width || pixel_y >= height) continue;

                int pixel_idx = pixel_y * width + pixel_x;
                float magnitude = d_magnitude[pixel_idx];
                float orientation = d_orientation[pixel_idx];

                int bin = static_cast<int>(orientation / (180.0f / num_bins)) % num_bins;
                atomicAdd(&shared_histogram[bin], magnitude);
            }
        }
        __syncthreads();

        // Write shared histogram to global memory
        if (threadIdx.x == 0) {
            for (int bin = 0; bin < num_bins; ++bin) {
                atomicAdd(&d_histogram[histogram_idx + bin], shared_histogram[bin]);
            }
        }
    }
}


void computeHOG(const std::string& inputImagePath, const std::string& outputHistogramPath) {
    cv::Mat inputImage = cv::imread(inputImagePath, cv::IMREAD_GRAYSCALE);
    if (inputImage.empty()) {
        std::cerr << "Error: Could not load " << inputImagePath << ". Check the file path." << std::endl;
        return;
    }

    cv::resize(inputImage, inputImage, cv::Size(64, 128), cv::INTER_LINEAR);

    int width = inputImage.cols;
    int height = inputImage.rows;

    cv::Mat floatImage;
    inputImage.convertTo(floatImage, CV_32F);

    const int cell_size = 8;
    const int num_bins = 9;

    // Allocate device memory
    float *d_image, *d_gx, *d_gy, *d_magnitude, *d_orientation, *d_histogram;
    CUDA_CHECK(cudaMalloc(&d_image, width * height * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gx, width * height * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gy, width * height * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_magnitude, width * height * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_orientation, width * height * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_histogram, (width / cell_size) * (height / cell_size) * num_bins * sizeof(float)));

    // Initialize histogram memory
    CUDA_CHECK(cudaMemset(d_histogram, 0, (width / cell_size) * (height / cell_size) * num_bins * sizeof(float)));

    // Copy input image to device
    CUDA_CHECK(cudaMemcpy(d_image, floatImage.ptr<float>(), width * height * sizeof(float), cudaMemcpyHostToDevice));

    // Launch gradient computation kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    computeGradientKernel<<<gridSize, blockSize>>>(d_image, d_gx, d_gy, d_magnitude, d_orientation, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Launch HOG kernel
    dim3 hogBlockSize(8, 8);
    dim3 hogGridSize((width / cell_size + hogBlockSize.x - 1) / hogBlockSize.x, (height / cell_size + hogBlockSize.y - 1) / hogBlockSize.y);
    computeHOGKernel<<<hogGridSize, hogBlockSize>>>(d_magnitude, d_orientation, d_histogram, width, height, cell_size, num_bins);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy histogram data back to host
    std::vector<float> h_histogram((width / cell_size) * (height / cell_size) * num_bins);
    CUDA_CHECK(cudaMemcpy(h_histogram.data(), d_histogram, h_histogram.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // Normalize histogram
    float max_val = *std::max_element(h_histogram.begin(), h_histogram.end());
    for (auto& val : h_histogram) {
        val /= max_val;
    }

    // Save to CSV
    // std::ofstream outFile(outputHistogramPath);
    // for (size_t i = 0; i < h_histogram.size(); ++i) {
    //     outFile << h_histogram[i];
    //     if ((i + 1) % num_bins == 0) {
    //         outFile << "\n";
    //     } else {
    //         outFile << ",";
    //     }
    // }
    // outFile.close();

    cudaFree(d_image);
    cudaFree(d_gx);
    cudaFree(d_gy);
    cudaFree(d_magnitude);
    cudaFree(d_orientation);
    cudaFree(d_histogram);

    // std::cout << "HOG histograms saved to " << outputHistogramPath << "." << std::endl;
}