
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

// Constants
#define BLOCK_SIZE 16
#define COEFF 0.5f             // Smaller coefficient for sharper edge differentiation
#define THRESHOLD_VALUE 0.05f  // Adaptive thresholding used below

// CUDA kernel for horizontal Gaussian blur
__global__ void gaussianBlurHorizontal(const float* input, float* output, int width, int height, const float* kernel, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float sum = 0.0f;
    int halfKernel = kernelSize / 2;
    for (int i = -halfKernel; i <= halfKernel; ++i) {
        int neighborX = min(max(x + i, 0), width - 1);
        sum += input[y * width + neighborX] * kernel[halfKernel + i];
    }
    output[y * width + x] = sum;
}

// CUDA kernel for vertical Gaussian blur
__global__ void gaussianBlurVertical(const float* input, float* output, int width, int height, const float* kernel, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float sum = 0.0f;
    int halfKernel = kernelSize / 2;
    for (int i = -halfKernel; i <= halfKernel; ++i) {
        int neighborY = min(max(y + i, 0), height - 1);
        sum += input[neighborY * width + x] * kernel[halfKernel + i];
    }
    output[y * width + x] = sum;
}

// CUDA kernel for computing illustration differences
__global__ void computeIllustration(const float* original, const float* blurred, float* output, int width, int height, float coeff) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float diff = (original[idx] - blurred[idx]) / (coeff + original[idx]);
    output[idx] += diff;
}

// CUDA kernel for smooth thresholding
__global__ void applySmoothThreshold(const float* input, float* output, int width, int height, float threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float value = input[idx];
    // Sigmoid function for smooth thresholding
    output[idx] = 1.0f / (1.0f + expf(-10.0f * (value - threshold)));
}

// Generate Gaussian kernel
void generateGaussianKernel(std::vector<float>& kernel, int size, float alpha) {
    int halfSize = size / 2;
    float sum = 0.0f;
    for (int i = -halfSize; i <= halfSize; ++i) {
        float value = expf(-(i * i) / (2.0f * alpha * alpha));
        kernel[halfSize + i] = value;
        sum += value;
    }
    for (float& v : kernel) v /= sum;
}

// Function to process a single frame
void processFrame(const float* src, float* dest, int width, int height, int levels, int frameCount, float adaptiveThreshold) {
    size_t imageSize = width * height * sizeof(float);

    // Device memory allocation
    float *d_original, *d_blurred, *d_temp, *d_output, *d_kernel;
    cudaMalloc(&d_original, imageSize);
    cudaMalloc(&d_blurred, imageSize);
    cudaMalloc(&d_temp, imageSize);
    cudaMalloc(&d_output, imageSize);
    cudaMemcpy(d_original, src, imageSize, cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, imageSize);

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    for (int s = 1; s <= levels; ++s) {
        float alpha = pow(1.6f, s);
        int kernelSize = ceil(3 * alpha) * 2 + 1;

        // Generate Gaussian kernel
        std::vector<float> h_kernel(kernelSize);
        generateGaussianKernel(h_kernel, kernelSize, alpha);
        cudaMalloc(&d_kernel, kernelSize * sizeof(float));
        cudaMemcpy(d_kernel, h_kernel.data(), kernelSize * sizeof(float), cudaMemcpyHostToDevice);

        // Apply horizontal and vertical Gaussian blur
        gaussianBlurHorizontal<<<blocks, threads>>>(d_original, d_temp, width, height, d_kernel, kernelSize);
        cudaDeviceSynchronize();
        gaussianBlurVertical<<<blocks, threads>>>(d_temp, d_blurred, width, height, d_kernel, kernelSize);
        cudaDeviceSynchronize();

        // Save intermediate blurred frame for debugging
        if (s == 1) {
            std::vector<float> blurredFrame(width * height);
            cudaMemcpy(blurredFrame.data(), d_blurred, imageSize, cudaMemcpyDeviceToHost);
            cv::Mat intermediateBlur(height, width, CV_32F, blurredFrame.data());
            intermediateBlur.convertTo(intermediateBlur, CV_8U, 255.0);
            //cv::imwrite("debug_blurred_frame_" + std::to_string(frameCount) + ".png", intermediateBlur);
        }

        // Compute illustration differences
        computeIllustration<<<blocks, threads>>>(d_original, d_blurred, d_output, width, height, COEFF);
        cudaDeviceSynchronize();

        // Swap the blurred image for the next iteration
        cudaMemcpy(d_original, d_blurred, imageSize, cudaMemcpyDeviceToDevice);
        cudaFree(d_kernel);
    }

    // Apply smooth thresholding
    applySmoothThreshold<<<blocks, threads>>>(d_output, d_blurred, width, height, adaptiveThreshold);
    cudaMemcpy(dest, d_blurred, imageSize, cudaMemcpyDeviceToHost);

    // Save final processed frame for debugging
    cv::Mat finalFrame(height, width, CV_32F, dest);
    finalFrame.convertTo(finalFrame, CV_8U, 255.0);
    //cv::imwrite("debug_final_frame_" + std::to_string(frameCount) + ".png", finalFrame);

    // Free device memory
    cudaFree(d_original);
    cudaFree(d_blurred);
    cudaFree(d_temp);
    cudaFree(d_output);
}

int main() {
    // Input video file from Python Script present in Google Colab
    std::string inputFile = "test.mp4"; 

    // OpenCV video capture
    cv::VideoCapture cap(inputFile);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open video file: " << inputFile << std::endl;
        return -1;
    }

    // Video properties
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));

    // Output video writer
    cv::VideoWriter writer("output_video.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(width, height), false);

    if (!writer.isOpened()) {
        std::cerr << "Error: Cannot open video writer!" << std::endl;
        return -1;
    }

    cv::Mat frame, grayFrame;
    std::vector<float> src, dest(width * height, 0.0f);
    int frameCount = 0;

    while (true) {
        // Capture a frame
        cap >> frame;
        if (frame.empty()) break;  // Exit if no more frames

        frameCount++;

        // Convert to grayscale and normalize
        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
        grayFrame.convertTo(grayFrame, CV_32F, 1.0 / 255.0);

        // Verify normalization
        double minVal, maxVal;
        cv::minMaxLoc(grayFrame, &minVal, &maxVal);
        //std::cout << "Input Frame " << frameCount << " range: [" << minVal << ", " << maxVal << "]" << std::endl;

        // Save normalized input frame for debugging
        // cv::imwrite("debug_input_frame_" + std::to_string(frameCount) + ".png", grayFrame);

        // Flatten the frame for CUDA processing
        src.assign((float*)grayFrame.datastart, (float*)grayFrame.dataend);

        // Calculate adaptive threshold based on mean intensity
        float adaptiveThreshold = cv::mean(grayFrame)[0] * 0.3f;

        // Process the frame with CUDA
        processFrame(src.data(), dest.data(), width, height, 7, frameCount, adaptiveThreshold);

        // Reshape and scale back to 8-bit
        cv::Mat outputFrame(height, width, CV_32F, dest.data());
        outputFrame.convertTo(outputFrame, CV_8U, 255.0);

        // Write to video file
        writer.write(outputFrame);

        // Display the processed frame
        // cv::imshow("Illustration Effect", outputFrame);
        // if (cv::waitKey(1) == 27) break;  // Exit on 'Esc' key
    }

    cap.release();
    writer.release();
    cv::destroyAllWindows();

    std::cout << "Video processing complete!" << std::endl;
    return 0;
}
