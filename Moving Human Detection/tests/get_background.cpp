#include <iostream>
#include <fstream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <cuda.h>
#include "../include/gmm.h"
#include <filesystem>

// Macro for error-checking
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            std::printf("CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

namespace fs = std::filesystem;

int processFolder(const std::string& folder_path, const std::string& output_background_name) {
    // Load all images from the folder
    std::vector<cv::Mat> frames;
    std::vector<std::string> file_paths;
    cv::glob(folder_path + "/*.bmp", file_paths, false);
    int count = 0;
    for (const auto& file_path : file_paths) {
        if (count % 4 != 0) {
          count+=1;
          continue;
        }
        cv::Mat image = cv::imread(file_path);
        if (!image.empty()) {
            frames.push_back(image);
        } else {
            std::cerr << "Failed to load image: " << file_path << std::endl;
        }
        count+=1;
    }

    if (frames.empty()) {
        std::cerr << "No images loaded in folder " << folder_path << ". Exiting." << std::endl;
        return -1;
    }

    std::cout << "Processing folder: " << folder_path << std::endl;
    std::cout << "Number of frames: " << frames.size() << std::endl;

    int n_frames = frames.size();
    int height = frames[0].rows;
    int width = frames[0].cols;

    // Check for correct image type (CV_8UC3)
    if (frames[0].type() != CV_8UC3) {
        std::cerr << "Expected images of type CV_8UC3 (8-bit, 3 channels). Exiting." << std::endl;
        return -1;
    }

    // Allocate memory for input frames
    float* d_frames;
    CUDA_CHECK(cudaMalloc(&d_frames, n_frames * height * width * 3 * sizeof(float)));
    // cudaMalloc(&d_frames, n_frames * height * width * 3 * sizeof(__half));

    // Allocate a temporary float array for frame data
    std::vector<float> frame_float(height * width * 3);

    // Copy frames to GPU
    for (int i = 0; i < n_frames; i++) {
        cv::Mat frame = frames[i];

        // Convert unsigned char data to float
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                for (int c = 0; c < 3; c++) {
                    frame_float[(h * width + w) * 3 + c] = static_cast<float>(frame.at<cv::Vec3b>(h, w)[c]);
                }
            }
        }

        // Copy the float data to the GPU
        CUDA_CHECK(cudaMemcpy(d_frames + i * height * width * 3, frame_float.data(), height * width * 3 * sizeof(float), cudaMemcpyHostToDevice));
    }

    // Allocate memory for GMM models and background
    GMMModel* d_gmm_models;
    CUDA_CHECK(cudaMalloc(&d_gmm_models, width * height * sizeof(GMMModel)));

    float* d_background;
    CUDA_CHECK(cudaMalloc(&d_background, height * width * 3 * sizeof(float)));

    // Launch GMM fitting kernel
    // Convert __half* to float*
    fitGMM(d_frames, n_frames, width, height, d_gmm_models, -1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());


    // Launch background generation kernel
    generateBackground(d_gmm_models, width, height, d_background);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Download background from GPU
    float* h_background = new float[height * width * 3];
    CUDA_CHECK(cudaMemcpy(h_background, d_background, height * width * 3 * sizeof(float), cudaMemcpyDeviceToHost));

    // Save background as image
    cv::Mat background_image(height, width, CV_32FC3, h_background);
    cv::imwrite(output_background_name, background_image);

    // Free GPU and host memory
    CUDA_CHECK(cudaFree(d_frames));
    CUDA_CHECK(cudaFree(d_gmm_models));
    CUDA_CHECK(cudaFree(d_background));
    delete[] h_background;

    std::cout << "Background saved to " << output_background_name << std::endl;

    return 0;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <root_folder>" << std::endl;
        return -1;
    }

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "Free memory: " << free_mem / (1024 * 1024) << " MB, Total memory: " << total_mem / (1024 * 1024) << " MB" << std::endl;

    std::string root_folder = argv[1];
    std::ofstream timingsFile("timings.csv");
    timingsFile << "Folder,Time Taken (ms)\n";

    for (const auto& entry : fs::directory_iterator(root_folder)) {
        if (entry.is_directory()) {
            std::string folder_path = entry.path().string();
            std::string output_background_name = "background_" + entry.path().filename().string() + ".png";

            auto start_time = std::chrono::high_resolution_clock::now();
            int result = processFolder(folder_path, output_background_name);
            auto end_time = std::chrono::high_resolution_clock::now();

            if (result == 0) {
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
                timingsFile << folder_path << "," << duration << "\n";
                std::cout << "Time taken for " << folder_path << ": " << duration << " ms" << std::endl;
            } else {
                std::cerr << "Failed to process folder: " << folder_path << std::endl;
            }
        }
    }

    timingsFile.close();
    return 0;
}

