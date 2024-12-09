// #include "gmm.h"
#include "gmm_cpu.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>

// Function to load frames from a folder
std::vector<float> loadFramesFromFolder(const std::string& folderPath, int& width, int& height, int& n_frames) {
    std::vector<std::string> filePaths;

    // Get all .bmp files in the folder
    for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
        if (entry.path().extension() == ".bmp") {
            filePaths.push_back(entry.path().string());
        }
    }

    // Sort the file paths to ensure consistent ordering
    std::sort(filePaths.begin(), filePaths.end());

    n_frames = filePaths.size();
    if (n_frames == 0) {
        throw std::runtime_error("No .bmp files found in the specified folder.");
    }

    // Load the first image to get dimensions
    cv::Mat firstImage = cv::imread(filePaths[0]);
    if (firstImage.empty()) {
        throw std::runtime_error("Could not load the first image.");
    }

    width = firstImage.cols;
    height = firstImage.rows;

    // Create a vector to store all frames
    std::vector<float> frames(n_frames * width * height * 3);

    for (int i = 0; i < n_frames; ++i) {
        cv::Mat image = cv::imread(filePaths[i]);
        if (image.empty()) {
            throw std::runtime_error("Could not load image: " + filePaths[i]);
        }

        // Ensure the image is in RGB format
        cv::Mat rgbImage;
        cv::cvtColor(image, rgbImage, cv::COLOR_BGR2RGB);

        // Copy pixel data into the frames vector
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                cv::Vec3b pixel = rgbImage.at<cv::Vec3b>(y, x);
                frames[(i * height * width + y * width + x) * 3 + 0] = static_cast<float>(pixel[0]);
                frames[(i * height * width + y * width + x) * 3 + 1] = static_cast<float>(pixel[1]);
                frames[(i * height * width + y * width + x) * 3 + 2] = static_cast<float>(pixel[2]);
            }
        }
    }

    return frames;
}

// Function to test GMM on CPU (GPU already obtained)
void testGMM(const std::string& folderPath) {
    int width, height, n_frames;

    // Load frames from the folder
    std::vector<float> frames = loadFramesFromFolder(folderPath, width, height, n_frames);
    std::vector<GMMModelCPU> gmm_models_cpu(width * height);

    // CPU Test
    auto start_cpu = std::chrono::high_resolution_clock::now();
    fitGMM_CPU(frames.data(), n_frames, width, height, gmm_models_cpu.data());
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_duration = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();

    // Output Results
    std::cout << "GMM Test Results:" << std::endl;
    std::cout << "CPU Version Time: " << cpu_duration << " ms" << std::endl;
    // std::cout << "GPU Version Time: " << gpu_duration << " ms" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_folder_path>" << std::endl;
        return EXIT_FAILURE;
    }

    // Input folder path
    std::string folderPath = argv[1];

    try {
        // Run the GMM test
        testGMM(folderPath);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

