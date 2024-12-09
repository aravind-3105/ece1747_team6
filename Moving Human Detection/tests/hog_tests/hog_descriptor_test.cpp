#include "hog_descriptor.h"
#include "hog_descriptor_cpu.h"
#include <iostream>
#include <chrono>
#include <string>

// Function to measure execution time of CPU and GPU HOG descriptor
void testHOGDescriptor(const std::string& inputImagePath, const std::string& cpuOutputPath, const std::string& gpuOutputPath) {
    // Measure CPU version
    auto start_cpu = std::chrono::high_resolution_clock::now();
    computeHOG_cpu(inputImagePath, cpuOutputPath);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_duration = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();

    // // Measure GPU version
    auto start_gpu = std::chrono::high_resolution_clock::now();
    computeHOG(inputImagePath, gpuOutputPath);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    double gpu_duration = std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count();

    // Output results
    std::cout << "HOG Descriptor Test Results:" << std::endl;
    std::cout << "GPU Version Time: " << cpu_duration << " ms" << std::endl;
    std::cout << "CPU Version Time: " << gpu_duration << " ms" << std::endl;

    // std::cout << "Histogram outputs saved to:" << std::endl;
    // std::cout << " - CPU: " << cpuOutputPath << std::endl;
}

int main(int argc, char* argv[]) {
    // Check command-line arguments
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_image_path>" << std::endl;
        return EXIT_FAILURE;
    }

    // Input image path
    std::string inputImagePath = argv[1];

    // Output paths for CPU and GPU results
    std::string cpuOutputPath = "tests/hog_tests/hog_histogram_cpu.csv";
    std::string gpuOutputPath = "tests/hog_tests/hog_histogram_gpu.csv";

    // Run the test
    testHOGDescriptor(inputImagePath, cpuOutputPath, gpuOutputPath);

    return EXIT_SUCCESS;
}
