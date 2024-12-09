#include "fourier_descriptor.h"
#include "fourier_descriptor_cpu.h"
#include <iostream>
#include <chrono>
#include <string>
#include <vector>
#include <fstream>

// Function to measure execution time of CPU and GPU Fourier descriptors
void testFourierDescriptor(const std::string& inputImagePath, const std::string& cpuOutputPath, const std::string& gpuOutputPath) {
    // Measure CPU version
    auto start_cpu = std::chrono::high_resolution_clock::now();
    std::vector<float> cpu_descriptor = computeFourierDescriptor_pathCPU(inputImagePath);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_duration = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();

    // Save CPU descriptor to file
    std::ofstream cpuOut(cpuOutputPath);
    for (const auto& val : cpu_descriptor) {
        cpuOut << val << "\n";
    }
    cpuOut.close();

    // Measure GPU version
    auto start_gpu = std::chrono::high_resolution_clock::now();
    std::vector<float> gpu_descriptor = computeFourierDescriptor_path(inputImagePath);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    double gpu_duration = std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count();

    // Save GPU descriptor to file
    std::ofstream gpuOut(gpuOutputPath);
    for (const auto& val : gpu_descriptor) {
        gpuOut << val << "\n";
    }
    gpuOut.close();

    // Output results
    std::cout << "Fourier Descriptor Test Results:" << std::endl;
    std::cout << "CPU Version Time: " << cpu_duration << " ms" << std::endl;
    std::cout << "GPU Version Time: " << gpu_duration << " ms" << std::endl;

    // std::cout << "Descriptor outputs saved to:" << std::endl;
    // std::cout << "*CPU: " << cpuOutputPath << std::endl;
    // std::cout << "*GPU: " << gpuOutputPath << std::endl;
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
    std::string cpuOutputPath = "tests/fourier_tests/fourier_descriptor_cpu.csv";
    std::string gpuOutputPath = "tests/fourier_tests/fourier_descriptor_gpu.csv";

    // Run the test
    testFourierDescriptor(inputImagePath, cpuOutputPath, gpuOutputPath);

    return EXIT_SUCCESS;
}
