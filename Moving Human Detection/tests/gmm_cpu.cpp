#include "gmm.h"
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <fstream>

namespace fs = std::filesystem;

// Define the structure for a Gaussian model per pixel
struct GaussianModel {
    float mean;
    float variance;
    float weight;
};

void cpuGMM(const cv::Mat& frame, std::vector<std::vector<GaussianModel>>& gmmModels, float alpha) {
    int rows = frame.rows;
    int cols = frame.cols;
    int channels = frame.channels();

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cv::Vec3b pixel = frame.at<cv::Vec3b>(i, j);
            for (int c = 0; c < channels; c++) {
                float value = pixel[c];
                GaussianModel& model = gmmModels[i * cols + j][c];

                // Update Gaussian parameters (weight, mean, variance)
                float diff = value - model.mean;
                model.mean += alpha * diff;
                model.variance += alpha * (diff * diff - model.variance);
                model.weight = (1 - alpha) * model.weight + alpha;  // Adjust the weight
            }
        }
    }
}

void initializeGMM(std::vector<std::vector<GaussianModel>>& gmmModels, int rows, int cols) {
    gmmModels.resize(rows * cols, std::vector<GaussianModel>(N_COMPONENTS));
    for (auto& pixelModels : gmmModels) {
        for (auto& model : pixelModels) {
            model.mean = 0.0f;
            model.variance = 15.0f;  // Initial variance
            model.weight = 1.0f / N_COMPONENTS;  // Equal weight
        }
    }
}

void measureCpuGMM(const cv::Mat& frame) {
    // Initialize the GMM
    std::vector<std::vector<GaussianModel>> gmmModels;
    initializeGMM(gmmModels, frame.rows, frame.cols);

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Run the CPU GMM
    cpuGMM(frame, gmmModels, 0.01f);  // Alpha is learning rate

    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "CPU GMM Time: " << elapsed << " ms" << std::endl;
}

int processFolderCPU(const std::string& folder_path, const std::string& output_background_name) {
    // Load all images from the folder
    std::vector<cv::Mat> frames;
    std::vector<std::string> file_paths;
    cv::glob(folder_path + "/*.bmp", file_paths, false);

    int count = 0;
    for (const auto& file_path : file_paths) {
        if (count % 2 == 0) {
            count++;
            continue;
        }
        cv::Mat image = cv::imread(file_path);
        if (!image.empty()) {
            frames.push_back(image);
        } else {
            std::cerr << "Failed to load image: " << file_path << std::endl;
        }
        count++;
    }

    if (frames.empty()) {
        std::cerr << "No images loaded in folder " << folder_path << ". Exiting." << std::endl;
        return -1;
    }

    std::cout << "Processing folder (CPU): " << folder_path << std::endl;
    std::cout << "Number of frames: " << frames.size() << std::endl;

    int height = frames[0].rows;
    int width = frames[0].cols;

    // Check for correct image type (CV_8UC3)
    if (frames[0].type() != CV_8UC3) {
        std::cerr << "Expected images of type CV_8UC3 (8-bit, 3 channels). Exiting." << std::endl;
        return -1;
    }

    // Initialize GMM models for each pixel
    std::vector<std::vector<GaussianModel>> gmmModels;
    initializeGMM(gmmModels, height, width);

    // Process each frame
    for (const auto& frame : frames) {
        cpuGMM(frame, gmmModels, 0.01f); // Alpha is the learning rate
    }

    // Generate background
    cv::Mat background_image(height, width, CV_8UC3);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            cv::Vec3b& pixel = background_image.at<cv::Vec3b>(i, j);
            for (int c = 0; c < 3; c++) {
                pixel[c] = static_cast<uchar>(gmmModels[i * width + j][c].mean);
            }
        }
    }

    // Save background as an image
    cv::imwrite(output_background_name, background_image);

    std::cout << "Background saved to " << output_background_name << std::endl;
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <root_folder>" << std::endl;
        return -1;
    }

    std::string root_folder = argv[1];
    std::ofstream timingsFile("cpu_timings.csv");
    timingsFile << "Folder,Time Taken (ms)\n";

    for (const auto& entry : fs::directory_iterator(root_folder)) {
        if (entry.is_directory()) {
            std::string folder_path = entry.path().string();
            std::string output_background_name = "cpu_background_" + entry.path().filename().string() + ".png";

            auto start_time = std::chrono::high_resolution_clock::now();
            int result = processFolderCPU(folder_path, output_background_name);
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


// // Single image int main()
// int main() {
//     cv::Mat frame = cv::imread("frame.bmp");

//     if (frame.empty()) {
//         std::cerr << "Failed to load image." << std::endl;
//         return -1;
//     }

//     // Measure CPU GMM timing
//     measureCpuGMM(frame);

//     return 0;
// }
