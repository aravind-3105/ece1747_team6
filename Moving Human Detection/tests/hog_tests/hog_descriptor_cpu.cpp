#include "hog_descriptor_cpu.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

// Helper function to compute gradients (Gx, Gy) and gradient magnitude and orientation
void computeGradientKernel_cpu(const cv::Mat& inputImage, cv::Mat& gx, cv::Mat& gy, cv::Mat& magnitude, cv::Mat& orientation) {
    int width = inputImage.cols;
    int height = inputImage.rows;

    // Initialize output matrices
    gx = cv::Mat::zeros(height, width, CV_32F);
    gy = cv::Mat::zeros(height, width, CV_32F);
    magnitude = cv::Mat::zeros(height, width, CV_32F);
    orientation = cv::Mat::zeros(height, width, CV_32F);

    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            float gx_val = inputImage.at<float>(y, x + 1) - inputImage.at<float>(y, x - 1);
            float gy_val = inputImage.at<float>(y + 1, x) - inputImage.at<float>(y - 1, x);

            gx.at<float>(y, x) = gx_val;
            gy.at<float>(y, x) = gy_val;
            magnitude.at<float>(y, x) = std::sqrt(gx_val * gx_val + gy_val * gy_val);
            float angle = std::atan2(gy_val, gx_val) * 180.0f / CV_PI;
            orientation.at<float>(y, x) = (angle < 0) ? angle + 180.0f : angle;
        }
    }
}

// Helper function to compute HOG histograms for each cell
void computeHOGKernel_cpu(const cv::Mat& magnitude, const cv::Mat& orientation, std::vector<float>& histogram, int cell_size, int num_bins) {
    int width = magnitude.cols;
    int height = magnitude.rows;
    int cells_x = width / cell_size;
    int cells_y = height / cell_size;
    histogram.resize(cells_x * cells_y * num_bins, 0.0f);

    float bin_size = 180.0f / num_bins;

    for (int cell_y = 0; cell_y < cells_y; ++cell_y) {
        for (int cell_x = 0; cell_x < cells_x; ++cell_x) {
            int histogram_idx = (cell_y * cells_x + cell_x) * num_bins;

            for (int y = 0; y < cell_size; ++y) {
                for (int x = 0; x < cell_size; ++x) {
                    int pixel_x = cell_x * cell_size + x;
                    int pixel_y = cell_y * cell_size + y;

                    if (pixel_x >= width || pixel_y >= height) continue;

                    float mag = magnitude.at<float>(pixel_y, pixel_x);
                    float angle = orientation.at<float>(pixel_y, pixel_x);

                    int bin = static_cast<int>(angle / bin_size) % num_bins;
                    histogram[histogram_idx + bin] += mag;
                }
            }
        }
    }
}

void computeHOG_cpu(const std::string& inputImagePath, const std::string& outputHistogramPath) {
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

    // Compute gradients
    cv::Mat gx, gy, magnitude, orientation;
    computeGradientKernel_cpu(floatImage, gx, gy, magnitude, orientation);

    // Compute HOG histograms
    std::vector<float> histogram;
    computeHOGKernel_cpu(magnitude, orientation, histogram, cell_size, num_bins);

    // Normalize histogram
    float max_val = *std::max_element(histogram.begin(), histogram.end());
    if (max_val > 0.0f) {
        for (auto& val : histogram) {
            val /= max_val;
        }
    }

    // // Save to CSV
    // std::ofstream outFile(outputHistogramPath);
    // for (size_t i = 0; i < histogram.size(); ++i) {
    //     outFile << histogram[i];
    //     if ((i + 1) % num_bins == 0) {
    //         outFile << "\n";
    //     } else {
    //         outFile << ",";
    //     }
    // }
    // outFile.close();

    // std::cout << "HOG histograms saved to " << outputHistogramPath << "." << std::endl;
}
