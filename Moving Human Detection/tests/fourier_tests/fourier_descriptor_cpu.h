#ifndef FOURIER_DESCRIPTOR_CPU_H
#define FOURIER_DESCRIPTOR_CPU_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <complex>

// Function to extract the largest contour from a binary image
std::vector<cv::Point> extractContourCPU(const cv::Mat& binaryImage);

// Function to resample a contour to a fixed number of points
std::vector<cv::Point2f> resampleContourCPU(const std::vector<cv::Point>& contour, int numPoints);

// Function to compute the Discrete Fourier Transform (DFT)
std::vector<std::complex<float>> computeDFTCPU(const std::vector<std::complex<float>>& input);

// Function to normalize Fourier descriptors
std::vector<float> normalizeDescriptorCPU(const std::vector<std::complex<float>>& descriptor);

// Function to compute the Fourier descriptor from an image path
std::vector<float> computeFourierDescriptor_pathCPU(const std::string& imagePath, int numPoints = 128);

#endif // FOURIER_DESCRIPTOR_CPU_H
