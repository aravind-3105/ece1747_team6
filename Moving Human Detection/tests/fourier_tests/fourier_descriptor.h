#ifndef FOURIER_DESCRIPTOR_H
#define FOURIER_DESCRIPTOR_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <cufft.h>
#include <complex>

// CUDA kernel to compute the magnitude of complex numbers
__global__ void computeMagnitude(const cufftComplex* input, float* output, int numPoints);

// Function to extract the largest contour from a binary image
std::vector<cv::Point> extractContour(const cv::Mat& binaryImage);

// Function to resample a contour to a fixed number of points
std::vector<cv::Point2f> resampleContour(const std::vector<cv::Point>& contour, int numPoints);

// Function to normalize Fourier descriptors
std::vector<float> normalizeDescriptor(const std::vector<std::complex<float>>& descriptor);

// Function to compute the Fourier descriptor from an image path
std::vector<float> computeFourierDescriptor_path(const std::string& imagePath, int numPoints = 128);

// Function to compute the Fourier descriptor from an in-memory image
std::vector<float> computeFourierDescriptorFromImage(const cv::Mat& image, int numPoints = 128);

#endif // FOURIER_DESCRIPTOR_H
