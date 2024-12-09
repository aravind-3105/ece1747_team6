#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>
#include <complex>
#include "fourier_descriptor_cpu.h"

// Helper function to extract contours using OpenCV
std::vector<cv::Point> extractContourCPU(const cv::Mat& binaryImage) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binaryImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    if (contours.empty()) {
        throw std::runtime_error("No contours found in the image.");
    }

    // Return the largest contour
    return *std::max_element(contours.begin(), contours.end(),
        [](const std::vector<cv::Point>& c1, const std::vector<cv::Point>& c2) {
            return cv::contourArea(c1) < cv::contourArea(c2);
        });
}

// Resample contour to a fixed number of points
std::vector<cv::Point2f> resampleContourCPU(const std::vector<cv::Point>& contour, int numPoints) {
    std::vector<cv::Point2f> resampledContour;

    // Calculate total contour length
    float totalLength = cv::arcLength(contour, true);
    float step = totalLength / numPoints;

    float accumulatedLength = 0;
    size_t pointIndex = 0;

    for (int i = 0; i < numPoints; ++i) {
        float targetLength = i * step;

        while (pointIndex + 1 < contour.size()) {
            float segmentLength = cv::norm(contour[pointIndex + 1] - contour[pointIndex]);

            if (accumulatedLength + segmentLength >= targetLength) {
                float alpha = (targetLength - accumulatedLength) / segmentLength;
                cv::Point2f interpolatedPoint =
                    (1 - alpha) * contour[pointIndex] + alpha * contour[pointIndex + 1];
                resampledContour.push_back(interpolatedPoint);
                break;
            }

            accumulatedLength += segmentLength;
            ++pointIndex;
        }
    }

    return resampledContour;
}

// Perform manual DFT on the input data
std::vector<std::complex<float>> computeDFTCPU(const std::vector<std::complex<float>>& input) {
    size_t N = input.size();
    std::vector<std::complex<float>> output(N);

    const float PI = 3.14159265358979323846;
    for (size_t k = 0; k < N; ++k) {
        std::complex<float> sum(0.0f, 0.0f);
        for (size_t n = 0; n < N; ++n) {
            float angle = -2.0f * PI * k * n / N;
            std::complex<float> expTerm(cos(angle), sin(angle));
            sum += input[n] * expTerm;
        }
        output[k] = sum;
    }

    return output;
}

// Normalize descriptor
std::vector<float> normalizeDescriptorCPU(const std::vector<std::complex<float>>& descriptor) {
    if (descriptor.empty()) return {};

    // Find first non-zero magnitude
    auto firstNonZero = std::find_if(descriptor.begin(), descriptor.end(),
        [](const std::complex<float>& c) { return std::abs(c) != 0; });

    float scaleFactor = firstNonZero != descriptor.end() ? std::abs(*firstNonZero) : 1.0f;

    // Normalize and take absolute values
    std::vector<float> normalized;
    normalized.reserve(descriptor.size());

    std::transform(descriptor.begin(), descriptor.end(), std::back_inserter(normalized),
        [scaleFactor](const std::complex<float>& c) {
            return std::abs(c / scaleFactor);
        });

    return normalized;
}

// Function to compute Fourier descriptor
std::vector<float> computeFourierDescriptor_pathCPU(const std::string& imagePath, int numPoints) {
    // Load and preprocess the image
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        throw std::runtime_error("Could not load image: " + imagePath);
    }

    // Threshold the image
    cv::Mat binaryImage;
    cv::threshold(image, binaryImage, 128, 255, cv::THRESH_BINARY);
    cv::resize(binaryImage, binaryImage, cv::Size(64, 128));

    // Extract the largest contour
    auto contour = extractContourCPU(binaryImage);

    // Resample the contour to a fixed number of points
    std::vector<cv::Point2f> resampledContour = resampleContourCPU(contour, numPoints);

    // Convert to complex representation
    std::vector<std::complex<float>> complexContour;
    complexContour.reserve(numPoints);

    for (const auto& point : resampledContour) {
        complexContour.emplace_back(point.x, point.y);
    }

    // Compute DFT manually
    std::vector<std::complex<float>> dftResult = computeDFTCPU(complexContour);

    // Normalize descriptor
    return normalizeDescriptorCPU(dftResult);
}
