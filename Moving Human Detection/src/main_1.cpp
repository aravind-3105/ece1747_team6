#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <complex>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <filesystem>

// Extract the largest contour from a binary image
std::vector<cv::Point> extractContour(const cv::Mat& binaryImage) {
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
std::vector<cv::Point2f> resampleContour(const std::vector<cv::Point>& contour, int numPoints) {
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

// Normalize descriptor
std::vector<float> normalizeDescriptor(const std::vector<std::complex<float>>& descriptor) {
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

// Compute Fourier descriptor for an image path
std::vector<float> computeFourierDescriptor(const std::string& imagePath, int numPoints = 128) {
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
    auto contour = extractContour(binaryImage);
    
    // Resample the contour to a fixed number of points
    std::vector<cv::Point2f> resampledContour = resampleContour(contour, numPoints);
    
    // Convert to complex representation
    std::vector<std::complex<float>> complexContour;
    complexContour.reserve(numPoints);
    
    for (const auto& point : resampledContour) {
        complexContour.emplace_back(point.x, point.y);
    }
    
    // Compute FFT using OpenCV's DFT
    cv::Mat complexMat(complexContour.size(), 1, CV_32FC2);
    std::memcpy(complexMat.data, complexContour.data(), 
                complexContour.size() * sizeof(std::complex<float>));
    
    cv::Mat fftResult;
    cv::dft(complexMat, fftResult, cv::DFT_COMPLEX_OUTPUT);
    
    // Extract magnitudes
    std::vector<std::complex<float>> fftComplex(
        reinterpret_cast<std::complex<float>*>(fftResult.data),
        reinterpret_cast<std::complex<float>*>(fftResult.data) + fftResult.rows
    );
    
    // Normalize descriptor
    return normalizeDescriptor(fftComplex);
}

// Compute Fourier descriptor from an image matrix
std::vector<float> computeFourierDescriptorFromImage(const cv::Mat& image, int numPoints = 128) {
    // Ensure grayscale
    cv::Mat grayImage;
    if (image.channels() > 1) {
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    } else {
        grayImage = image.clone();
    }
    
    // Threshold the image
    cv::Mat binaryImage;
    cv::threshold(grayImage, binaryImage, 128, 255, cv::THRESH_BINARY);
    cv::resize(binaryImage, binaryImage, cv::Size(64, 128));
    
    // Extract the largest contour
    auto contour = extractContour(binaryImage);
    
    // Resample the contour to a fixed number of points
    std::vector<cv::Point2f> resampledContour = resampleContour(contour, numPoints);
    
    // Convert to complex representation
    std::vector<std::complex<float>> complexContour;
    complexContour.reserve(numPoints);
    
    for (const auto& point : resampledContour) {
        complexContour.emplace_back(point.x, point.y);
    }
    
    // Compute FFT using OpenCV's DFT
    cv::Mat complexMat(complexContour.size(), 1, CV_32FC2);
    std::memcpy(complexMat.data, complexContour.data(), 
                complexContour.size() * sizeof(std::complex<float>));
    
    cv::Mat fftResult;
    cv::dft(complexMat, fftResult, cv::DFT_COMPLEX_OUTPUT);
    
    // Extract magnitudes
    std::vector<std::complex<float>> fftComplex(
        reinterpret_cast<std::complex<float>*>(fftResult.data),
        reinterpret_cast<std::complex<float>*>(fftResult.data) + fftResult.rows
    );
    
    // Normalize descriptor
    return normalizeDescriptor(fftComplex);
}


namespace fs = std::filesystem;

// // Function to save descriptors to a file
void saveDescriptorToFile(const std::vector<float>& descriptor, const std::string& filePath) {
    std::ofstream outFile(filePath);
    if (!outFile.is_open()) {
        throw std::runtime_error("Could not open file to write: " + filePath);
    }
    for (size_t i = 0; i < descriptor.size(); ++i) {
        outFile << descriptor[i];
        if (i != descriptor.size() - 1) outFile << ",";
    }
    outFile << "\n";
    outFile.close();
}

// Process all images in a directory
void processDirectory(const std::string& inputDir, const std::string& outputDir, int numPoints = 128) {
    for (const auto& entry : fs::recursive_directory_iterator(inputDir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".png") {
            try {
                std::string inputFilePath = entry.path().string();
                std::string relativePath = fs::relative(entry.path(), inputDir).string();
                std::string outputFilePath = outputDir + "/" + relativePath + ".csv";
                
                // Create output directories if they don't exist
                fs::create_directories(fs::path(outputFilePath).parent_path());

                // Compute Fourier descriptor
                auto descriptor = computeFourierDescriptor(inputFilePath, numPoints);

                // Save descriptor to file
                saveDescriptorToFile(descriptor, outputFilePath);

                std::cout << "Processed: " << inputFilePath << " -> " << outputFilePath << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Error processing file " << entry.path() << ": " << e.what() << std::endl;
            }
        }
    }
}

int main() {
    // Input and output directories
    // std::string inputDir = "/content/data/fourier_data/";
    std::string inputDir = "/content/collab_backup/combined_steps/bounding_images";
    
    std::string outputDir ="/content/collab_backup/combined_steps/bounding_images_cpp/";

    try {
        // Process the entire directory
        processDirectory(inputDir, outputDir);

        std::cout << "Processing complete. Fourier descriptors saved to: " << outputDir << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}




// int main() {
//     try {
//         // Example usage
//         auto descriptor = computeFourierDescriptor("binary_bounding_2.png");
        
//         // Print descriptor
//         for (const auto& val : descriptor) {
//             std::cout << val << " ";
//         }
        
//         std::cout << std::endl;
//         // Save the descriptor to a file
//         try {
            
//             std::ofstream outFile("descriptor_test_bounding_2_cpp.csv");
//             if (!outFile.is_open()) {
//                 throw std::runtime_error("Could not open file to write: descriptor_test_bounding_2.csv");
//             }
//             for (size_t i = 0; i < descriptor.size(); ++i) {
//                 outFile << descriptor[i];
//                 if (i != descriptor.size() - 1) outFile << ",";
//             }
//             outFile << "\n";
//             outFile.close();
            
//             std::cout << "Descriptor saved to: descriptor_test_bounding_2_cpp.csv" << std::endl;
//         }
//         catch (const std::exception& e) {
//             std::cerr << "Error saving descriptor: " << e.what() << std::endl;
//             return 1;
//         }
//     }
//     catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//         return 1;
//     }

    
    
//     return 0;
// }