#include <iostream>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cuda.h>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <string>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <tuple>
#include "gmm.h"
#include "fourier_descriptor.h"
#include "hog_descriptor.h"

#include "json.hpp"  // Include JSON library
namespace fs = std::filesystem;

using json = nlohmann::json;


// Macro for error-checking
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            std::printf("CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)


// Function to execute the GMM module
cv::Mat processGMM(const std::string& input_folder, const std::string& output_file) {
    // Load all images from the folder
    std::vector<cv::Mat> frames;
    std::vector<std::string> file_paths;
    cv::glob(input_folder + "/*.bmp", file_paths, false);

    for (const auto& file_path : file_paths) {
        cv::Mat image = cv::imread(file_path);
        if (!image.empty()) {
            frames.push_back(image);
        } else {
            std::cerr << "Failed to load image: " << file_path << std::endl;
        }
    }

    if (frames.empty()) {
        std::cerr << "No images loaded. Exiting." << std::endl;
        return cv::Mat();
    }

    int n_frames = frames.size();
    int height = frames[0].rows;
    int width = frames[0].cols;

    // Check for correct image type (CV_8UC3)
    if (frames[0].type() != CV_8UC3) {
        std::cerr << "Expected images of type CV_8UC3 (8-bit, 3 channels). Exiting." << std::endl;
        return cv::Mat();
    }

    // Allocate memory for input frames
    float* d_frames;
    CUDA_CHECK(cudaMalloc(&d_frames, n_frames * height * width * 3 * sizeof(float)));

    // Allocate a temporary float array for frame data
    std::vector<float> frame_float(height * width * 3);

    // Copy frames to GPU
    for (int i = 0; i < n_frames; i++) {
        cv::Mat frame = frames[i];
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                for (int c = 0; c < 3; c++) {
                    frame_float[(h * width + w) * 3 + c] = static_cast<float>(frame.at<cv::Vec3b>(h, w)[c]);
                }
            }
        }
        CUDA_CHECK(cudaMemcpy(d_frames + i * height * width * 3, frame_float.data(), height * width * 3 * sizeof(float), cudaMemcpyHostToDevice));
    }

    // Allocate memory for GMM models and background
    GMMModel* d_gmm_models;
    CUDA_CHECK(cudaMalloc(&d_gmm_models, width * height * sizeof(GMMModel)));

    float* d_background;
    CUDA_CHECK(cudaMalloc(&d_background, height * width * 3 * sizeof(float)));

    // Launch GMM fitting kernel
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
    cv::imwrite(output_file, background_image);

    // Free GPU and host memory
    CUDA_CHECK(cudaFree(d_frames));
    CUDA_CHECK(cudaFree(d_gmm_models));
    CUDA_CHECK(cudaFree(d_background));
    delete[] h_background;

    std::cout << "Background saved to " << output_file << std::endl;

    return background_image;
}

cv::Mat extractForeground(const cv::Mat background, const cv::Mat frame, const std::string& output_path) {
    // Load the background model

    // Convert to float for precise calculations
    cv::Mat backgroundFloat, frameFloat;
    background.convertTo(backgroundFloat, CV_32F);
    frame.convertTo(frameFloat, CV_32F);

    // Compute the absolute difference between the frame and the background
    cv::Mat foreground;
    cv::absdiff(frameFloat, backgroundFloat, foreground);

    // Normalize the result for visualization (0-255)
    cv::Mat foregroundNormalized;
    cv::normalize(foreground, foregroundNormalized, 0, 255, cv::NORM_MINMAX);

    // Convert back to uint8 for saving/viewing
    cv::Mat foregroundUint8;
    foregroundNormalized.convertTo(foregroundUint8, CV_8UC3);

    // Save the result
    cv::imwrite(output_path, foregroundUint8);

    // std::cout << "Foreground saved to " << output_path << std::endl;

    return foregroundUint8;
}

// Function to preprocess the foreground image
cv::Mat preprocessForeground(const cv::Mat& foreground, const std::string& output_preprocessed) {
    cv::Mat grayForeground;

    // Convert to grayscale if necessary
    if (foreground.channels() > 1) {
        cv::cvtColor(foreground, grayForeground, cv::COLOR_BGR2GRAY);
    } else {
        grayForeground = foreground.clone();
    }

    if (grayForeground.empty()) {
        std::cerr << "Error converting foreground to grayscale." << std::endl;
        return cv::Mat();
    }

    // Blur Filtering
    cv::Mat blurred;
    cv::blur(grayForeground, blurred, cv::Size(3, 3));

    // Thresholding to binary
    cv::Mat thresholded;
    cv::threshold(blurred, thresholded, 50, 255, cv::THRESH_BINARY);

    // Save the preprocessed image (optional)
    if (!output_preprocessed.empty()) {
        cv::imwrite(output_preprocessed, thresholded);
    }

    return thresholded;
}


// Function to apply Connected Component Labeling (CCL)
std::vector<cv::Mat> applyCCL(const cv::Mat binaryImage, const cv::Mat originalImage, const std::string& output_labeled) {

    // Apply Connected Component Labeling (CCL)
    cv::Mat labels, stats, centroids;
    if (binaryImage.type() != CV_8UC1) {
        throw std::runtime_error("Input to connectedComponentsWithStats must be CV_8UC1 (binary image).");
    }
    int numComponents = cv::connectedComponentsWithStats(binaryImage, labels, stats, centroids);
    // Print number of components
    // std::cout << "Number of components: " << numComponents << std::endl;
    std::vector<cv::Mat> extractedComponents;

    int printedComponents = 0;

    // Extract components and draw bounding boxes
    for (int i = 1; i < numComponents; i++) { // Start from 1 to skip the background
        int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        int y = stats.at<int>(i, cv::CC_STAT_TOP);
        int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        int area = stats.at<int>(i, cv::CC_STAT_AREA);

        // Filter out small components based on area (optional)
        if (area > 50) { // Adjust the threshold as needed
            // Extract the individual component from the original image
            x = std::max(5, x);
            y = std::max(5, y);
            width = std::min(width, originalImage.cols - x - 5);
            height = std::min(height, originalImage.rows - y - 5);
            cv::Mat component = originalImage(cv::Rect(x-5, y-5, width+10, height+10)).clone();
            
            
            cv::resize(component, component, cv::Size(64, 128));
            extractedComponents.push_back(component);

            // Draw a rectangle around the component
            cv::rectangle(originalImage, cv::Rect(x, y, width, height), cv::Scalar(0, 255, 0), 2);

            // Save the extracted component
            std::string filename = "component_" + std::to_string(printedComponents) + ".png";
            cv::imwrite(filename, component);
            printedComponents++;
        }
    }

    // Save the result with bounding boxes
    cv::imwrite(output_labeled, originalImage);
    // std::cout << "Labeled output saved to " << output_labeled << std::endl;

    return extractedComponents;
}


// Function to generate binary bounding images
std::vector<cv::Mat> generateBinaryBoundingImages(const cv::Mat binaryImage) {
    // Load the binary preprocessed image
    // cv::Mat binaryImage = cv::imread(preprocessed_image, cv::IMREAD_GRAYSCALE);
    // if (binaryImage.empty()) {
    //     std::cerr << "Error: Could not load " << preprocessed_image << ". Check the file path." << std::endl;
    //     return {};
    // }

    // Apply Connected Component Labeling (CCL)
    cv::Mat labels, stats, centroids;
    int numComponents = cv::connectedComponentsWithStats(binaryImage, labels, stats, centroids);

    // Vector to store the binary bounding images
    std::vector<cv::Mat> binaryBoundingImages;

    // Iterate over each component to create binary bounding images
    for (int i = 1; i < numComponents; i++) { // Skip the background (label 0)
        int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        int y = stats.at<int>(i, cv::CC_STAT_TOP);
        int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);

        x = std::max(0, x);
        y = std::max(0, y);
        width = std::min(width, binaryImage.cols - x);
        height = std::min(height, binaryImage.rows - y);

        // Extract binary mask for this component
        cv::Mat componentMask = (labels == i);

        // Crop the mask to the bounding box
        cv::Mat binaryBoundingImage = componentMask(cv::Rect(x, y, width, height));

        // Add the binary bounding image to the vector
        binaryBoundingImages.push_back(binaryBoundingImage);

        // Save the binary bounding image
        std::string filename = "binary_bounding_" + std::to_string(i) + ".png";
        cv::imwrite(filename, binaryBoundingImage * 255); // Scale mask to 255 for saving

        // std::cout << "Saved: " << filename << std::endl;
    }

    // std::cout << "Binary bounding images saved for all components." << std::endl;

    return binaryBoundingImages;
}

// // Function to compute Fourier Descriptor
// void normalizeDescriptor(std::vector<float>& descriptor, const std::vector<float>& mean, const std::vector<float>& stdDev) {
//     for (size_t i = 0; i < descriptor.size(); ++i) {
//         descriptor[i] = (descriptor[i] - mean[i]) / stdDev[i];
//     }
// }

// Function to load SVM weights
std::pair<std::vector<float>, float> loadSVMModel(const std::string& modelPath) {
    std::ifstream file(modelPath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open SVM model file.");
    }

    std::vector<float> weights;
    float weight;
    while (file >> weight) {
        weights.push_back(weight);
    }

    // The last value is the bias term
    float bias = weights.back();
    weights.pop_back();

    return {weights, bias};
}

// Function to normalise SVM scaler
std::vector<float> scaleDescriptorWithJson(const std::vector<float>& descriptor, const std::string& jsonPath) {
    // Load the JSON file
    nlohmann::json jsonData;
    std::ifstream file(jsonPath);
    if (!file.is_open()) {
        throw std::runtime_error("Error opening JSON file: " + jsonPath);
    }
    file >> jsonData;

    // Extract mean and variance from JSON
    std::vector<float> mean = jsonData["mean"].get<std::vector<float>>();
    std::vector<float> variance = jsonData["variance"].get<std::vector<float>>();

    // Check if sizes match
    if (descriptor.size() != mean.size() || descriptor.size() != variance.size()) {
        throw std::invalid_argument("Descriptor size does not match the mean or variance size in the JSON file.");
    }

    // Scale the descriptor
    std::vector<float> scaledDescriptor(descriptor.size());
    for (size_t i = 0; i < descriptor.size(); ++i) {
        if (variance[i] == 0.0f) {
            // Handle zero variance: set scaled value to 0.0
            scaledDescriptor[i] = 0.0f;
            continue;
        }
        scaledDescriptor[i] = (descriptor[i] - mean[i]) / std::sqrt(variance[i]);
    }

    return scaledDescriptor;
}



// Function to make predictions SVM prediction function
int predict(const std::vector<float>& descriptor, const std::vector<float>& weights, float bias) {
    if (descriptor.size() != weights.size()) {
        throw std::invalid_argument("Descriptor and weights must have the same size.");
    }

    float dotProduct = 0.0f;
    for (size_t i = 0; i < descriptor.size(); ++i) {
        dotProduct += descriptor[i] * weights[i];
    }

    float score = dotProduct + bias;
    return score >= 0 ? 1 : 0;
}


std::vector<float> flattenWeights(const std::vector<std::vector<float>>& weights2D) {
    std::vector<float> weights1D;
    for (const auto& row : weights2D) {
        weights1D.insert(weights1D.end(), row.begin(), row.end());
    }
    return weights1D;
}


// Function to load the SVM model from JSON
void loadSVMModel(const std::string& jsonPath, std::vector<std::vector<float>>& weights, std::vector<float>& bias) {
    std::ifstream inputFile(jsonPath);
    if (!inputFile.is_open()) {
        throw std::runtime_error("Unable to open JSON file");
    }

    json svmModel;
    inputFile >> svmModel;

    // Extract weights and bias
    weights = svmModel["weights"].get<std::vector<std::vector<float>>>();
    bias = svmModel["bias"].get<std::vector<float>>();
}

// Function to compute mean and standard deviation for each feature
void computeMeanAndStdDev(const std::vector<std::vector<float>>& data, std::vector<float>& mean, std::vector<float>& stdDev) {
    if (data.empty()) {
        throw std::runtime_error("Data is empty.");
    }

    size_t numSamples = data.size();
    size_t numFeatures = data[0].size();

    // Initialize mean and stdDev vectors
    mean.assign(numFeatures, 0.0f);
    stdDev.assign(numFeatures, 0.0f);

    // Compute the mean for each feature
    for (const auto& sample : data) {
        for (size_t i = 0; i < numFeatures; ++i) {
            mean[i] += sample[i];
        }
    }

    for (size_t i = 0; i < numFeatures; ++i) {
        mean[i] /= static_cast<float>(numSamples);
    }

    // Compute the standard deviation for each feature
    for (const auto& sample : data) {
        for (size_t i = 0; i < numFeatures; ++i) {
            float diff = sample[i] - mean[i];
            stdDev[i] += diff * diff;
        }
    }

    for (size_t i = 0; i < numFeatures; ++i) {
        stdDev[i] = std::sqrt(stdDev[i] / static_cast<float>(numSamples));
    }
}

// Function to normalize the descriptor
void normalizeHOG(std::vector<float>& descriptor, const std::vector<float>& mean, const std::vector<float>& std_dev) {
    for (size_t i = 0; i < descriptor.size(); ++i) {
        descriptor[i] = (descriptor[i] - mean[i]) / std_dev[i];
    }
}

// Helper function to compute HOG descriptors for an image
std::vector<float> computeHOG_predict(const std::string& imagePath, const std::vector<float>& mean, const std::vector<float>& std_dev) {
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        throw std::runtime_error("Could not load image: " + imagePath);
    }

    cv::resize(image, image, cv::Size(128, 128));

    cv::HOGDescriptor hog(
        cv::Size(128, 128),
        cv::Size(16, 16),
        cv::Size(8, 8),
        cv::Size(8, 8),
        9
    );

    std::vector<float> descriptors;
    hog.compute(image, descriptors);

    normalizeHOG(descriptors, mean, std_dev);

    return descriptors;
}


// Function to predict if an image is of a human or not
std::string predictHumanOrNot(const std::string& imagePath, const cv::Ptr<cv::ml::SVM>& svm, 
                              const std::vector<float>& mean, const std::vector<float>& std_dev) {
    try {
        // Compute and normalize HOG descriptor
        std::vector<float> descriptor = computeHOG_predict(imagePath, mean, std_dev);

        // Convert to cv::Mat for prediction
        cv::Mat descriptorMat(1, descriptor.size(), CV_32F, descriptor.data());

        // Predict using SVM
        float response = svm->predict(descriptorMat);
        return (response == 1) ? "person" : "non-person";
    } catch (const std::exception& e) {
        std::cerr << "Error processing " << imagePath << ": " << e.what() << std::endl;
        return "error";
    }
}

int main(int argc, char* argv[]) {
    // Default values
    std::string input_folder = "../../data/OSUdata/5b";
    // std::string output_background = "background.png";
    std::string output_background = "/content/data/OSUdata/background/background_5b.png";
    std::string output_foreground = "foreground.png";
    std::string output_preprocessed = "preprocessed.png";
    std::string output_labeled = "labeled_output.png";
    std::string output_labeled_folder = "labeled_output";
    std::string predictions_file = "predictions_5b.csv";

    // Overwrite defaults if arguments are provided
    if (argc > 1) input_folder = argv[1];
    if (argc > 2) output_background = argv[2];

    // Get list of all .bmp files in the input folder
    std::vector<std::string> file_paths;
    cv::glob(input_folder + "/*.bmp", file_paths, false);

    // Check if the input folder is empty
    if (file_paths.empty()) {
        std::cerr << "No .bmp files found in the input folder. Exiting." << std::endl;
        return -1;
    }

    // Prepare results vector
    // std::vector<std::tuple<std::string, std::string, std::string, int, int>> results;
    std::vector<std::tuple<std::string, std::string, std::string, int, int, std::string>> results;


    // Load precomputed background
    cv::Mat background = cv::imread(output_background);
    if (background.empty()) {
        std::cerr << "Error: Could not load background image: " << output_background << std::endl;
        return -1;
    }

    // Load HOG SVM model
    std::string modelPath = "hog_svm_model.xml";
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load(modelPath);
    if (svm.empty()) {
        std::cerr << "Error loading SVM model from " << modelPath << std::endl;
        return -1;
    }

    // Load normalization parameters
    std::string normalizationPath = "hog_normalization.json";
    std::ifstream file(normalizationPath);
    nlohmann::json normalizationParams;
    file >> normalizationParams;
    std::vector<float> mean = normalizationParams["mean"];
    std::vector<float> std_dev = normalizationParams["std_dev"];

    // Load the Fourier SVM model
    std::vector<std::vector<float>> weights;
    std::vector<float> bias;
    loadSVMModel("svm_model_v3.json", weights, bias);
    std::vector<float> weightsFlattened = flattenWeights(weights);
    std::string svm_scalar_values = "scaler_values.json";


    // Process each file
    // file_paths = {"../../data/OSUdata/5b/img_00882.bmp"};
    for (const auto& frame_file : file_paths) {
        std::cout << "Processing file: " << frame_file << std::endl;

        // Load the frame
        cv::Mat frame = cv::imread(frame_file);
        cv::Mat frame_gray = cv::imread(frame_file, cv::IMREAD_GRAYSCALE);

        if (frame.empty() || frame_gray.empty()) {
            std::cerr << "Error: Could not load frame: " << frame_file << std::endl;
            continue;
        }

        // Extract foreground
        cv::Mat foreground = extractForeground(background, frame, output_foreground);

        // Preprocess the foreground
        cv::Mat preprocessed = preprocessForeground(foreground, output_preprocessed);

        // Check if the preprocessed image is valid
        if (preprocessed.empty() || preprocessed.type() != CV_8UC1) {
            std::cerr << "Preprocessed image is invalid or not binary. Skipping this file." << std::endl;
            continue;
        }

        // Apply Connected Component Labeling (CCL)
        std::string frame_filename = fs::path(frame_file).stem().string();
        std::string output_labeled_path = output_labeled_folder + "/" + frame_filename + "_labeled_output.png";
        std::vector<cv::Mat> components = applyCCL(preprocessed, frame, output_labeled_path);

        // Generate binary bounding images
        std::vector<cv::Mat> binaryBoundingImages = generateBinaryBoundingImages(preprocessed);

        // Fourier Prediction
        std::string fourier_predictions;
        int fourier_people_count = 0;

        // for (const auto& binaryImage : binaryBoundingImages) {
        //     auto descriptor = computeFourierDescriptor_image(binaryImage, 128);
        //     int prediction_fourier = predict(descriptor, weightsFlattened, bias[0]);
        //     std::string prediction = (prediction_fourier == 1 ? "person" : "non-person");
        //     fourier_predictions += prediction + "; ";  // Concatenate predictions
        //     if (prediction_fourier == 1) {
        //         fourier_people_count++;
        //     }
        // }

        // std::cout<<"Number of components: "<<components.size()<<std::endl;
        // std::cout<<"Number of binary components: "<<binaryBoundingImages.size()<<std::endl;
        // std::vector<std::string> binaryPaths_test = {"bounding_images/binary_bounding_1.png","binary_bounding_2.png","binary_bounding_3.png","binary_bounding_4.png"};
        int tt = 0;
        for (const auto& binaryImage : binaryBoundingImages) {
            // Ensure the image is binary before computing the descriptor
            // if (binaryImage.type() != CV_8UC1) {
            //     std::cerr << "Skipping non-binary image component." << std::endl;
            //     continue;
            // }

            // Compute Fourier descriptor

            auto descriptor = computeFourierDescriptorFromImage(binaryImage, 128);
            // auto descriptor = computeFourierDescriptor_path(binaryPaths_test[tt], 128);
            tt+=1;
            auto descriptor_scalar = scaleDescriptorWithJson(descriptor, svm_scalar_values);
            // Perform classification using the Fourier descriptor
            int prediction_fourier = predict(descriptor_scalar, weightsFlattened, bias[0]);
            std::string prediction = (prediction_fourier == 1 ? "person" : "non-person");
            // std::cout<<"Prediction for Fourier: "<<prediction<<std::endl;
            fourier_predictions += prediction + "; ";  // Concatenate predictions
            if (prediction_fourier == 1) {
                fourier_people_count++;
            }
        }

        // HOG Prediction
        std::string hog_predictions;
        int hog_people_count = 0;

        for (size_t i = 0; i < components.size(); ++i) {
            std::string componentPath = "component_" + std::to_string(i) + ".png";
            cv::imwrite(componentPath, components[i]);
            std::string prediction = predictHumanOrNot(componentPath, svm, mean, std_dev);
            hog_predictions += prediction + "; ";  // Concatenate predictions
            if (prediction == "person") {
                hog_people_count++;
            }
        }

        // Combined Prediction
        std::string combined_prediction;
        if (fourier_people_count > 0 && hog_people_count > 0) {
            combined_prediction = "person";
        } else {
            combined_prediction = "non-person";
        }

        // Store results
        // results.emplace_back(frame_file, fourier_predictions, hog_predictions, fourier_people_count, hog_people_count);
        results.emplace_back(frame_file, fourier_predictions, hog_predictions, fourier_people_count, hog_people_count, combined_prediction);



        // Clean up temporary files
        for (size_t i = 0; i < binaryBoundingImages.size(); ++i) {
            std::string binaryBoundingPath = "binary_bounding_" + std::to_string(i) + ".png";
            std::remove(binaryBoundingPath.c_str());
        }
        for (size_t i = 0; i < components.size(); ++i) {
            std::string componentPath = "component_" + std::to_string(i) + ".png";
            std::remove(componentPath.c_str());
        }
    }

    // Save results to a CSV file
    // std::ofstream resultsFile("predictions.csv");
    // resultsFile << "Filename,Fourier Predictions,HOG Predictions,Fourier People Count,HOG People Count\n";
    // for (const auto& [filename, fourier_preds, hog_preds, fourier_count, hog_count] : results) {
    //     resultsFile << filename << ",\"" << fourier_preds << "\",\"" << hog_preds << "\"," << fourier_count << "," << hog_count << "\n";
    // }
    // resultsFile.close();

    // std::cout << "Predictions saved to predictions.csv" << std::endl;

    // Save results to a CSV file
    std::ofstream resultsFile(predictions_file);
    resultsFile << "Filename,Fourier Predictions,HOG Predictions,Fourier People Count,HOG People Count,Combined Prediction\n";
    for (const auto& [filename, fourier_preds, hog_preds, fourier_count, hog_count, combined_pred] : results) {
        resultsFile << filename << ",\"" << fourier_preds << "\",\"" << hog_preds << "\"," << fourier_count << "," << hog_count << "," << combined_pred << "\n";
    }
    resultsFile.close();

    std::cout << "Predictions saved to predictions.csv" << std::endl;

    return 0;
}