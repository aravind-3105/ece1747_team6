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
#include <chrono>
#include "../include/gmm.h"
// If i want to have two options of header files - GPU and CPU for the same function fourier_descriptor.h or fourier_descriptor_gpu.h
#define USE_GPU 1
#ifdef USE_GPU
#include "../include/fourier_descriptor.h"
#include "../include/hog_descriptor.h"
#else
#include "../include/fourier_descriptor_cpu.h"
#include "../include/hog_descriptor_cpu.h"
#endif
#include "../include/json.hpp"  // Include JSON library

using namespace std;
namespace fs = filesystem;
using json = nlohmann::json;

// Macro for error-checking
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            printf("CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)


template <typename TimeT = chrono::milliseconds>
struct Timer {
    chrono::time_point<chrono::steady_clock> start;

    Timer() : start(chrono::steady_clock::now()) {}

    double elapsed() const {
        return chrono::duration_cast<TimeT>(chrono::steady_clock::now() - start).count();
    }

    void reset() {
        start = chrono::steady_clock::now();
    }
};

// Function to execute the GMM module - `
cv::Mat processGMM(const string& input_folder, const string& output_file) {
    // Load all images from the folder
    vector<cv::Mat> frames;
    vector<string> file_paths;
    cv::glob(input_folder + "/*.bmp", file_paths, false);

    for (const auto& file_path : file_paths) {
        cv::Mat image = cv::imread(file_path);
        if (!image.empty()) {
            frames.push_back(image);
        } else {
            cerr << "Failed to load image: " << file_path << endl;
        }
    }

    if (frames.empty()) {
        cerr << "No images loaded. Exiting." << endl;
        return cv::Mat();
    }

    int n_frames = frames.size();
    int height = frames[0].rows;
    int width = frames[0].cols;

    // Check for correct image type (CV_8UC3)
    if (frames[0].type() != CV_8UC3) {
        cerr << "Expected images of type CV_8UC3 (8-bit, 3 channels). Exiting." << endl;
        return cv::Mat();
    }

    // Allocate memory for input frames
    float* d_frames;
    CUDA_CHECK(cudaMalloc(&d_frames, n_frames * height * width * 3 * sizeof(float)));

    // Allocate a temporary float array for frame data
    vector<float> frame_float(height * width * 3);

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

    cout << "Background saved to " << output_file << endl;

    return background_image;
}

// Function to compute the Fourier descriptor from an image path 
cv::Mat extractForeground(const cv::Mat background, const cv::Mat frame, const string& output_path) {
    // Check if the input images are valid
    if (background.empty() || frame.empty()) {
        cerr << "Error: Input images are empty." << endl;
        return cv::Mat();
    }
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

    // cout << "Foreground saved to " << output_path << endl;

    return foregroundUint8;
}

// Function to preprocess the foreground image 
cv::Mat preprocessForeground(const cv::Mat& foreground, const string& preprocessed_path) {
    cv::Mat grayForeground;

    // Convert to grayscale if necessary
    if (foreground.channels() > 1) {
        cv::cvtColor(foreground, grayForeground, cv::COLOR_BGR2GRAY);
    } else {
        grayForeground = foreground.clone();
    }

    if (grayForeground.empty()) {
        cerr << "Error converting foreground to grayscale." << endl;
        return cv::Mat();
    }

    // Blur Filtering
    cv::Mat blurred;
    cv::blur(grayForeground, blurred, cv::Size(3, 3));

    // Thresholding to binary
    cv::Mat thresholded;
    cv::threshold(blurred, thresholded, 50, 255, cv::THRESH_BINARY);

    // Save the preprocessed image (optional)
    if (!preprocessed_path.empty()) {
        cv::imwrite(preprocessed_path, thresholded);
    }

    return thresholded;
}


// Function to apply Connected Component Labeling (CCL) - used for extracting individual components
vector<cv::Mat> applyCCL(const cv::Mat binaryImage, const cv::Mat originalImage, const string& output_labeled) {

    // Apply Connected Component Labeling (CCL)
    cv::Mat labels, stats, centroids;
    if (binaryImage.type() != CV_8UC1) {
        throw runtime_error("Input to connectedComponentsWithStats must be CV_8UC1 (binary image).");
    }
    int numComponents = cv::connectedComponentsWithStats(binaryImage, labels, stats, centroids);
    // Print number of components
    // cout << "Number of components: " << numComponents << endl;
    vector<cv::Mat> extractedComponents;

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
            x = max(5, x);
            y = max(5, y);
            width = min(width, originalImage.cols - x - 5);
            height = min(height, originalImage.rows - y - 5);
            cv::Mat component = originalImage(cv::Rect(x-5, y-5, width+10, height+10)).clone();
            
            
            cv::resize(component, component, cv::Size(64, 128));
            extractedComponents.push_back(component);

            // Draw a rectangle around the component
            cv::rectangle(originalImage, cv::Rect(x, y, width, height), cv::Scalar(0, 255, 0), 2);

            // Save the extracted component
            string filename = "component_" + to_string(printedComponents) + ".png";
            cv::imwrite(filename, component);
            printedComponents++;
        }
    }

    // Save the result with bounding boxes
    cv::imwrite(output_labeled, originalImage);
    // cout << "Labeled output saved to " << output_labeled << endl;

    return extractedComponents;
}

// Function to generate binary bounding images - used for Fourier descriptor
vector<cv::Mat> generateBinaryBoundingImages(const cv::Mat binaryImage) {
    // Apply Connected Component Labeling (CCL)
    cv::Mat labels, stats, centroids;
    int numComponents = cv::connectedComponentsWithStats(binaryImage, labels, stats, centroids);

    // Vector to store the binary bounding images
    vector<cv::Mat> binaryBoundingImages;

    // Iterate over each component to create binary bounding images
    for (int i = 1; i < numComponents; i++) { // Skip the background (label 0)
        int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        int y = stats.at<int>(i, cv::CC_STAT_TOP);
        int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);

        x = max(0, x);
        y = max(0, y);
        width = min(width, binaryImage.cols - x);
        height = min(height, binaryImage.rows - y);

        // Extract binary mask for this component
        cv::Mat componentMask = (labels == i);

        // Crop the mask to the bounding box
        cv::Mat binaryBoundingImage = componentMask(cv::Rect(x, y, width, height));

        // Add the binary bounding image to the vector
        binaryBoundingImages.push_back(binaryBoundingImage);

        // Save the binary bounding image
        // string filename = "binary_bounding_" + to_string(i) + ".png";
        // cv::imwrite(filename, binaryBoundingImage * 255); // Scale mask to 255 for saving
        // cout << "Saved: " << filename << endl;
    }

    // cout << "Binary bounding images saved for all components." << endl;

    return binaryBoundingImages;
}

// Function to normalise SVM scaler - used for Fourier descriptor
vector<float> scaleDescriptorWithJson(const vector<float>& descriptor, const string& jsonPath) {
    // Load the JSON file
    nlohmann::json jsonData;
    ifstream file(jsonPath);
    if (!file.is_open()) {
        throw runtime_error("Error opening JSON file: " + jsonPath);
    }
    file >> jsonData;

    // Extract mean and variance from JSON
    vector<float> mean = jsonData["mean"].get<vector<float>>();
    vector<float> variance = jsonData["variance"].get<vector<float>>();

    // Check if sizes match
    if (descriptor.size() != mean.size() || descriptor.size() != variance.size()) {
        throw invalid_argument("Descriptor size does not match the mean or variance size in the JSON file.");
    }

    // Scale the descriptor
    vector<float> scaledDescriptor(descriptor.size());
    for (size_t i = 0; i < descriptor.size(); ++i) {
        if (variance[i] == 0.0f) {
            // Handle zero variance: set scaled value to 0.0
            scaledDescriptor[i] = 0.0f;
            continue;
        }
        scaledDescriptor[i] = (descriptor[i] - mean[i]) / sqrt(variance[i]);
    }

    return scaledDescriptor;
}

// Function to make predictions SVM prediction function - used for Fourier descriptor
int predictHumanOrNotFourier(const vector<float>& descriptor, const vector<float>& weights, float bias) {
    if (descriptor.size() != weights.size()) {
        throw invalid_argument("Descriptor and weights must have the same size.");
    }

    float dotProduct = 0.0f;
    for (size_t i = 0; i < descriptor.size(); ++i) {
        dotProduct += descriptor[i] * weights[i];
    }

    float score = dotProduct + bias;
    return score >= 0 ? 1 : 0;
}

// Function to flatten 2D weights to 1D - used for Fourier descriptor
vector<float> flattenWeights(const vector<vector<float>>& weights2D) {
    vector<float> weights1D;
    for (const auto& row : weights2D) {
        weights1D.insert(weights1D.end(), row.begin(), row.end());
    }
    return weights1D;
}

// Function to load the SVM model from JSON - used for Fourier descriptor
void loadSVMModel(const string& jsonPath, vector<vector<float>>& weights, vector<float>& bias) {
    ifstream inputFile(jsonPath);
    if (!inputFile.is_open()) {
        throw runtime_error("Unable to open JSON file");
    }

    json svmModel;
    inputFile >> svmModel;

    // Extract weights and bias
    weights = svmModel["weights"].get<vector<vector<float>>>();
    bias = svmModel["bias"].get<vector<float>>();
}

// Function to compute mean and standard deviation for each feature - used for HOG descriptor
void computeMeanAndStdDev(const vector<vector<float>>& data, vector<float>& mean, vector<float>& stdDev) {
    if (data.empty()) {
        throw runtime_error("Data is empty.");
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
        stdDev[i] = sqrt(stdDev[i] / static_cast<float>(numSamples));
    }
}

// Function to normalize the descriptor - used for HOG descriptor
void normalizeHOG(vector<float>& descriptor, const vector<float>& mean, const vector<float>& std_dev) {
    for (size_t i = 0; i < descriptor.size(); ++i) {
        descriptor[i] = (descriptor[i] - mean[i]) / std_dev[i];
    }
}

// Helper function to compute HOG descriptors for an image - used for HOG descriptor
vector<float> computeHOG_predict(const string& imagePath, const vector<float>& mean, const vector<float>& std_dev) {
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        throw runtime_error("Could not load image: " + imagePath);
    }

    cv::resize(image, image, cv::Size(128, 128));

    cv::HOGDescriptor hog(
        cv::Size(128, 128),
        cv::Size(16, 16),
        cv::Size(8, 8),
        cv::Size(8, 8),
        9
    );

    vector<float> descriptors;
    hog.compute(image, descriptors);

    normalizeHOG(descriptors, mean, std_dev);

    return descriptors;
}

// Function to predict if an image is of a human or not - used for HOG descriptor
string predictHumanOrNotHOG(const string& imagePath, const cv::Ptr<cv::ml::SVM>& svm, 
                              const vector<float>& mean, const vector<float>& std_dev) {
    try {
        // Compute and normalize HOG descriptor
        vector<float> descriptor = computeHOG_predict(imagePath, mean, std_dev);

        // Convert to cv::Mat for prediction
        cv::Mat descriptorMat(1, descriptor.size(), CV_32F, descriptor.data());

        // Predict using SVM
        float response = svm->predict(descriptorMat);
        return (response == 1) ? "person" : "non-person";
    } catch (const exception& e) {
        cerr << "Error processing " << imagePath << ": " << e.what() << endl;
        return "error";
    }
}


int main(int argc, char* argv[]) {
    // Default values

    // If running locally:
    string folder_name = "5b";
    string input_folder = "../data/OSUdata/" + folder_name;
    string background_path = "../data/OSUdata/background/background_" + folder_name + "_1x.png";
    string foreground_path = "temp/foreground.png";
    string preprocessed_path = "temp/preprocessed.png";
    string output_labeled = "labeled_output.png";
    string output_labeled_folder = "temp/labeled_output";
    string predictions_path = "temp/predictions_" + folder_name + ".csv";
    string metrics_path = "temp/performance_metrics" + folder_name + ".csv";
    string modelHog_path = "../models//hog_svm_model.xml";
    string modelFourier_path = "../models/svm_model_v3.json";
    string normalization_path = "../models/hog_normalization.json";
    string svm_scalar_values_path = "../models/scaler_values.json";

    // If running on colab comment the above and uncomment the below:
    // ---------------------
    // string folder_name = "5b";
    // string input_folder = "/content/5b_upscale/3x";
    // string background_path = "/content/data/OSUdata/background/background_5b_3x.png";
    // string foreground_path = "/content/src/temp/foreground.png";
    // string preprocessed_path = "/content/src/temp/preprocessed.png";
    // string output_labeled = "labeled_output.png";
    // string output_labeled_folder = "/content/src/temp/labeled_output";
    // string predictions_path = "/content/src/temp/predictions_5b_3x.csv";
    // string metrics_path = "/content/src/temp/performance_metrics_5b_3x.csv";
    // string modelHog_path = "/content/models/hog_svm_model.xml";
    // string modelFourier_path = "/content/models/svm_model_v3.json";
    // string normalization_path = "/content/models/hog_normalization.json";
    // string svm_scalar_values_path = "/content/models/scaler_values.json";
    // ---------------------



    // Overwrite defaults if arguments are provided
    if (argc > 1) input_folder = argv[1];
    if (argc > 2) background_path = argv[2];
    if (argc > 3) foreground_path = argv[3];
    if (argc > 4) preprocessed_path = argv[4];
    if (argc > 5) output_labeled = argv[5];
    if (argc > 6) output_labeled_folder = argv[6];
    if (argc > 7) predictions_path = argv[7];
    if (argc > 8) metrics_path = argv[8];
    if (argc > 9) modelHog_path = argv[9];
    if (argc > 10) modelFourier_path = argv[10];
    if (argc > 11) normalization_path = argv[11];
    if (argc > 12) svm_scalar_values_path = argv[12];
    

    // Get list of all .bmp files in the input folder
    vector<string> file_paths;
    cv::glob(input_folder + "/*.bmp", file_paths, false);

    // Check if the input folder is empty
    if (file_paths.empty()) {
        cerr << "No .bmp files found in the input folder. Exiting." << endl;
        return -1;
    }

    // Prepare results vector <filename, fourier_predictions, hog_predictions, fourier_people_count, hog_people_count>
    // vector<tuple<string, string, string, int, int, string>> results;

    // Prepare results vector <filename, people_count, prediction>
    vector<tuple<string, int, string>> results;

    // Process GMM
        // Run the GMM module - run this only once to generate the background image and then comment it out
    // std::cout << "Running GMM module..." << std::endl;
    // processGMM(input_folder, background_path);

    // Load precomputed background
    cv::Mat background = cv::imread(background_path);
    if (background.empty()) {
        cerr << "Error: Could not load background image: " << background_path << endl;
        return -1;
    }

    // Load HOG SVM model
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load(modelHog_path);
    if (svm.empty()) {
        cerr << "Error loading SVM model from " << modelHog_path << endl;
        return -1;
    }

    // Load normalization parameters for HOG
    ifstream file(normalization_path);
    nlohmann::json normalizationParams;
    file >> normalizationParams;
    if (normalizationParams.empty()) {
        cerr << "Error loading normalization parameters from " << normalization_path << endl;
        return -1;
    }
    vector<float> mean = normalizationParams["mean"];
    vector<float> std_dev = normalizationParams["std_dev"];

    // Load the Fourier SVM model
    vector<vector<float>> weights;
    vector<float> bias;
    loadSVMModel(modelFourier_path, weights, bias);
    if (weights.empty() || bias.empty()) {
        cerr << "Error loading SVM model from " << modelFourier_path << endl;
        return -1;
    }
    vector<float> weightsFlattened = flattenWeights(weights);
    


    // Performance metrics initialization
    ofstream metricsFile(metrics_path);
    metricsFile << "Frame,Foreground Extraction (ms),Preprocessing (ms),CCL (ms),HOG Prediction (ms),Fourier Prediction (ms),Total Time (ms),FPS\n";
    metricsFile.close();
    double total_extractForeground = 0, total_preprocessForeground = 0, total_applyCCL = 0;
    double total_hogPrediction = 0, total_fourierPrediction = 0, total_pipeline_time = 0;
    int processed_frames = 0;



    // To test on a single file uncomment the below and replace the file path with the desired file path
    // file_paths = {"../data/OSUdata/5b/5b_0882.bmp"};

    // Process each file
    for (const auto& frame_file : file_paths) {
        // cout << "Processing file: " << frame_file << endl;

        // Timer for overall frame processing
        Timer<> frameTimer;
        double time_extractForeground = 0, time_preprocessForeground = 0, time_applyCCL = 0;
        double time_hogPrediction = 0, time_fourierPrediction = 0;

        frameTimer.reset();
        // Load the frame
        cv::Mat frame = cv::imread(frame_file);
        cv::Mat frame_gray = cv::imread(frame_file, cv::IMREAD_GRAYSCALE);
        if (frame.empty() || frame_gray.empty()) {
            cerr << "Error: Could not load frame: " << frame_file << endl;
            continue;
        }

        // Generate background


        // Extract foreground
        Timer<> timer;
        timer.reset();
        cv::Mat foreground = extractForeground(background, frame, foreground_path);
        time_extractForeground = timer.elapsed();

        // Preprocess the foreground
        timer.reset();
        cv::Mat preprocessed = preprocessForeground(foreground, preprocessed_path);
        time_preprocessForeground = timer.elapsed();

        // Check if the preprocessed image is valid
        if (preprocessed.empty() || preprocessed.type() != CV_8UC1) {
            cerr << "Preprocessed image is invalid or not binary. Skipping this file." << endl;
            continue;
        }

        // Apply Connected Component Labeling (CCL)
        timer.reset();
        string frame_filename = fs::path(frame_file).stem().string();
        string output_labeled_path = output_labeled_folder + "/" + frame_filename + "_labeled_output.png";
        vector<cv::Mat> components = applyCCL(preprocessed, frame, output_labeled_path);
        time_applyCCL = timer.elapsed();

        // Generate binary bounding images
        vector<cv::Mat> binaryBoundingImages = generateBinaryBoundingImages(preprocessed);

        // Fourier Prediction
        string fourier_predictions;
        int fourier_people_count = 0;


        // cout<<"Number of components: "<<components.size()<<endl;
        // cout<<"Number of binary components: "<<binaryBoundingImages.size()<<endl;
        // int tt = 0;
        // vector<string> binaryPaths_test = {"bounding_images/binary_bounding_1.png","binary_bounding_2.png","binary_bounding_3.png","binary_bounding_4.png"};

        timer.reset();
        for (const auto& binaryImage : binaryBoundingImages) {
            // Ensure the image is binary before computing the descriptor
            // if (binaryImage.type() != CV_8UC1) {
            //     cerr << "Skipping non-binary image component." << endl;
            //     continue;
            // }

            // Compute Fourier descriptor
            auto descriptor = computeFourierDescriptorFromImage(binaryImage, 128);

            // auto descriptor = computeFourierDescriptor_path(binaryPaths_test[tt], 128);
            // tt+=1;

            // Scale the descriptor using the precomputed mean and standard deviation
            auto descriptor_scalar = scaleDescriptorWithJson(descriptor, svm_scalar_values_path);
            // Perform classification using the Fourier descriptor
            int prediction_fourier = predictHumanOrNotFourier(descriptor_scalar, weightsFlattened, bias[0]);
            string prediction = (prediction_fourier == 1 ? "person" : "non-person");
            // cout<<"Prediction for Fourier: "<<prediction<<endl;
            fourier_predictions += prediction + "; ";  // Concatenate predictions
            if (prediction_fourier == 1) {
                fourier_people_count++;
            }
        }
        time_fourierPrediction = timer.elapsed();

        // HOG Prediction
        string hog_predictions;
        int hog_people_count = 0;
        timer.reset();
        for (size_t i = 0; i < components.size(); ++i) {
            string componentPath = "component_" + to_string(i) + ".png";
            cv::imwrite(componentPath, components[i]);
            string prediction = predictHumanOrNotHOG(componentPath, svm, mean, std_dev);
            hog_predictions += prediction + "; ";  // Concatenate predictions
            if (prediction == "person") {
                hog_people_count++;
            }
        }
        time_hogPrediction = timer.elapsed();


        // Combined Prediction
        string combined_prediction;
        if (fourier_people_count > 0 && hog_people_count > 0) {
            combined_prediction = "person";
        } else {
            combined_prediction = "non-person";
        }

        // Store results
        // To store <filename, fourier_predictions, hog_predictions, fourier_people_count, hog_people_count>
        // results.emplace_back(frame_file, fourier_predictions, hog_predictions, fourier_people_count, hog_people_count, combined_prediction);
        // To store <filename, people_count, prediction>
        results.emplace_back(frame_file, hog_people_count, combined_prediction);

        double total_pipeline_time = frameTimer.elapsed();
        double fps = 1000.0 / total_pipeline_time; // Convert ms to FPS (1 s = 1000 ms)

        // Log metrics for each frame
        cout << "Frame: " << frame_file << "\n";
        cout << "Foreground Extraction: " << time_extractForeground << " ms\n";
        cout << "Preprocessing: " << time_preprocessForeground << " ms\n";
        cout << "CCL: " << time_applyCCL << " ms\n";
        cout << "HOG Prediction: " << time_hogPrediction << " ms\n";
        cout << "Fourier Prediction: " << time_fourierPrediction << " ms\n";
        cout << "Total Pipeline Time: " << total_pipeline_time << " ms\n";
        cout << "FPS: " << fps << "\n";

        // Save metrics to CSV file
        ofstream metricsFile(metrics_path , ios::app);
        metricsFile << frame_file << ","
                    << time_extractForeground << ","
                    << time_preprocessForeground << ","
                    << time_applyCCL << ","
                    << time_hogPrediction << ","
                    << time_fourierPrediction << ","
                    << total_pipeline_time << ","
                    << fps << "\n";
        metricsFile.close();

        // Accumulate total times
        total_extractForeground += time_extractForeground;
        total_preprocessForeground += time_preprocessForeground;
        total_applyCCL += time_applyCCL;
        total_hogPrediction += time_hogPrediction;
        total_fourierPrediction += time_fourierPrediction;
        total_pipeline_time += frameTimer.elapsed();

        processed_frames++;



        // Clean up temporary files
        for (size_t i = 0; i < binaryBoundingImages.size(); ++i) {
            string binaryBoundingPath = "binary_bounding_" + to_string(i) + ".png";
            remove(binaryBoundingPath.c_str());
        }
        for (size_t i = 0; i < components.size(); ++i) {
            string componentPath = "component_" + to_string(i) + ".png";
            remove(componentPath.c_str());
        }
    }


    // Save results to a CSV file
    ofstream resultsFile(predictions_path);
    // To store <filename, fourier_predictions, hog_predictions, fourier_people_count, hog_people_count, combined_prediction>
    // resultsFile << "Filename,Fourier Predictions,HOG Predictions,Fourier People Count,HOG People Count,Combined Prediction\n";
    // for (const auto& [filename, fourier_preds, hog_preds, fourier_count, hog_count, combined_pred] : results) {
    //     resultsFile << filename << ",\"" << fourier_preds << "\",\"" << hog_preds << "\"," << fourier_count << "," << hog_count << "," << combined_pred << "\n";
    // }
    // resultsFile.close();

    // To store <filename, people_count, prediction>
    resultsFile << "Filename,People Count,Prediction\n";
    for (const auto& [filename, people_count, prediction] : results) {
        resultsFile << filename << "," << people_count << "," << prediction << "\n";
    }
    resultsFile.close();
    cout << "Predictions saved to " << predictions_path << endl;


    return 0;
}