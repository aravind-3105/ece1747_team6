// video_stabilization.cpp

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <chrono>


// Extern declaration for CUDA kernel launcher
void warpFrameKernelLauncher(const uchar3* input, uchar3* output, int width, int height, float motionX, float motionY, cudaStream_t stream);

// Function to check CUDA errors
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort=true){
    if (code != cudaSuccess){
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

int main(int argc, char** argv) {

    int device;
cudaGetDevice(&device);
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, device);
printf("Using device %d: %s\n", device, prop.name);

    // Initialize video capture
    cv::VideoCapture cap;

    if (argc > 1) {
        // Open the video file provided as a command-line argument
        cap.open(argv[1]);
        std::cout << "Opening video file: " << argv[1] << std::endl;
    } else {
        // Default video file
        std::string default_video ="/workspace/input_video.mp4";
;
        cap.open(default_video);
        std::cout << "Opening default video file: " << default_video << std::endl;
    }

    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open video source." << std::endl;
        return -1;
    }

    // Get frame properties
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0.0) fps = 30.0; // Default to 30 if unable to get FPS

    std::cout << "Frame Size: " << frame_width << "x" << frame_height << ", FPS: " << fps << std::endl;

    // Initialize variables for stabilization
    cv::Mat prev_gray, curr_gray;
    cv::Mat prev_frame, curr_frame;

    // Read the first frame
    if (!cap.read(prev_frame)) {
        std::cerr << "Error: Unable to read the first frame." << std::endl;
        return -1;
    }

    // Convert to grayscale
    cv::cvtColor(prev_frame, prev_gray, cv::COLOR_BGR2GRAY);

    // Accumulated transformation
    double cumulative_trans_x = 0.0;
    double cumulative_trans_y = 0.0;

    // Smoothing parameters
    const double smoothing_factor = 0.95; // Adjusted for better smoothing
    double smoothed_trans_x = 0.0;
    double smoothed_trans_y = 0.0;
    const float max_translation = 300.0f; // Reduced to prevent out-of-bounds and minimize black borders

    // Define cropping parameters based on max_translation
    int crop_x = static_cast<int>(max_translation);
    int crop_y = static_cast<int>(max_translation);
    int crop_width = frame_width - 2 * crop_x;
    int crop_height = frame_height - 2 * crop_y;

    // Optional scaling factor to compensate for cropping
    const double scaling_factor = 1.05; // 5% zoom-in

    // Initialize VideoWriter
    std::string output_video = "/workspace/stabilized_output.mp4";
    cv::VideoWriter writer(
    output_video,
    cv::VideoWriter::fourcc('X', 'V', 'I', 'D'),  // Use XVID codec
    fps,
    cv::Size(frame_width, frame_height)
);


    if (!writer.isOpened()) {
        std::cerr << "Error: Unable to open VideoWriter." << std::endl;
        return -1;
    }

    // Allocate device memory for input and output frames
    size_t frame_size = frame_width * frame_height * sizeof(uchar3);
    uchar3* d_input = nullptr;
    uchar3* d_output = nullptr;
    cudaCheckError(cudaMalloc(&d_input, frame_size));
    cudaCheckError(cudaMalloc(&d_output, frame_size));

    // Initialize vectors to store metrics
    std::vector<std::tuple<int, float, float, double, double, double, double>> translation_metrics;
    std::vector<std::pair<int, double>> processing_time_metrics;
    std::vector<std::tuple<int, int, int>> feature_tracking_metrics;

    int frame_number = 1; // Starting from frame 1

    // Create CUDA streams
    cudaStream_t stream1, stream2;
    cudaCheckError(cudaStreamCreate(&stream1));
    cudaCheckError(cudaStreamCreate(&stream2));

    // Main processing loop
    while (true) {
        // Start timing for processing this frame
        auto start_time = std::chrono::high_resolution_clock::now();

        // Capture the current frame
        if (!cap.read(curr_frame)) {
            std::cout << "End of video stream." << std::endl;
            break;
        }

        // Convert to grayscale
        cv::cvtColor(curr_frame, curr_gray, cv::COLOR_BGR2GRAY);

        // Feature detection and tracking using Optical Flow (Good Features to Track + Lucas-Kanade)
        std::vector<cv::Point2f> prev_pts, curr_pts;
        std::vector<uchar> status;
        std::vector<float> err;

        // Detect features in the previous frame
        cv::goodFeaturesToTrack(prev_gray, prev_pts, 200, 0.01, 30);
        int features_detected = static_cast<int>(prev_pts.size());

        if (prev_pts.empty()) {
            std::cerr << "Warning: No features found in the previous frame." << std::endl;
            prev_frame = curr_frame.clone();
            prev_gray = curr_gray.clone();
            // Log metrics with zero tracked features
            feature_tracking_metrics.emplace_back(frame_number, features_detected, 0);
            // Log translation metrics with zero translations
            translation_metrics.emplace_back(frame_number, 0.0f, 0.0f, cumulative_trans_x, cumulative_trans_y, smoothed_trans_x, smoothed_trans_y);
            // Log processing time as negligible
            processing_time_metrics.emplace_back(frame_number, 0.0);
            frame_number++;
            continue;
        }

        // Calculate optical flow (i.e., track feature points)
        cv::calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, curr_pts, status, err);

        // Filter out bad points
        std::vector<cv::Point2f> good_prev_pts, good_curr_pts;
        for (size_t i = 0; i < status.size(); ++i) {
            if (status[i]) {
                good_prev_pts.push_back(prev_pts[i]);
                good_curr_pts.push_back(curr_pts[i]);
            }
        }
        int features_tracked = static_cast<int>(good_prev_pts.size());

        if (good_prev_pts.size() < 10) {
            std::cerr << "Warning: Not enough good feature points." << std::endl;
            prev_frame = curr_frame.clone();
            prev_gray = curr_gray.clone();
            // Log metrics
            feature_tracking_metrics.emplace_back(frame_number, features_detected, features_tracked);
            translation_metrics.emplace_back(frame_number, 0.0f, 0.0f, cumulative_trans_x, cumulative_trans_y, smoothed_trans_x, smoothed_trans_y);
            processing_time_metrics.emplace_back(frame_number, 0.0);
            frame_number++;
            continue;
        }

        // Estimate affine transform between frames
        cv::Mat transform = cv::estimateAffine2D(good_prev_pts, good_curr_pts);

        if (transform.empty()) {
            std::cerr << "Warning: Unable to estimate transform." << std::endl;
            prev_frame = curr_frame.clone();
            prev_gray = curr_gray.clone();
            // Log metrics
            feature_tracking_metrics.emplace_back(frame_number, features_detected, features_tracked);
            translation_metrics.emplace_back(frame_number, 0.0f, 0.0f, cumulative_trans_x, cumulative_trans_y, smoothed_trans_x, smoothed_trans_y);
            processing_time_metrics.emplace_back(frame_number, 0.0);
            frame_number++;
            continue;
        }

        // Extract translation components
        double dx = transform.at<double>(0, 2);
        double dy = transform.at<double>(1, 2);

        // Accumulate the translations
        cumulative_trans_x += dx;
        cumulative_trans_y += dy;

        // Apply smoothing to the accumulated translations
        smoothed_trans_x = smoothing_factor * smoothed_trans_x + (1.0 - smoothing_factor) * cumulative_trans_x;
        smoothed_trans_y = smoothing_factor * smoothed_trans_y + (1.0 - smoothing_factor) * cumulative_trans_y;

        // Clamp the smoothed translations to prevent excessive shifts
        if (smoothed_trans_x > max_translation) smoothed_trans_x = max_translation;
        if (smoothed_trans_x < -max_translation) smoothed_trans_x = -max_translation;
        if (smoothed_trans_y > max_translation) smoothed_trans_y = max_translation;
        if (smoothed_trans_y < -max_translation) smoothed_trans_y = -max_translation;

        // For stabilization, apply the inverse of the smoothed cumulative motion
        float stabilize_x = -static_cast<float>(smoothed_trans_x);
        float stabilize_y = -static_cast<float>(smoothed_trans_y);

        // Log the translation values for debugging
        std::cout << "Frame: " << frame_number
                  << " | dx: " << dx 
                  << " | dy: " << dy 
                  << " | Cumulative: (" << cumulative_trans_x << ", " << cumulative_trans_y << ")"
                  << " | Smoothed: (" << smoothed_trans_x << ", " << smoothed_trans_y << ")"
                  << std::endl;

        // Prepare the input frame data (convert from BGR to uchar3)
        cv::Mat input_frame;
        curr_frame.convertTo(input_frame, CV_8UC3);
        uchar3* h_input = reinterpret_cast<uchar3*>(input_frame.data);

        // Allocate a host buffer for the output
        std::vector<uchar3> h_output(frame_width * frame_height);

        // Determine current stream based on frame number for alternating
        cudaStream_t current_stream = (frame_number % 2 == 0) ? stream1 : stream2;

        // Copy input frame to device asynchronously
        cudaCheckError(cudaMemcpyAsync(d_input, h_input, frame_size, cudaMemcpyHostToDevice, current_stream));

     printf("Launching CUDA Kernel...\n"); // Debug message before kernel launch
warpFrameKernelLauncher(d_input, d_output, frame_width, frame_height, stabilize_x, stabilize_y, current_stream);

// Check for kernel launch errors
cudaError_t cuda_err = cudaGetLastError(); // Rename to avoid conflicts
if (cuda_err != cudaSuccess) {
    fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(cuda_err)); // Properly print the error string
} else {
    printf("Kernel launched successfully.\n"); // Confirm successful launch
}

// Ensure the kernel completes execution
cudaDeviceSynchronize();

        // Copy the output frame back to host asynchronously
        cudaCheckError(cudaMemcpyAsync(h_output.data(), d_output, frame_size, cudaMemcpyDeviceToHost, current_stream));

        // Synchronize the current stream to ensure data is copied before using it
        cudaCheckError(cudaStreamSynchronize(current_stream));

        // Create an OpenCV Mat from the output data
        cv::Mat stabilized_frame(frame_height, frame_width, CV_8UC3, h_output.data());

        // Define the ROI based on max_translation
        cv::Rect roi(crop_x, crop_y, crop_width, crop_height);
        cv::Mat cropped_frame = stabilized_frame(roi);

        // (Optional) Apply scaling to compensate for cropping
        cv::Mat final_frame;
        cv::resize(cropped_frame, final_frame, cv::Size(frame_width, frame_height), 0, 0, cv::INTER_LINEAR);

        // Write the final frame to the output video
        writer.write(final_frame);

        // Stop timing for processing this frame
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> processing_time = end_time - start_time;

        // Log processing time
        processing_time_metrics.emplace_back(frame_number, processing_time.count());

        // Log feature tracking metrics
        feature_tracking_metrics.emplace_back(frame_number, features_detected, features_tracked);

        // Log translation metrics
        translation_metrics.emplace_back(frame_number, stabilize_x, stabilize_y, cumulative_trans_x, cumulative_trans_y, smoothed_trans_x, smoothed_trans_y);

        // Update previous frame and grayscale image
        prev_frame = curr_frame.clone();
        prev_gray = curr_gray.clone();

        frame_number++;
    }

    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    cap.release();
    writer.release();

    // Destroy CUDA streams
    cudaCheckError(cudaStreamDestroy(stream1));
    cudaCheckError(cudaStreamDestroy(stream2));

    // Write translation metrics to CSV
    std::ofstream translation_file("translation_metrics.csv");
    if (translation_file.is_open()) {
        translation_file << "Frame,stabilize_x,stabilize_y,cumulative_dx,cumulative_dy,smoothed_dx,smoothed_dy\n";
        for (const auto& row : translation_metrics) {
            translation_file << std::get<0>(row) << ","
                             << std::get<1>(row) << ","
                             << std::get<2>(row) << ","
                             << std::get<3>(row) << ","
                             << std::get<4>(row) << ","
                             << std::get<5>(row) << ","
                             << std::get<6>(row) << "\n";
        }
        translation_file.close();
        std::cout << "Translation metrics saved to translation_metrics.csv" << std::endl;
    } else {
        std::cerr << "Unable to open translation_metrics.csv for writing." << std::endl;
    }

    // Write processing time metrics to CSV
    std::ofstream processing_time_file("processing_time.csv");
    if (processing_time_file.is_open()) {
        processing_time_file << "Frame,Processing_Time_ms\n";
        for (const auto& row : processing_time_metrics) {
            processing_time_file << row.first << "," << row.second << "\n";
        }
        processing_time_file.close();
        std::cout << "Processing time metrics saved to processing_time.csv" << std::endl;
    } else {
        std::cerr << "Unable to open processing_time.csv for writing." << std::endl;
    }

    // Write feature tracking metrics to CSV
    std::ofstream feature_tracking_file("feature_tracking.csv");
    if (feature_tracking_file.is_open()) {
        feature_tracking_file << "Frame,Features_Detected,Features_Tracked\n";
        for (const auto& row : feature_tracking_metrics) {
            feature_tracking_file << std::get<0>(row) << ","
                                   << std::get<1>(row) << ","
                                   << std::get<2>(row) << "\n";
        }
        feature_tracking_file.close();
        std::cout << "Feature tracking metrics saved to feature_tracking.csv" << std::endl;
    } else {
        std::cerr << "Unable to open feature_tracking.csv for writing." << std::endl;
    }

    std::cout << "Stabilized video saved to: " << output_video << std::endl;
    std::cout << "Available OpenCV Codecs:" << std::endl;
std::cout << "FOURCC MP4V: " << cv::VideoWriter::fourcc('m', 'p', '4', 'v') << std::endl;
std::cout << "FOURCC AVC1: " << cv::VideoWriter::fourcc('a', 'v', 'c', '1') << std::endl;
std::cout << "FOURCC XVID: " << cv::VideoWriter::fourcc('X', 'V', 'I', 'D') << std::endl;
std::cout << "FOURCC MJPG: " << cv::VideoWriter::fourcc('M', 'J', 'P', 'G') << std::endl;

    return 0;
}
