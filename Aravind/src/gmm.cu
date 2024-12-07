#include <cuda_runtime.h>
#include <cstdio>
#include "gmm.h"

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            printf("CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)




__global__ void fitGMMKernel(const float* frames, int n_frames, int width, int height, GMMModel* gmm_models) {
    // Pixel location
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= height || j >= width) return;

    int pixel_index = i * width + j;

    // Debug: Launch confirmation for the first pixel
    if (i == 0 && j == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
        printf("Kernel execution started. Processing pixel (0, 0).\n");
    }

    // Allocate memory for pixel data across frames
    float pixel_data[3]; // For a single RGB frame

    // Initialize GMM model
    GMMModel gmm = {};
    for (int k = 0; k < N_COMPONENTS; k++) {
        for (int c = 0; c < 3; c++) {
            gmm.means[k][c] = 0.0f;
        }
        gmm.weights[k] = 1.0f / N_COMPONENTS;  // Equal weights
        gmm.covariances[k] = 1.0f;            // Initialize covariance
    }

    // Temporary storage for responsibilities
    float responsibilities[N_COMPONENTS];

    // Fit GMM (Expectation-Maximization loop)
    for (int iter = 0; iter < MAX_ITER; iter++) {
        float log_likelihood = 0.0f;

        // E-Step: Compute responsibilities
        for (int frame = 0; frame < n_frames; frame++) {
            for (int c = 0; c < 3; c++) {
                pixel_data[c] = frames[(frame * height * width + i * width + j) * 3 + c];
            }

            // Calculate log probabilities and responsibilities
            float max_log_prob = -INFINITY;
            float log_probs[N_COMPONENTS];

            for (int k = 0; k < N_COMPONENTS; k++) {
                float diff[3];
                float variance = gmm.covariances[k];
                float log_prob = 0.0f;

                // Compute log Gaussian probability for this component
                for (int c = 0; c < 3; c++) {
                    diff[c] = pixel_data[c] - gmm.means[k][c];
                    log_prob += -0.5f * (diff[c] * diff[c]) / variance;
                }
                log_prob += -0.5f * 3 * log(2.0f * M_PI * variance);
                log_prob += log(gmm.weights[k]);

                log_probs[k] = log_prob;
                max_log_prob = fmaxf(max_log_prob, log_prob);
            }

            // Compute log-sum-exp for normalization
            float sum_exp = 0.0f;
            for (int k = 0; k < N_COMPONENTS; k++) {
                sum_exp += expf(log_probs[k] - max_log_prob);
            }
            float log_sum_exp = max_log_prob + logf(sum_exp);

            // Update responsibilities and log-likelihood
            for (int k = 0; k < N_COMPONENTS; k++) {
                responsibilities[k] = expf(log_probs[k] - log_sum_exp);
            }
            log_likelihood += log_sum_exp;
        }

        // M-Step: Update means, weights, and covariances
        float total_responsibility[N_COMPONENTS] = {0.0f};

        for (int k = 0; k < N_COMPONENTS; k++) {
            float weighted_sum[3] = {0.0f, 0.0f, 0.0f};
            float weighted_variance = 0.0f;

            for (int frame = 0; frame < n_frames; frame++) {
                for (int c = 0; c < 3; c++) {
                    pixel_data[c] = frames[(frame * height * width + i * width + j) * 3 + c];
                }

                for (int c = 0; c < 3; c++) {
                    weighted_sum[c] += responsibilities[k] * pixel_data[c];
                }
                weighted_variance += responsibilities[k] * (pixel_data[0] - gmm.means[k][0]) * (pixel_data[0] - gmm.means[k][0]);
                total_responsibility[k] += responsibilities[k];
            }

            // Update means
            for (int c = 0; c < 3; c++) {
                gmm.means[k][c] = weighted_sum[c] / total_responsibility[k];
            }

            // Update covariance (diagonal, shared for simplicity)
            gmm.covariances[k] = weighted_variance / total_responsibility[k];

            // Update weights
            gmm.weights[k] = total_responsibility[k] / n_frames;
        }

        // Check for convergence (optional, log-likelihood difference)
        if (iter > 0 && fabsf(log_likelihood - gmm_models[pixel_index].weights[0]) < TOL) {
            break;
        }
    }

    // Store the fitted model
    gmm_models[pixel_index] = gmm;
}

__global__ void fitGMMUpdateKernel(const float* frame, int width, int height, GMMModel* gmm_models) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= height || j >= width) return;

    int pixel_index = i * width + j;

    float pixel_data[3];
    for (int c = 0; c < 3; c++) {
        pixel_data[c] = frame[(i * width + j) * 3 + c];
    }

    // Update GMM models using pixel data
    for (int k = 0; k < N_COMPONENTS; k++) {
        float diff[3];
        float variance = gmm_models[pixel_index].covariances[k];
        float weight = gmm_models[pixel_index].weights[k];

        for (int c = 0; c < 3; c++) {
            diff[c] = pixel_data[c] - gmm_models[pixel_index].means[k][c];
        }

        float probability = expf(-0.5f * (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]) / variance);
        float responsibility = weight * probability;

        // Update means, variances, and weights
        for (int c = 0; c < 3; c++) {
            gmm_models[pixel_index].means[k][c] += responsibility * diff[c];
        }
        gmm_models[pixel_index].covariances[k] += responsibility * (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);
        gmm_models[pixel_index].weights[k] = responsibility;
    }

    // Normalize weights
    float weight_sum = 0.0f;
    for (int k = 0; k < N_COMPONENTS; k++) {
        weight_sum += gmm_models[pixel_index].weights[k];
    }
    for (int k = 0; k < N_COMPONENTS; k++) {
        gmm_models[pixel_index].weights[k] /= weight_sum;
    }
}



// Wrapper for CUDA kernel
extern "C" void fitGMM(const float* frames, int n_frames, int width, int height, GMMModel* gmm_models, int n_jobs) {
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    fitGMMKernel<<<gridDim, blockDim>>>(frames, n_frames, width, height, gmm_models);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in fitGMM kernel launch: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();
}

extern "C" void fitGMMUpdate(const float* frame, int width, int height, GMMModel* gmm_models) {
    dim3 blockDim(32, 32);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    fitGMMUpdateKernel<<<gridDim, blockDim>>>(frame, width, height, gmm_models);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in fitGMMUpdate kernel launch: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();
}



// CUDA kernel to generate background image
__global__ void generateBackgroundKernel(const GMMModel* gmm_models, int width, int height, float* background) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= height || j >= width) return;

    int pixel_index = i * width + j;

    // Debug: Launch confirmation for the first pixel
    if (i == 0 && j == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
        printf("Background generation kernel started. Processing pixel (0, 0).\n");
    }

    // Use the Gaussian component with the highest weight
    int best_component = 0;
    for (int k = 1; k < N_COMPONENTS; k++) {
        if (gmm_models[pixel_index].weights[k] > gmm_models[pixel_index].weights[best_component]) {
            best_component = k;
        }
    }

    // Assign the mean of the best component to the background
    for (int c = 0; c < 3; c++) {
        background[(i * width + j) * 3 + c] = gmm_models[pixel_index].means[best_component][c];
    }
}

// Wrapper for background generation
extern "C" void generateBackground(const GMMModel* gmm_models, int width, int height, float* background) {
    dim3 blockDim(32, 32);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    generateBackgroundKernel<<<gridDim, blockDim>>>(gmm_models, width, height, background);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in generateBackground kernel launch: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();
}
