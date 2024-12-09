#include "gmm_cpu.h"
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>

// Helper function to compute log Gaussian probability
float computeLogGaussian(const float* pixel_data, const float* mean, float covariance) {
    float log_prob = 0.0f;
    for (int c = 0; c < 3; ++c) {
        float diff = pixel_data[c] - mean[c];
        log_prob += -0.5f * (diff * diff) / covariance;
    }
    log_prob += -0.5f * 3 * std::log(2.0f * M_PI * covariance);
    return log_prob;
}

// Function to fit GMM
void fitGMM_CPU(const float* frames, int n_frames, int width, int height, GMMModelCPU* gmm_models) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int pixel_index = i * width + j;

            GMMModelCPU gmm = {};
            for (int k = 0; k < GMM_N_COMPONENTS; ++k) {
                for (int c = 0; c < 3; ++c) {
                    gmm.means[k][c] = 0.0f;
                }
                gmm.weights[k] = 1.0f / GMM_N_COMPONENTS;
                gmm.covariances[k] = 1.0f;
            }

            std::vector<float> responsibilities(GMM_N_COMPONENTS, 0.0f);

            for (int iter = 0; iter < GMM_MAX_ITER; ++iter) {
                float log_likelihood = 0.0f;

                for (int frame = 0; frame < n_frames; ++frame) {
                    const float* pixel_data = &frames[(frame * height * width + i * width + j) * 3];

                    std::vector<float> log_probs(GMM_N_COMPONENTS, 0.0f);
                    float max_log_prob = -INFINITY;

                    for (int k = 0; k < GMM_N_COMPONENTS; ++k) {
                        log_probs[k] = computeLogGaussian(pixel_data, gmm.means[k], gmm.covariances[k]) + std::log(gmm.weights[k]);
                        max_log_prob = std::max(max_log_prob, log_probs[k]);
                    }

                    float sum_exp = 0.0f;
                    for (int k = 0; k < GMM_N_COMPONENTS; ++k) {
                        sum_exp += std::exp(log_probs[k] - max_log_prob);
                    }
                    float log_sum_exp = max_log_prob + std::log(sum_exp);

                    for (int k = 0; k < GMM_N_COMPONENTS; ++k) {
                        responsibilities[k] += std::exp(log_probs[k] - log_sum_exp);
                    }

                    log_likelihood += log_sum_exp;
                }

                for (int k = 0; k < GMM_N_COMPONENTS; ++k) {
                    float weighted_sum[3] = {0.0f, 0.0f, 0.0f};
                    float weighted_variance = 0.0f;
                    float total_responsibility = 0.0f;

                    for (int frame = 0; frame < n_frames; ++frame) {
                        const float* pixel_data = &frames[(frame * height * width + i * width + j) * 3];

                        for (int c = 0; c < 3; ++c) {
                            weighted_sum[c] += responsibilities[k] * pixel_data[c];
                        }
                        weighted_variance += responsibilities[k] * (pixel_data[0] - gmm.means[k][0]) * (pixel_data[0] - gmm.means[k][0]);
                        total_responsibility += responsibilities[k];
                    }

                    for (int c = 0; c < 3; ++c) {
                        gmm.means[k][c] = weighted_sum[c] / total_responsibility;
                    }

                    gmm.covariances[k] = weighted_variance / total_responsibility;
                    gmm.weights[k] = total_responsibility / n_frames;
                }

                if (iter > 0 && std::fabs(log_likelihood - gmm.weights[0]) < GMM_TOL) {
                    break;
                }
            }

            gmm_models[pixel_index] = gmm;
        }
    }
}


// Function to update GMM using CPU
void fitGMMUpdate_CPU(const float* frame, int width, int height, GMMModelCPU* gmm_models) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int pixel_index = i * width + j;

            const float* pixel_data = &frame[(i * width + j) * 3];
            GMMModelCPU& gmm = gmm_models[pixel_index];

            // Update GMM components
            for (int k = 0; k < GMM_N_COMPONENTS; ++k) {
                float diff[3] = {pixel_data[0] - gmm.means[k][0],
                                 pixel_data[1] - gmm.means[k][1],
                                 pixel_data[2] - gmm.means[k][2]};
                float variance = gmm.covariances[k];
                float weight = gmm.weights[k];

                float probability = std::exp(-0.5f * (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]) / variance);
                float responsibility = weight * probability;

                for (int c = 0; c < 3; ++c) {
                    gmm.means[k][c] += responsibility * diff[c];
                }
                gmm.covariances[k] += responsibility * (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);
                gmm.weights[k] = responsibility;
            }

            // Normalize weights
            float weight_sum = 0.0f;
            for (int k = 0; k < GMM_N_COMPONENTS; ++k) {
                weight_sum += gmm.weights[k];
            }
            for (int k = 0; k < GMM_N_COMPONENTS; ++k) {
                gmm.weights[k] /= weight_sum;
            }
        }
    }
}

// Function to generate background image
void generateBackground_CPU(const GMMModelCPU* gmm_models, int width, int height, float* background) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int pixel_index = i * width + j;

            const GMMModelCPU& gmm = gmm_models[pixel_index];
            int best_component = 0;

            for (int k = 1; k < GMM_N_COMPONENTS; ++k) {
                if (gmm.weights[k] > gmm.weights[best_component]) {
                    best_component = k;
                }
            }

            for (int c = 0; c < 3; ++c) {
                background[(i * width + j) * 3 + c] = gmm.means[best_component][c];
            }
        }
    }
}
