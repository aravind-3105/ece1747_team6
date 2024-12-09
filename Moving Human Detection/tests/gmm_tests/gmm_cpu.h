#ifndef GMM_CPU_H
#define GMM_CPU_H

#include <vector>

// Constants for GMM
constexpr int GMM_N_COMPONENTS = 5;
constexpr int GMM_MAX_ITER = 100; 
constexpr float GMM_TOL = 1e-3;  

// Structure for GMM model (CPU version)
struct GMMModelCPU {
    float means[GMM_N_COMPONENTS][3];     // Means for RGB
    float weights[GMM_N_COMPONENTS];     // Weights of components
    float covariances[GMM_N_COMPONENTS]; // Covariances for components
};

// Function declarations
void fitGMM_CPU(const float* frames, int n_frames, int width, int height, GMMModelCPU* gmm_models);
void fitGMMUpdate_CPU(const float* frame, int width, int height, GMMModelCPU* gmm_models);
void generateBackground_CPU(const GMMModelCPU* gmm_models, int width, int height, float* background);

#endif // GMM_CPU_H
