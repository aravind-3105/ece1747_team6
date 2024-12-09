#ifndef GMM_H
#define GMM_H

#include <vector>

// Constants for GMM

#define N_COMPONENTS 5
#define MAX_ITER 100
#define TOL 1e-3

// Structure for GMM model
struct GMMModel {
    float means[N_COMPONENTS][3];     // Means for RGB
    float weights[N_COMPONENTS];     // Weights of components
    float covariances[N_COMPONENTS]; // Covariances for components
};

// Use extern "C" to ensure C linkage
#ifdef __cplusplus
extern "C" {
#endif

// Initial fitting of GMM using all frames (batch processing)
void fitGMM(const float* frames, int n_frames, int width, int height, GMMModel* gmm_models, int n_jobs);

// Dynamic update of GMM for a single frame
void fitGMMUpdate(const float* frame, int width, int height, GMMModel* gmm_models);

// Generate background image from the current GMM models
void generateBackground(const GMMModel* gmm_models, int width, int height, float* background);

// Wrapper function for background subtraction
void subtractBackground(const float* frame, const float* background, float* foreground, int width, int height);

#ifdef __cplusplus
}
#endif

#endif