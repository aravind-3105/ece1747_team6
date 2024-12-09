# Moving Human Detection

Name: Aravind Narayanan

## Overview
This project implements a GPU-accelerated pipeline for real-time moving human detection. It utilizes CUDA for Gaussian Mixture Models (GMM) and Fourier Descriptor (FD) computations, OpenCV for preprocessing and feature extraction, and SVM classifiers for human detection. Both CPU and GPU implementations are included for comparative performance analysis.

## Directory Structure
```
├── data/                  # Input data (images, videos, etc.)
├── include/               # Header files for GMM, FD, and HOG computations
├── models/                # Pre-trained models and normalization parameters
│   ├── hog_normalization.json
│   ├── hog_svm_model.xml
│   ├── svm_model_v3.json
│   ├── scaler_values.json
├── performance_metrics/   # Logs and performance metrics for analysis
├── results/               # Outputs such as processed videos and GIFs
├── src/                   # Main source code and Makefile
│   ├── main.cpp           # Main program entry point
│   ├── gmm.cu             # CUDA implementation of GMM
│   ├── hog_descriptor.cu  # CUDA implementation of HOG
│   ├── fourier_descriptor.cu  # CUDA implementation of Fourier Descriptor
├── tests/                 # Test scripts for components
│   ├── gmm_tests/
│   ├── hog_tests/
│   ├── fourier_tests/
```

## Data folder
Before running any test cases, please copy the data.zip file to the root directory and extract it. This will create a data folder with the required data for the test cases. The current data folder in repository is empty due to size constraints.
Link for data.zip: [data.zip](https://drive.google.com/file/d/1vo_uJjONBMsVsHEd0t0DZnUz9oTUvjCo/view?usp=sharing)


## Features
- **Background Subtraction (GMM):** Isolates moving objects from the background using GPU-accelerated GMM.
- **Fourier Descriptor:** Extracts shape-based features for rotation and scale invariance.
- **HOG Descriptor:** Extracts texture-based gradient features for human classification.
- **Connected Component Labeling (CCL):** Segments regions in the image.
- **SVM Classification:** Determines human presence using pre-trained models.

## Requirements
- **CUDA Toolkit (11.0+)**
- **OpenCV (4.x)**
- **C++17 Standard**
- **Python (optional for preprocessing and visualization)**

## Setup
1. **Clone Repository:**
   ```bash
   git clone https://github.com/aravind-3105/ece1747_team6/tree/main
   cd MovingHumanDetection
   ```

2. **Install Dependencies:**
   - CUDA Toolkit (ensure the environment has NVIDIA GPUs).
   - OpenCV libraries for image processing.

3. **Compile the Code:**
   - Navigate to the `src` directory and run:
     ```bash
     make
     ```

4. **Run Tests:**
   - To test individual components CPU vs GPU:-
   1. GMM:
     ```bash
     cd tests/gmm_tests
     make
     ./gmm_test <input_folder>
     ```
    2. HOG (test on any color image as arguement):
      ```bash
        cd tests/hog_tests
        make
        ./hog_test tests/hog_tests/test.png
    ```
    3. Fourier Descriptor (test on any binary image as arguement):
      ```bash
        cd tests/fourier_tests
        make
        ./fourier_test tests/bounding_images/binary_bounding_1.png
      ```


## Running the Program
1. **Input Data:**
   Place the images/videos in the `data/` directory.

2. **Run Main Program:**
   ```bash
   cd src
   ./human_detection <input_folder> <background_path> <foreground_path> <preprocessed_path> <output_labeled>
   ```

3. **Example Command:**
   ```bash
   ./human_detection ../data/OSUdata/ ../data/background.png ../results/foreground.png ../results/preprocessed.png ../results/labeled_output.png
   ```
   <!-- or to run default test case after copying the data.zip as a data folder in the root directory> -->
   Or to run the default test case (after copying the data.zip as a data folder in the root directory):

   ```bash
   ./human_detection
   ```

## Key Functions
1. **Background Subtraction (GMM):**
   - Located in `gmm.cu`.
   - Generates a background image from input frames.

2. **Fourier Descriptor:**
   - Located in `fourier_descriptor.cu`.
   - Computes scale and rotation-invariant shape features.

3. **HOG Descriptor:**
   - Located in `hog_descriptor.cu`.
   - Extracts texture and gradient features.

4. **Connected Component Labeling:**
   - Implements labeling of segmented regions and extracts bounding boxes.

5. **SVM Classification:**
   - Uses pre-trained models to classify human presence.

6. **Performance Analysis:**
   - Logs CPU vs GPU execution times for each step.


## SVM Training
- About the dataset:

   - **Training Data:**
      - The SVM model is trained on a dataset of 1500 person images and 2000 non-person images obtained from various databases like CUHK01, MIT, INRIA, MPEG7, and PEDCUT as per the reference paper.
      - The dataset is split into training, validation, and test sets.

   - **Trained Models:**
      - The SVM model is trained using the HOG features extracted from the dataset.
      - The model is saved in `models/hog_svm_model.xml`.
      - The normalization parameters are saved in `models/hog_normalization.json`.
      - The script uses the `hog_descriptor.cpp` file to extract HOG features.
      - The model is saved in `models/svm_model_v3.json`.
      - The normalization parameters are saved in `models/scaler_values.json`.
      - The script uses the `fourier_descriptor.cpp` file to extract Fourier Descriptor features.

## Performance
- **Metrics:** CPU vs GPU execution times are logged in `performance_metrics/`.
- **Visualization:** Performance graphs and results are stored in `results/`.

## Jupyter Notebook
To run the pipeline in Google Colab, use the provided notebook `movingHumanDetection.ipynb` (PDF backup in `movingHumanDetection.pdf`). It includes:
- Code snippets for running each step.
- Steps to mount Google Drive and load input/output.
- 

<!-- ## Example Output
- **Foreground Detection:**
  ![Foreground](example_foreground_output.png)
- **Labeled Bounding Boxes:**
  ![Labeled Output](example_labeled_output.png) -->

## Contributions
- Main program: **main.cpp**
- CUDA modules: `gmm.cu`, `hog_descriptor.cu`, `fourier_descriptor.cu`

## Future Enhancements
- **Optimization:** Integrate Tensor Cores for enhanced GPU performance.
- **Scalability:** Adapt pipeline for edge devices with limited compute power.

## Contact
For queries, contact [aravind.narayanan@mail.utoronto.ca].

---
