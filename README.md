# ECE1747 Team 6 - Parallel Video Processing

## Overview
This repository implements GPU-accelerated solutions for various real-time video processing tasks using parallel programming techniques. By leveraging NVIDIA's CUDA, the project demonstrates significant performance improvements across four distinct video processing applications: **Edge Detection**, **Moving Human Detection**, **Video Illustration Effects**, and **Video Stabilization**.

## Repository Structure
```
├── Edge Detection/            # Task A: Canny Edge Detection
├── Moving Human Detection/    # Task B: Moving Object Detection
├── Video Illustrations/       # Task C: Non-photorealistic Illustration Effects
├── Video Stabilization/       # Task D: Real-Time Video Stabilization
```

Each subfolder contains:
- **Source Code:** Implementation of the task, including CUDA kernels and main logic.
- **Models:** Pre-trained models, parameter files, or normalization data (if applicable).
- **Performance Metrics:** CSV files or logs for benchmarking CPU vs GPU performance.
- **Results:** Sample outputs and visualizations.

## Tasks Overview

### 1. **Edge Detection**
Implements Canny Edge Detection using a hybrid parallel approach:
- **CPU Implementation:** Sequential, pthread-based multithreading, and MPI-based distributed parallelism.
- **GPU Implementation:** CUDA-accelerated Gaussian blur, Sobel gradients, and edge thinning for real-time performance.
- **Output:** Edge-detected images with multi-resolution processing for robustness.

### 2. **Moving Human Detection**
Detects moving humans in video streams using:
- **GPU Accelerated Modules:** Gaussian Mixture Models for background subtraction, Fourier Descriptors for shape analysis.
- **Hybrid Pipeline:** Combines GPU and CPU modules for HOG-based gradient features and SVM classification.
- **Output:** Bounding boxes and classification results for humans in video frames.

### 3. **Video Illustration Effects**
Applies non-photorealistic rendering effects (e.g., cartoon/illustration effects) to video streams:
- **Key Techniques:** Separable Gaussian blur and pixel-difference computations.
- **Output:** Illustration-style videos with smooth rendering at high resolutions.

### 4. **Video Stabilization**
Stabilizes shaky video streams for smoother playback:
- **Motion Estimation:** GPU-accelerated optical flow for motion vectors.
- **Smoothing and Frame Warping:** Combines GPU and CPU techniques to correct frame jitter.
- **Output:** Stabilized videos with high-quality rendering.

## Requirements
- **Hardware:**
  - NVIDIA GPU with CUDA support (Compute Capability 6.0+ recommended)
- **Software:**
  - CUDA Toolkit (11.0+)
  - OpenCV (4.x)
  - C++17 or later
  - Python (for preprocessing or running Jupyter notebooks)
- **Libraries:**
  - JSON for model parameters (`nlohmann::json`)
  - NVIDIA cuFFT for Fourier operations (if applicable)

## Setup Instructions
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/aravind-3105/ece1747_team6.git
   cd ECE1747_Team6
   ```

2. **Navigate to a Task Directory:**
   ```bash
   cd Moving Human Detection
   ```

3. **Compile Source Code:**
   - Use the provided instructions in each task folder to compile the source code based on the task requirements and respective README files.


4. **Run Tests:**
   - Test individual modules or end-to-end pipelines using provided test scripts based on instructuions in the README files of each task folder.

5. **Run Main Program:**
   - Refer to the README in each task folder for specific instructions.

<!-- ## Example Commands
### Edge Detection
```bash
cd Edge Detection/src
./edge_detection ../data/input_image.jpg ../results/output_image.jpg
```

### Moving Human Detection
```bash
cd Moving Human Detection/src
./human_detection ../data/input_frames/ ../results/background.png ../results/output_labeled.png
```

### Video Illustration Effects
```bash
cd Video Illustrations/src
./video_illustration ../data/input_video.mp4 ../results/output_video.mp4
```

### Video Stabilization
```bash
cd Video Stabilization/src
./video_stabilization ../data/shaky_video.mp4 ../results/stabilized_video.mp4
``` -->

## Performance Metrics
Performance benchmarks for all tasks (CPU vs GPU) are stored in the `performance_metrics/` folders within each task directory. Key observations include:
- **GPU Speedups:** Tasks achieve up to **20x performance improvements** over CPU implementations.
- **Real-Time Processing:** All tasks maintain real-time frame rates for high-resolution video streams.

## Results
- **Sample Outputs:** Refer to the `results/` folders for processed images/videos for each task.
- **Visualization:** Performance comparison plots are included in task-specific folders.

## Future Work
- Extend pipelines to additional video processing tasks (e.g., object tracking, motion magnification).
- Optimize for newer NVIDIA hardware using Tensor Cores or NVENC for video encoding.

## Contact
For questions or contributions, contact [aravind.narayanan@mail.utoronto.ca].

--- 

This README provides an organized overview for the entire project while directing users to individual task folders for more specific details. Let me know if you need further refinements!