
# Image Processing with Multi-threading and CUDA

This repository contains two implementations of Gaussian Edge Detection:

1. **CUDA Implementation** (`cuda_gaussian_edge_detection.cu`)
2. **Multi-threaded and MPI Implementation** (`image_processor.cpp`)

## Features

### `cuda_gaussian_edge_detection.cu`
- Implements Gaussian Edge Detection using CUDA.
- Exploits GPU acceleration for high-performance image processing.
- **Modes:**
  - **Single**: Processes the image using a single Canny Edge Filtering.
  - **Multi**: Processes the images using multi gaussian canny edge filtering.
- Efficient for large-scale image datasets.

### `image_processor.cpp`
- Implements image edge detection with the following methods:
  - **Sequence mode**: Single-threaded execution.
  - **Pthread mode**: Multi-threaded execution using POSIX threads.
  - **MPI mode**: Parallel execution using the MPI framework.
- Profiling tools to measure the performance of each processing stage.
- Supports multiple image processing techniques:
  - Gaussian Blur
  - Sobel Gradient Calculation
  - Non-Max Suppression
  - Double Thresholding
  - Hysteresis

## Requirements

### Common
- C++ compiler with C++11 or newer standard support.
- OpenCV library (version 4.0 or higher).
- CUDA Toolkit installed (for CUDA implementation).
- MPI runtime environment (e.g., OpenMPI).

## Build and Run

### Compiling the CUDA Implementation
```bash
nvcc cuda_gaussian_edge_detection.cu -o cuda_gaussian_edge_detection -lcuda -lcudart
```

### Compiling the C++ Implementation
```bash
g++ image_processor.cpp -o image_processor `pkg-config --cflags --libs opencv4` -lpthread -lmpi -std=c++11
```

### Running the CUDA Implementation
#### Single-threaded Mode:
```bash
./cuda_gaussian_edge_detection <input_image> single
```

#### Multi-threaded Mode:
```bash
./cuda_gaussian_edge_detection <input_image> multi 0.5,1,1.5
```

### Running the C++ Implementation
#### Sequence Mode:
```bash
./image_processor <image_path> 1
```

#### Pthread Mode:
```bash
./image_processor <image_path> 2 <num_threads>
```

#### MPI Mode:
```bash
mpirun -np <num_processes> ./image_processor <image_path> 3
```

## Profiling
After execution, both implementations output a detailed profiling report for each processing stage.

## Outputs
- `canny_edges.png`: Result from CUDA mode.
- `output_sequence.png`: Result from sequence mode.
- `output_pthread.png`: Result from pthread mode.
- `output_mpi.png`: Result from MPI mode.

## Contributions
Contributions are welcome! Feel free to open issues or submit pull requests to improve the code.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
