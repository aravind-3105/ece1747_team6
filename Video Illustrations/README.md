
# Real-Time Video Illustrations using CUDA

This repository contains implementations of Real-time Video processing:

0. **Google Colab** 

- This project requires a system which is well equiped with GPU as well as Camera for Video Capturing. 
- Due to lack of such specific resources, We have implemented this Project using Google Colab.
- Have included 'Google Colab Video Illustration' Folder consisting of .ipynb file which captures the Video and stores as test.mp4
- A python program in JavaScript is included which will capture the Video in Run Time via Browser.
- This video recording is stored as 'test.mp4' which is included in the CUDA file as an input file for further processing.
- One can directly download the entire .ipynb file included in 'Google Colab Video Illustration' and upload it as a notebook in respective Google Colab.
- Select the Runtime type to T4 so that we are able to access the GPU on Google Colab
- Run all the cells
- It shall ask for Camera permissions. Please allow
- After capturing the Video it shall create a 'test.mp4' video file.
- Which acts as an input for our CUDA implementation.
- After application of Gaussian filters - 'output_video.avi' output file shall be produced.
- This .ipynb file consists of the required commands for compilation of CUDA implementation as well.

1. **CUDA Implementation** (`video_illustration_full.cu`)

## Features

### `video_illustration_full.cu`
- Implements CUDA for Video Processing
- Exploits GPU acceleration for high-performance Video conversion into Illustration.
- Efficient for large-scale and high resolution Video processing.

## Requirements

### Common
- C++ compiler with C++11 or newer standard support.
- OpenCV library (version 4.0 or higher).
- CUDA Toolkit installed (for CUDA implementation).
- MPI runtime environment (e.g., OpenMPI).

## Build and Run

### Compiling the CUDA Implementation
```bash
nvcc video_illustration_full.cu -o video_illustration_full `pkg-config --cflags --libs opencv4`
```

### Running the CUDA Implementation
```bash
./video_illustration_full
```

## Profiling
After execution, output video shall be output video consisting of applied Gaussian filters.

## Outputs
- `output_video.avi`: Result from CUDA mode.

## Contributions
Contributions are welcome! Feel free to open issues or submit pull requests to improve the code.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
