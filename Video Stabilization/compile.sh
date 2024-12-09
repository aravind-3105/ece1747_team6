#!/bin/bash

# Remove previous builds
rm -f video_stabilization.o video_stabilization_kernel.o video_stabilization

# Compile the CUDA Kernel
nvcc -c video_stabilization_kernel.cu -o video_stabilization_kernel.o
if [ $? -ne 0 ]; then
    echo "Error compiling CUDA kernel."
    exit 1
fi

# Compile the C++ Source File
g++ -c video_stabilization.cpp -o video_stabilization.o -std=c++11 -fopenmp \
    -I/usr/local/include/opencv4 -I/usr/local/cuda/include
if [ $? -ne 0 ]; then
    echo "Error compiling C++ source file."
    exit 1
fi

# Link the Object Files
g++ video_stabilization.o video_stabilization_kernel.o -o video_stabilization \
    -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_videoio \
    -lopencv_imgproc -lopencv_cudaoptflow -lopencv_cudawarping -lopencv_video \
    -lopencv_calib3d \
    -L/usr/local/cuda/lib64 -lcudart -fopenmp -lgomp
if [ $? -ne 0 ]; then
    echo "Error linking object files."
    exit 1
fi

echo "Compilation Successful. Run ./video_stabilization to execute."
