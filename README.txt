Rask D: Real-time GPU-Accelerated Video Stabilization

This task involves using CUDA-accelerated optical flow and frame warping to stabilize a given input video in real-time. The project utilizes CUDA kernels for computation and OpenCV for video handling.

System Requirements
GPU: An NVIDIA GPU with CUDA capability.
Drivers: NVIDIA drivers compatible with CUDA (e.g., CUDA 11.x or later).
Docker: Docker Engine installed on your system.
Host OS: Linux (Ubuntu 20.04 or newer recommended) or Windows with WSL2 Docker support.
Architecture Needed:

A 64-bit architecture system is required.
At least one NVIDIA GPU and corresponding CUDA drivers must be present.

The code and image have been tested on systems with NVIDIA GeForce RTX 4070 and CUDA toolkit 11.8+.

Files in video_processing Folder:
video_processing_image.tar: The prebuilt Docker image containing all necessary dependencies (CUDA, OpenCV) and the code.

Source Code Files:
video_stabilization.cpp (Main host code: motion estimation and smoothing)
video_stabilization_kernel.cu (CUDA kernel code for frame warping)
compile.sh (Build script for compiling the application)
input_video.mp4 (Example input video)
README.txt (This file)

Output file (stabilized_output.mp4) will be generated after running the application.

Steps to Run the Docker Image and Project

Navigate to the video_processing Directory

bash
cd /path/to/video_processing

Ensure video_processing_image.tar and all source code files are present in this directory.

Load the Docker Image

Load the Docker image from the tar file:

bash
docker load -i video_processing_image.tar
After this command, the image will be available locally. You can verify with:

bash
docker images
Run the Docker Container with Folder Linking

Run the container, mounting the current video_processing directory into /workspace inside the container. Also, allow GPU access:

bash

docker run --rm -it --gpus all -v $(pwd):/workspace video_processing_image:latest /bin/bash
Explanation:

--rm: Removes the container when you exit.
-it: Interactive terminal.
--gpus all: Grants GPU access to the container.
-v $(pwd):/workspace: Mounts the current directory to /workspace inside the container.
video_processing_image:latest: The name of the image you loaded.
/bin/bash: Starts a bash shell inside the container.
Compile the Code Inside the Container

Once inside the container, change to the /workspace directory:

bash
cd /workspace
Convert the build script to Unix format and make it executable:

bash
dos2unix compile.sh
chmod +x compile.sh
Now compile the code:

bash
./compile.sh
This will compile the CUDA kernels and host code, producing video_stabilization.

Run the Stabilization Application

After compilation, run:

bash
./video_stabilization
The program will process the input_video.mp4 and produce a stabilized output file named stabilized_output.mp4 in the same directory (/workspace).

Verifying Output
Once execution completes, use ls to confirm the presence of stabilized_output.mp4.

You can copy this file back to your host system if needed or play it directly inside the container using installed tools.

Additional Notes
If you want to profile GPU usage, consider using tools like Nsight Systems or Nsight Compute. These are not included by default but can be integrated if needed.
If you change the input video or code, simply re-run compile.sh and ./video_stabilization