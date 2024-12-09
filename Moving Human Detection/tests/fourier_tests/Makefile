# Compiler
CXX = g++
NVCC = nvcc

# Compiler flags
CXXFLAGS = -std=c++17 -I. -I/usr/include/opencv4 -I/usr/local/cuda/include
NVCCFLAGS = -std=c++17 -I. -I/usr/include/opencv4 -diag-suppress=611

# Linker flags
LDFLAGS = -L/usr/lib \
          -L/usr/local/cuda/lib64 \
          -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs \
          -lcudart -lcufft

# Source and object files
CPU_SRCS = fourier_descriptor_test.cpp fourier_descriptor_cpu.cpp
GPU_SRCS = fourier_descriptor.cu
CPU_OBJS = $(CPU_SRCS:.cpp=.o)
GPU_OBJS = $(GPU_SRCS:.cu=.o)
TARGET = fourier_descriptor_test

# Default target
all: $(TARGET)

# Link all object files into the final executable
$(TARGET): $(CPU_OBJS) $(GPU_OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)

# Compile CPU source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile GPU source files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -f $(CPU_OBJS) $(GPU_OBJS) $(TARGET)