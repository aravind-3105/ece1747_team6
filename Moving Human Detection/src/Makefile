# Compiler
CXX = g++
NVCC = nvcc

# Compiler flags
CXXFLAGS = -std=c++17 -I/usr/include/opencv4 -I/usr/local/cuda/include -I../include
NVCCFLAGS = -std=c++17 -I/usr/include/opencv4 -I/usr/local/cuda/include -I../include -diag-suppress=611

# Linker flags
LDFLAGS = -L/usr/lib \
          -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs \
          -lopencv_ml -lopencv_objdetect -lopencv_features2d -lopencv_flann \
          -L/usr/local/cuda/lib64 -lcudart -lcufft

# Source and object files
ALL_CPP_SRCS = $(wildcard *.cpp)
ALL_CU_SRCS = $(wildcard *.cu)
ALL_CPP_OBJS = $(ALL_CPP_SRCS:%.cpp=build/%.o)
ALL_CU_OBJS = $(ALL_CU_SRCS:%.cu=build/%.o)

# Target executable
ALL_TARGET = human_detection

# Default target builds all
all: $(ALL_TARGET)

# Linking rule
$(ALL_TARGET): $(ALL_CPP_OBJS) $(ALL_CU_OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)

# Rule to compile C++ source files
build/%.o: %.cpp | build
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule to compile CUDA source files
build/%.o: %.cu | build
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Create build directory if it doesn't exist
build:
	mkdir -p build

# Clean rule
clean:
	rm -rf build $(ALL_TARGET)
