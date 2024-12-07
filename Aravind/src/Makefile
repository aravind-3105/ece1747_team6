# Compiler
CXX = g++
NVCC = nvcc

# Compiler flags
CXXFLAGS = -std=c++17 -I/usr/include/opencv4 -I/usr/local/cuda/include
NVCCFLAGS = -std=c++17 -I/usr/include/opencv4 -I/usr/local/cuda/include -diag-suppress=611

# Linker flags
LDFLAGS = -L/usr/lib \
          -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs \
          -lopencv_ml -lopencv_objdetect -lopencv_features2d -lopencv_flann \
          -L/usr/local/cuda/lib64 -lcudart -lcufft

# Combined steps
ALL_CPP_SRCS = main.cpp
ALL_CU_SRCS = fourier_descriptor.cu gmm.cu hog_descriptor.cu
ALL_CPP_OBJS = $(ALL_CPP_SRCS:.cpp=.o)
ALL_CU_OBJS = $(ALL_CU_SRCS:.cu=.o)
ALL_TARGET = all_code

# Default target builds all
all: $(ALL_TARGET)

# All build rule
$(ALL_TARGET): $(ALL_CPP_OBJS) $(ALL_CU_OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)

# Generic rules for object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -f $(ALL_CPP_OBJS) $(ALL_CU_OBJS) $(ALL_TARGET)
