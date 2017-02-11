#CC = icc -O3 -g -Wall -std=c99 -openmp -mkl
#CXX = icpc -O3 -g -Wall -std=c++0x -Wno-deprecated -openmp -mkl

CUDA_PATH = /home/xiaocen/Software/cuda/cuda-8.0
CC = gcc -O3 -g -Wall -std=c99 -fopenmp
CXX = g++ -O3 -g -Wall -std=c++0x -Wno-deprecated -fopenmp

NVCC = nvcc -ccbin g++ -Xcompiler -fopenmp

CUDA_INCLUDE = $(CUDA_PATH)/include
CUDA_COMMON_INCLUDE = /home/xiaocen/Software/cuda/samples/NVIDIA_CUDA-8.0_Samples/common/inc

OPENCV_PATH = /home/xiaocen/Software/opencv
OPENCV_INCLUDE = $(OPENCV_PATH)/include

INCLUDES = -I$(CUDA_COMMON_INCLUDE) -I$(CUDA_INCLUDE) -I$(OPENCV_INCLUDE)

GENCODE_FLAGS = -m64 -gencode arch=compute_50,code=sm_50
CUDA_FLAGS = --ptxas-options=-v
CFLAGS = $(CUDA_FLAGS)

CXXFLAGS = -D_SSE2 $(INCLUDES)

LIBRARIES = -L$(OPENCV_PATH)/lib -L$(CUDA_PATH)/lib64

LDFLAGS = -lm -lpthread -lcudart -lopencv_calib3d -lopencv_contrib -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_highgui -lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_ml -lopencv_nonfree -lopencv_objdetect -lopencv_ocl -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_ts -lopencv_video -lopencv_videostab

target = canny

all: $(target)

canny: canny.o
	$(CXX) -o $@ $+ $(LIBRARIES) $(LDFLAGS)

.SUFFIXES: .o .cpp .c

.cpp.o:
	$(CXX) -c $(CXXFLAGS) $(INCLUDES) $<

.c.o:
	$(CC) -c $(CXXFLAGS) $(INCLUDES) $<

.PHONY: clean
clean:
	-rm -f *.o
	-rm -f canny
