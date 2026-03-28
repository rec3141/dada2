# Makefile for building standalone libdada2.so (Python-compatible)
# Usage:
#   make                          # CPU-only build
#   make CUDA_HOME=/path/to/cuda  # GPU-enabled build

CXX = g++
CXXFLAGS = -O3 -fPIC -std=c++11 -DNO_RCPP -DNDEBUG -march=native -fopenmp -Wno-format -flto

# Source files for standalone build (excludes Rmain.cpp, RcppExports.cpp,
# taxonomy.cpp, chimera.cpp, evaluate.cpp, filter.cpp)
CSRCS = src/derep.c

SRCS = src/dada2_capi.cpp \
       src/taxonomy_capi.cpp \
       src/cluster.cpp \
       src/containers.cpp \
       src/pval.cpp \
       src/error.cpp \
       src/kmers.cpp \
       src/misc.cpp \
       src/nwalign_endsfree.cpp \
       src/nwalign_vectorized.cpp

OBJS = $(SRCS:.cpp=.o)
LDFLAGS = -lm

# CUDA support (optional)
ifdef CUDA_HOME
  NVCC = $(CUDA_HOME)/bin/nvcc
  CUDA_INCL = $(shell test -d $(CUDA_HOME)/targets/x86_64-linux/include && echo "-I$(CUDA_HOME)/targets/x86_64-linux/include" || echo "-I$(CUDA_HOME)/include")
  CXXFLAGS += -DHAVE_CUDA $(CUDA_INCL)
  CUDA_OBJ = src/cuda_compare.o
  CUDA_LIB_DIR = $(shell test -d $(CUDA_HOME)/lib64 && echo $(CUDA_HOME)/lib64 || echo $(CUDA_HOME)/lib)
  LDFLAGS += -L$(CUDA_LIB_DIR) -Wl,-rpath,$(CUDA_LIB_DIR) -Wl,--no-as-needed -lcudart -Wl,--as-needed
  # Detect GPU arch
  CUDA_ARCH ?= -gencode arch=compute_89,code=sm_89
  NVCC_FLAGS = -Xcompiler -fPIC -O2 $(CUDA_ARCH) $(CUDA_INCL)
endif

.PHONY: all clean

all: libdada2.so

COBJS = $(CSRCS:.c=.o)

libdada2.so: $(OBJS) $(COBJS) $(CUDA_OBJ)
	$(CXX) -shared -o $@ $^ $(LDFLAGS) -fopenmp -lz

src/%.o: src/%.c
	gcc -O3 -fPIC -DNDEBUG -march=native -c $< -o $@

src/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

ifdef CUDA_HOME
src/cuda_compare.o: src/cuda_compare.cu src/cuda_compare.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@
endif

clean:
	rm -f src/*.o libdada2.so

# Print config
info:
	@echo "CXX: $(CXX)"
	@echo "CXXFLAGS: $(CXXFLAGS)"
	@echo "CUDA_HOME: $(CUDA_HOME)"
	@echo "LDFLAGS: $(LDFLAGS)"
