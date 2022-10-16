CUDA_ROOT_DIR = C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7
CUDA_LIB_DIR = $(CUDA_ROOT_DIR)/lib64
CUDA_INC_DIR = $(CUDA_ROOT_DIR)/include

SRC_DIR = src
OBJ_DIR = bin
INC_DIR = lib/UtilNPP

EXE = histEqualisation
OBJS = $(OBJ_DIR)/histEqualisation.o 

INCLUDES_PROJECT += -I/lib/
LINK_FLAGS = -L$(CUDA_LIB_DIR)
LINK_LIBS = -lcudart -lnppc -lnppidei -lnppist -lnppisu -lnppial -lnppitc -lnppicc -lculibos -lfreeimage

CC = g++
CC_FLAGS = -I$(INC_DIR) -I$(CUDA_INC_DIR)

NVCC = nvcc
NVCC_FLAGS = -I$(INC_DIR) -I$(CUDA_INC_DIR)

.PHONY : clean

all: build

build: histEqualisation

# Link c++ and CUDA compiled object files to target executable:
histEqualisation : $(OBJS)
	$(CC) -I/$(CUDA_INC_DIR) $(LINK_FLAGS) $(OBJS) -o $@ $(LINK_LIBS)

# Compile C++ source files to object files:
$(OBJS): $(SRC_DIR)/$(EXE).cpp 
	$(CC) -I/$(CUDA_INC_DIR) $(CC_FLAGS) -c $< -o $@

# Compile CUDA source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu
	$(NVCC) $(INCLUDES) $(NVCC_FLAGS) -c $< -o $@

# Clean objects in object directory.
clean:
	$(RM) bin/*.o $(EXE)