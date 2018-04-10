dbg				:= 0
BIN           	:= $(GOPATH)/bin
CUDA_PATH 		:= /usr/local/cuda
CC				:= g++
NVCC			:= $(CUDA_PATH)/bin/nvcc -ccbin $(CC)
INCLUDES  		:= -I/usr/include -I.
LIBRARIES 		:=

# Debug build flags
ALL_CCFLAGS :=
ifeq ($(dbg),1)
      ALL_CCFLAGS += -g -G
endif

# Gencode arguments
SMS ?= 30 35 37 50 52 60

ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif

%.o:%.cpp
	$(CC) $(INCLUDES) $(ALL_CCFLAGS) -o $@ -c $<

%.o:%.cu
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(BIN)/pricer: Main.o IPC.o option/European.o option/Volatility.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

cpu_test: CpuTest.o option/European.o option/Option.o option/Norm.o
	$(CC) -o $@ $+ $(LIBRARIES)

gpu_test: GpuTest.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

clean:
	@rm -rf *.o
	@rm -rf *_test