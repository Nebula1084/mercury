dbg				:= 0
BIN           	:= $(GOPATH)/bin
CUDA_PATH 		:= /usr/local/cuda
CC				:= g++
NVCC			:= $(CUDA_PATH)/bin/nvcc -ccbin $(CC)
INCLUDES  		:= -I/usr/include -I.
LIBRARIES 		:=

# Debug build flags
ALL_CCFLAGS := -std=c++11
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

%.o:%.cu
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

%.o:%.cpp
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

pricer: Main.o asian/Asian.o american/American.o american/BinomialKernel.o \
	simulate/MonteCarloKernel.o simulate/MonteCarlo.o european/European.o \
	european/BasketEuropean.o european/GeometricEuropean.o european/Volatility.o \
	option/BlackScholes.o option/Norm.o option/Option.o european/ArithmeticEuropean.o \
	asian/GeometricAsian.o asian/ArithmeticAsian.o comm/Protocol.o comm/IPC.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
	cp pricer $(BIN)/pricer

cpu_test: CpuTest.o european/European.o option/Option.o option/Norm.o option/BlackScholes.o european/Volatility.o
	$(CC) -o $@ $+ $(LIBRARIES)

gpu_test: GpuTest.o asian/Asian.o american/American.o american/BinomialKernel.o \
	simulate/MonteCarloKernel.o simulate/MonteCarlo.o european/European.o \
	european/BasketEuropean.o european/GeometricEuropean.o european/Volatility.o \
	option/BlackScholes.o option/Norm.o option/Option.o european/ArithmeticEuropean.o \
	asian/GeometricAsian.o asian/ArithmeticAsian.o comm/Protocol.o comm/IPC.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

clean:
	@rm -rf pricer
	@rm -rf *.o
	@rm -rf *_test
	@rm -rf asian/*.o
	@rm -rf european/*.o
	@rm -rf simulate/*.o
	@rm -rf american/*.o
	@rm -rf option/*.o