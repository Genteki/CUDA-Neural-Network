OBJDIR=obj
INCDIR=inc
DEPDIR=dep
CUDASAMPLES=/usr/local/cuda/samples/Common
CUDA_INC=/usr/local/cuda/include
# Compilers
CC=mpic++
CUD=nvcc

TARGET := main

CPPSRCS := $(wildcard *.cpp)
CPPOBJS := $(CPPSRCS:%.cpp=$(OBJDIR)/%.o)

CUDSRCS := $(wildcard *.cu)
CUDOBJS := $(CUDSRCS:%.cu=$(OBJDIR)/%.o)

UTLSRCS := $(wildcard utils/*.cpp) 
UTLOBJS := $(UTLSRCS:utils/%.cpp=$(OBJDIR)/%.o)

TSTSRCS=test/test.cu
TSTOBJS=test.o

OBJS := $(CPPOBJS) $(CUDOBJS) $(UTLOBJS)

DEPFILES := $(OBJS:$(OBJDIR)/%.o=$(DEPDIR)/%.d)

# Flags
CFLAGS=-O3 -std=c++17 -DARMA_DONT_USE_WRAPPER -DARMA_USE_LAPACK
CUDFLAGS=-c -O3 -arch=compute_80 -code=sm_80 -Xcompiler -Wall,-Winline,-Wextra,-Wno-strict-aliasing
INCFLAGS=-I$(CUDASAMPLES) -I$(INCDIR) -I$(CUDA_INC)
LDFLAGS=-lblas -llapack -larmadillo -lcublas -lcudart
DEPFLAGS=-MT $@ -MMD -MF $(addprefix $(DEPDIR)/, $(notdir $*)).d

CC_CMD=$(CC) $(CFLAGS) $(INCFLAGS)
CU_CMD=$(CUD) $(CUDFLAGS) $(INCFLAGS)

# --fmad=false

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $@ $(LDFLAGS) 

$(CPPOBJS): $(OBJDIR)/%.o: %.cpp $(DEPDIR)/%.d
	@mkdir -p $(OBJDIR)
	@mkdir -p $(DEPDIR)
	$(CC_CMD) -c $< -o $@ $(DEPFLAGS)

$(UTLOBJS): $(OBJDIR)/%.o: utils/%.cpp $(DEPDIR)/%.d 
	@mkdir -p $(OBJDIR)
	@mkdir -p $(DEPDIR)
	$(CC_CMD) -c $< -o $@ $(DEPFLAGS)

$(CUDOBJS): $(OBJDIR)/%.o: %.cu $(DEPDIR)/%.d
	@mkdir -p $(OBJDIR)
	@mkdir -p $(DEPDIR)
	$(CU_CMD) -c $< -o $@ $(DEPFLAGS)

$(DEPFILES):
include $(wildcard $(DEPFILES))

# additional variable for the test binary
TSTBIN := testbin

# rule to build the test object file
$(OBJDIR)/$(TSTOBJS): $(TSTSRCS)
	@mkdir -p $(OBJDIR)
	@mkdir -p $(DEPDIR)
	$(CU_CMD) -c $< -o $@ $(DEPFLAGS)

# rule to build the test binary
$(TSTBIN): $(OBJDIR)/$(TSTOBJS) $(OBJDIR)/gpu_func.o
	$(CC) $^ -o $@ $(LDFLAGS)

# rule to run the test
test: $(TSTBIN)
	./$(TSTBIN)

clean:
	rm -rf $(OBJDIR)/*.o $(DEPDIR)/*.d main $(TSTBIN)

clear:
	rm -rf fp-* 
