PROG = hybrid
CUDA_HDRS = kernel.h crunch.h
CUDA_SRCS = kernel.cu crunch.cu
MPI_HDRS = 
MPI_SRCS = main.c

#######################
# Don't change these: #
#######################

# directory to store object files
CUDA_OBJDIR = ./object/cuda
CUDA_OBJS = $(patsubst %.cu, $(CUDA_OBJDIR)/%.o, $(CUDA_SRCS))
CUDA_CC = nvcc
CUDA_CFLAGS = -arch=sm_13 -Wno-deprecated-gpu-targets -L.
CUDA_LIBS = -lm

MPI_OBJDIR = ./object/mpi
MPI_OBJS = $(patsubst %.c, $(MPI_OBJDIR)/%.o, $(MPI_SRCS))
MPI_CC = mpic++
MPI_CFLAGS = -Wall -L/opt/cuda-6.5/lib64
MPI_LIBS = -lm -lcudart


####################
# Compiling rules: #
####################
# WARNING: *must* have a tab before each definition

# invoked when "make" is run
all : $(CUDA_OBJDIR) $(MPI_OBJDIR) $(PROG)

# links object files into executable
$(PROG) : $(CUDA_OBJS) $(MPI_OBJS)
	$(MPI_CC) $(MPI_CFLAGS) $^ -o $(PROG) $(MPI_LIBS)

# compiles source files into object files
$(CUDA_OBJDIR)/%.o : %.cu $(CUDA_HDRS)
	$(CUDA_CC) -c $(CUDA_CFLAGS) $< -o $@ $(CUDA_LIBS)

$(MPI_OBJDIR)/%.o : %.c $(MPI_HDRS)
	$(MPI_CC) -c $(MPI_CFLAGS) $< -o $@ $(MPI_LIBS)

# creates directory to store object files
$(CUDA_OBJDIR) :
	mkdir -p $@/

$(MPI_OBJDIR) :
	mkdir -p $@/

# cleans up object files and executable
# type "make clean" to use
# Note: you can add your own commands to remove other things (e.g. output files)
clean:
	rm -rf object/
	rm -f $(PROG)
