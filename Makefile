NAME = talsh

#Cross-compiling wrappers: [WRAP|NOWRAP]:
WRAP ?= NOWRAP
#Compiler: [GNU|PGI|INTEL|CRAY]:
TOOLKIT ?= GNU
#Optimization: [DEV|OPT]:
BUILD_TYPE ?= DEV
#MPI Library: [MPICH|OPENMPI]:
MPILIB ?= OPENMPI
#BLAS: [ATLAS|MKL|ACML]:
BLASLIB ?= ATLAS
#Nvidia GPU via CUDA: [CUDA|NOCUDA]:
GPU_CUDA ?= CUDA

#Local paths (for unwrapped compilation):
# MPI:
PATH_MPICH ?= /usr/local/mpich3.2
PATH_OPENMPI ?= /usr/local/openmpi1.10.1
# BLAS LIB:
PATH_BLAS_ATLAS ?= /usr/lib
PATH_BLAS_INTEL ?= /usr/lib
PATH_BLAS_ACML ?= /usr/lib
PATH_BLAS = $(PATH_BLAS_$(BLASLIB))
# CUDA:
PATH_CUDA ?= /usr/local/cuda
#DONE.

#=================
#Fortran compiler:
FC_GNU = gfortran
FC_PGI = pgf90
FC_INTEL = ifort
FC_CRAY = ftn
FC_MPICH = $(PATH_MPICH)/bin/mpif90
FC_OPENMPI = $(PATH_OPENMPI)/bin/mpifort
FC_NOWRAP = $(FC_$(MPILIB))
FC_WRAP = ftn
FCOMP = $(FC_$(WRAP))
#C compiler:
CC_GNU = gcc
CC_PGI = pgcc
CC_INTEL = icc
CC_CRAY = cc
CC_MPICH = $(PATH_MPICH)/bin/mpicc
CC_OPENMPI = $(PATH_OPENMPI)/bin/mpicc
CC_NOWRAP = $(CC_$(MPILIB))
CC_WRAP = cc
CCOMP = $(CC_$(WRAP))
#C++ compiler:
CPP_GNU = g++
CPP_PGI = pgc++
CPP_INTEL = icc
CPP_CRAY = CC
CPP_MPICH = $(PATH_MPICH)/bin/mpic++
CPP_OPENMPI = $(PATH_OPENMPI)/bin/mpic++
CPP_NOWRAP = $(CPP_$(MPILIB))
CPP_WRAP = CC
CPPCOMP = $(CPP_$(WRAP))
#CUDA compiler:
CUDA_COMP = nvcc

#COMPILER INCLUDES:
INC_GNU = -I.
INC_PGI = -I.
INC_INTEL = -I.
INC_CRAY = -I.
INC_NOWRAP = $(INC_$(TOOLKIT))
INC_WRAP = -I.
INC = $(INC_$(WRAP))

#COMPILER LIBS:
LIB_GNU = -L.
LIB_PGI = -L.
LIB_INTEL = -L.
LIB_CRAY = -L.
LIB_NOWRAP = $(LIB_$(TOOLKIT))
LIB_WRAP = -L.
LIB = $(LIB_$(WRAP)) -lstdc++

#MPI INCLUDES:
MPI_INC_MPICH = -I$(PATH_MPICH)/include
MPI_INC_OPENMPI = -I$(PATH_OPENMPI)/include
MPI_INC_NOWRAP = $(MPI_INC_$(MPILIB))
MPI_INC_WRAP = -I.
MPI_INC = $(MPI_INC_$(WRAP))

#MPI LIBS:
MPI_LINK_MPICH = -L$(PATH_MPICH)/lib
MPI_LINK_OPENMPI = -L$(PATH_OPENMPI)/lib
MPI_LINK_NOWRAP = $(MPI_LINK_$(MPILIB))
MPI_LINK_WRAP = -L.
MPI_LINK = $(MPI_LINK_$(WRAP))

#LINEAR ALGEBRA FLAGS:
LA_LINK_ATLAS = -L$(PATH_BLAS) -lblas -llapack
LA_LINK_MKL = -L$(PATH_BLAS) -lmkl_core -lmkl_intel_thread -lmkl_intel_lp64 -lmkl_blas95_lp64 -lmkl_lapack95_lp64 -lrt
LA_LINK_ACML = -L$(PATH_BLAS) -lacml_mp
LA_LINK_WRAP = -L.
LA_LINK_NOWRAP = $(LA_LINK_$(BLASLIB))
LA_LINK = $(LA_LINK_$(WRAP))

#CUDA INCLUDES:
CUDA_INC_CUDA = -I$(PATH_CUDA)/include
CUDA_INC_NOCUDA = -I.
CUDA_INC_NOWRAP = $(CUDA_INC_$(GPU_CUDA))
CUDA_INC_WRAP = -I.
CUDA_INC = $(CUDA_INC_$(WRAP))

#CUDA LIBS:
CUDA_LINK_NOWRAP = -L$(PATH_CUDA)/lib64 -lcudart -lcublas
CUDA_LINK_WRAP = -lcudart -lcublas
CUDA_LINK_CUDA = $(CUDA_LINK_$(WRAP))
CUDA_LINK_NOCUDA = -L.
CUDA_LINK = $(CUDA_LINK_$(GPU_CUDA))

#CUDA FLAGS:
CUDA_HOST_NOWRAP = --compiler-bindir /usr/bin
CUDA_HOST_WRAP = -I.
CUDA_HOST = $(CUDA_HOST_$(WRAP))
CUDA_FLAGS_DEV = --compile -arch=sm_35 -g -G -D CUDA_ARCH=350 -D DEBUG_GPU
CUDA_FLAGS_OPT = --compile -arch=sm_35 -O3 -D CUDA_ARCH=350
CUDA_FLAGS_CUDA = $(CUDA_HOST) $(CUDA_FLAGS_$(BUILD_TYPE))
CUDA_FLAGS_NOCUDA = -I.
CUDA_FLAGS = $(CUDA_FLAGS_$(GPU_CUDA))

#Accelerator support:
NO_ACCEL_CUDA = -D NO_AMD -D NO_PHI -D CUDA_ARCH=350
NO_ACCEL_NOCUDA = -D NO_AMD -D NO_PHI -D NO_GPU
NO_ACCEL = $(NO_ACCEL_$(GPU_CUDA))

#C FLAGS:
CFLAGS_DEV = -c -g $(NO_ACCEL)
CFLAGS_OPT = -c -O3 $(NO_ACCEL)
CFLAGS = $(CFLAGS_$(BUILD_TYPE))

#FORTRAN FLAGS:
FFLAGS_INTEL_DEV = -c -g -fpp -vec-threshold4 -openmp $(NO_ACCEL)
FFLAGS_INTEL_OPT = -c -O3 -fpp -vec-threshold4 -openmp $(NO_ACCEL)
FFLAGS_CRAY_DEV = -c -g $(NO_ACCEL)
FFLAGS_CRAY_OPT = -c -O3 $(NO_ACCEL)
FFLAGS_GNU_DEV = -c -fopenmp -fbacktrace -fcheck=bounds -fcheck=array-temps -fcheck=pointer -pg $(NO_ACCEL)
FFLAGS_GNU_OPT = -c -fopenmp -O3 $(NO_ACCEL)
FFLAGS_PGI_DEV = -c -mp -Mcache_align -Mbounds -Mchkptr -Mstandard -pg $(NO_ACCEL)
FFLAGS_PGI_OPT = -c -mp -Mcache_align -Mstandard -O3 $(NO_ACCEL)
FFLAGS = $(FFLAGS_$(TOOLKIT)_$(BUILD_TYPE))

#THREADS:
LTHREAD_GNU   = -lgomp
LTHREAD_PGI   = -lpthread
LTHREAD_INTEL = -liomp5
LTHREAD_CRAY  = -L.
LTHREAD = $(LTHREAD_$(TOOLKIT))

#LINKING:
LFLAGS = $(LIB) $(LTHREAD) $(MPI_LINK) $(LA_LINK) $(CUDA_LINK)

OBJS =  ./OBJ/dil_basic.o ./OBJ/stsubs.o ./OBJ/combinatoric.o ./OBJ/symm_index.o ./OBJ/timers.o \
	./OBJ/tensor_algebra.o ./OBJ/tensor_algebra_cpu.o ./OBJ/tensor_algebra_cpu_phi.o ./OBJ/tensor_dil_omp.o \
	./OBJ/mem_manager.o ./OBJ/tensor_algebra_gpu_nvidia.o ./OBJ/talshf.o ./OBJ/talshc.o

$(NAME): lib$(NAME).a ./OBJ/test.o ./OBJ/main.o
	$(FCOMP) ./OBJ/main.o ./OBJ/test.o lib$(NAME).a $(LFLAGS) -o $(NAME).x

lib$(NAME).a: $(OBJS)
	ar cr lib$(NAME).a $(OBJS)

./OBJ/dil_basic.o: dil_basic.F90
	$(FCOMP) $(INC) $(MPI_INC) $(CUDA_INC) $(FFLAGS) dil_basic.F90 -o ./OBJ/dil_basic.o

./OBJ/stsubs.o: stsubs.F90
	$(FCOMP) $(INC) $(MPI_INC) $(CUDA_INC) $(FFLAGS) stsubs.F90 -o ./OBJ/stsubs.o

./OBJ/combinatoric.o: combinatoric.F90
	$(FCOMP) $(INC) $(MPI_INC) $(CUDA_INC) $(FFLAGS) combinatoric.F90 -o ./OBJ/combinatoric.o

./OBJ/symm_index.o: symm_index.F90
	$(FCOMP) $(INC) $(MPI_INC) $(CUDA_INC) $(FFLAGS) symm_index.F90 -o ./OBJ/symm_index.o

./OBJ/timers.o: timers.F90
	$(FCOMP) $(INC) $(MPI_INC) $(CUDA_INC) $(FFLAGS) timers.F90 -o ./OBJ/timers.o

./OBJ/tensor_algebra.o: tensor_algebra.F90 ./OBJ/dil_basic.o
	$(FCOMP) $(INC) $(MPI_INC) $(CUDA_INC) $(FFLAGS) tensor_algebra.F90 -o ./OBJ/tensor_algebra.o

./OBJ/tensor_algebra_cpu.o: tensor_algebra_cpu.F90 ./OBJ/tensor_algebra.o ./OBJ/stsubs.o ./OBJ/combinatoric.o ./OBJ/symm_index.o ./OBJ/timers.o
	$(FCOMP) $(INC) $(MPI_INC) $(CUDA_INC) $(FFLAGS) tensor_algebra_cpu.F90 -o ./OBJ/tensor_algebra_cpu.o

./OBJ/tensor_algebra_cpu_phi.o: tensor_algebra_cpu_phi.F90 ./OBJ/tensor_algebra_cpu.o
	$(FCOMP) $(INC) $(MPI_INC) $(CUDA_INC) $(FFLAGS) tensor_algebra_cpu_phi.F90 -o ./OBJ/tensor_algebra_cpu_phi.o

./OBJ/tensor_dil_omp.o: tensor_dil_omp.F90 ./OBJ/timers.o
	$(FCOMP) $(INC) $(MPI_INC) $(CUDA_INC) $(FFLAGS) tensor_dil_omp.F90 -o ./OBJ/tensor_dil_omp.o

./OBJ/mem_manager.o: mem_manager.cpp mem_manager.h tensor_algebra.h
	$(CCOMP) $(INC) $(MPI_INC) $(CUDA_INC) $(CFLAGS) mem_manager.cpp -o ./OBJ/mem_manager.o

./OBJ/tensor_algebra_gpu_nvidia.o: tensor_algebra_gpu_nvidia.cu tensor_algebra.h
ifeq ($(GPU_CUDA),CUDA)
	$(CUDA_COMP) $(INC) $(MPI_INC) $(CUDA_INC) $(CUDA_FLAGS) -ptx tensor_algebra_gpu_nvidia.cu -o ./OBJ/tensor_algebra_gpu_nvidia.o
	$(CUDA_COMP) $(INC) $(MPI_INC) $(CUDA_INC) $(CUDA_FLAGS) tensor_algebra_gpu_nvidia.cu -o ./OBJ/tensor_algebra_gpu_nvidia.o
else
	cp tensor_algebra_gpu_nvidia.cu tensor_algebra_gpu_nvidia.cpp
	$(CCOMP) $(INC) $(MPI_INC) $(CUDA_INC) $(CFLAGS) tensor_algebra_gpu_nvidia.cpp -o ./OBJ/tensor_algebra_gpu_nvidia.o
	rm -f tensor_algebra_gpu_nvidia.cpp
endif

./OBJ/talshf.o: talshf.F90 ./OBJ/tensor_algebra_cpu_phi.o ./OBJ/tensor_algebra_gpu_nvidia.o ./OBJ/mem_manager.o
	$(FCOMP) $(INC) $(MPI_INC) $(CUDA_INC) $(FFLAGS) talshf.F90 -o ./OBJ/talshf.o

./OBJ/talshc.o: talshc.cpp talsh.h tensor_algebra.h ./OBJ/tensor_algebra_cpu_phi.o ./OBJ/tensor_algebra_gpu_nvidia.o ./OBJ/mem_manager.o
	$(CCOMP) $(INC) $(MPI_INC) $(CUDA_INC) $(CFLAGS) talshc.cpp -o ./OBJ/talshc.o

./OBJ/test.o: test.cpp talsh.h tensor_algebra.h lib$(NAME).a
	$(CCOMP) $(INC) $(MPI_INC) $(CUDA_INC) $(CFLAGS) test.cpp -o ./OBJ/test.o

./OBJ/main.o: main.F90 ./OBJ/test.o ./OBJ/talshf.o lib$(NAME).a
	$(FCOMP) $(INC) $(MPI_INC) $(CUDA_INC) $(FFLAGS) main.F90 -o ./OBJ/main.o


.PHONY: clean
clean:
	rm *.x *.a ./OBJ/* *.mod *.modmic *.ptx
