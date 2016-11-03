NAME = talsh

#ADJUST THE FOLLOWING ACCORDINGLY:
#Cross-compiling wrappers: [WRAP|NOWRAP]:
export WRAP ?= NOWRAP
#Compiler: [GNU|PGI|INTEL|CRAY|IBM]:
export TOOLKIT ?= GNU
#Optimization: [DEV|OPT]:
export BUILD_TYPE ?= OPT
#MPI Library: [MPICH|OPENMPI|NONE]:
export MPILIB ?= NONE
#BLAS: [ATLAS|MKL|ACML|ESSL|NONE]:
export BLASLIB ?= ATLAS
#Nvidia GPU via CUDA: [CUDA|NOCUDA]:
export GPU_CUDA ?= NOCUDA
#Nvidia GPU architecture (two digits):
export GPU_SM_ARCH ?= 35
#Operating system: [LINUX|NO_LINUX]:
export EXA_OS ?= LINUX

#EXTRAS:
#Fast GPU tensor transpose (cuTT library): [YES|NO]:
export WITH_CUTT ?= NO

#WORKAROUNDS (ignore if you do not experience problems):
#Fool CUDA 7.0 with GCC > 4.9: [YES|NO]:
export FOOL_CUDA ?= NO

#GPU FINE TIMING (for benchmarking only):
export GPU_FINE_TIMING ?= NO

#SET YOUR LOCAL PATHS (for unwrapped builds):
# MPI path (whichever you have chosen above):
export PATH_MPICH ?= /usr/local/mpich3.2
export PATH_OPENMPI ?= /usr/local/openmpi1.10.1
# BLAS lib path (whichever you have chosen above):
export PATH_BLAS_ATLAS ?= /usr/lib
export PATH_BLAS_MKL ?= /ccs/compilers/intel/rh6-x86_64/16.0.0/compilers_and_libraries/linux/mkl/lib
export PATH_BLAS_ACML ?= /opt/acml/5.3.1/gfortran64_fma4_mp/lib
export PATH_BLAS_ESSL ?= /sw/summitdev/essl/5.5.0/lib64
export PATH_BLAS_ESSL_DEP ?= /sw/summitdev/xl/161005/lib
# CUDA lib and include paths (if you build with CUDA):
export PATH_CUDA_LIB ?= /usr/lib/x86_64-linux-gnu
export PATH_CUDA_INC ?= /usr/include
# cuTT path (if you use cuTT library):
export PATH_CUTT ?= /home/dima/src/cutt

#YOU ARE DONE!


#=================
#Fortran compiler:
FC_GNU = gfortran
FC_PGI = pgf90
FC_INTEL = ifort
FC_CRAY = ftn
FC_IBM = xlf2008_r
FC_MPICH = $(PATH_MPICH)/bin/mpif90
FC_OPENMPI = $(PATH_OPENMPI)/bin/mpifort
ifeq ($(MPILIB),NONE)
FC_NOWRAP = $(FC_$(TOOLKIT))
else
FC_NOWRAP = $(FC_$(MPILIB))
endif
FC_WRAP = ftn
FCOMP = $(FC_$(WRAP))
#C compiler:
CC_GNU = gcc
CC_PGI = pgcc
CC_INTEL = icc
CC_CRAY = cc
CC_IBM = xlc_r
CC_MPICH = $(PATH_MPICH)/bin/mpicc
CC_OPENMPI = $(PATH_OPENMPI)/bin/mpicc
ifeq ($(MPILIB),NONE)
CC_NOWRAP = $(CC_$(TOOLKIT))
else
CC_NOWRAP = $(CC_$(MPILIB))
endif
CC_WRAP = cc
CCOMP = $(CC_$(WRAP))
#C++ compiler:
CPP_GNU = g++
CPP_PGI = pgc++
CPP_INTEL = icc
CPP_CRAY = CC
CPP_IBM = xlC_r
CPP_MPICH = $(PATH_MPICH)/bin/mpic++
ifeq ($(EXA_OS),LINUX)
CPP_OPENMPI = $(PATH_OPENMPI)/bin/mpic++
else
CPP_OPENMPI = $(PATH_OPENMPI)/bin/mpicxx
endif
ifeq ($(MPILIB),NONE)
CPP_NOWRAP = $(CPP_$(TOOLKIT))
else
CPP_NOWRAP = $(CPP_$(MPILIB))
endif
CPP_WRAP = CC
CPPCOMP = $(CPP_$(WRAP))
#CUDA compiler:
CUDA_COMP = nvcc

#COMPILER INCLUDES:
INC_GNU = -I.
INC_PGI = -I.
INC_INTEL = -I.
INC_CRAY = -I.
INC_IBM = -I.
INC_NOWRAP = $(INC_$(TOOLKIT))
INC_WRAP = -I.
INC = $(INC_$(WRAP))

#COMPILER LIBS:
LIB_GNU = -L.
LIB_PGI = -L.
LIB_INTEL = -L.
LIB_CRAY = -L.
LIB_IBM = -L.
LIB_NOWRAP = $(LIB_$(TOOLKIT))
LIB_WRAP = -L.
ifeq ($(TOOLKIT),PGI)
 LIB = $(LIB_$(WRAP))
else
 LIB = $(LIB_$(WRAP)) -lstdc++
endif

#MPI INCLUDES:
MPI_INC_MPICH = -I$(PATH_MPICH)/include
MPI_INC_OPENMPI = -I$(PATH_OPENMPI)/include
ifeq ($(MPILIB),NONE)
MPI_INC_NOWRAP = -I.
else
MPI_INC_NOWRAP = $(MPI_INC_$(MPILIB))
endif
MPI_INC_WRAP = -I.
MPI_INC = $(MPI_INC_$(WRAP))

#MPI LIBS:
MPI_LINK_MPICH = -L$(PATH_MPICH)/lib
MPI_LINK_OPENMPI = -L$(PATH_OPENMPI)/lib
ifeq ($(MPILIB),NONE)
MPI_LINK_NOWRAP = -L.
else
MPI_LINK_NOWRAP = $(MPI_LINK_$(MPILIB))
endif
MPI_LINK_WRAP = -L.
MPI_LINK = $(MPI_LINK_$(WRAP))

#LINEAR ALGEBRA FLAGS:
LA_LINK_ATLAS = -L$(PATH_BLAS_ATLAS) -lblas -llapack
ifeq ($(TOOLKIT),GNU)
LA_LINK_MKL = -L$(PATH_BLAS_MKL) -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -lpthread -lm -ldl
else
LA_LINK_MKL = -L$(PATH_BLAS_MKL) -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lpthread -lm -ldl
endif
LA_LINK_ACML = -L$(PATH_BLAS_ACML) -lacml_mp
LA_LINK_ESSL = -L$(PATH_BLAS_ESSL) -lessl -L$(PATH_BLAS_ESSL_DEP) -lxlf90_r -lxlfmath
ifeq ($(BLASLIB),NONE)
LA_LINK_NOWRAP = -L.
else
LA_LINK_NOWRAP = $(LA_LINK_$(BLASLIB))
endif
LA_LINK_WRAP = -L.
LA_LINK = $(LA_LINK_$(WRAP))

#CUDA INCLUDES:
ifeq ($(GPU_CUDA),CUDA)
CUDA_INC_NOWRAP = -I$(PATH_CUDA_INC)
CUDA_INC_WRAP = -I.
ifeq ($(WITH_CUTT),YES)
CUDA_INC = $(CUDA_INC_$(WRAP)) -I$(PATH_CUTT)/include
else
CUDA_INC = $(CUDA_INC_$(WRAP))
endif
else
CUDA_INC = -I.
endif

#CUDA LIBS:
CUDA_LINK_NOWRAP = -L$(PATH_CUDA_LIB) -lcudart -lcublas
CUDA_LINK_WRAP = -lcudart -lcublas
CUDA_LINK_CUDA = $(CUDA_LINK_$(WRAP))
CUDA_LINK_NOCUDA = -L.
CUDA_LINK = $(CUDA_LINK_$(GPU_CUDA))

#CUDA FLAGS:
ifeq ($(GPU_CUDA),CUDA)
GPU_SM = sm_$(GPU_SM_ARCH)
GPU_ARCH = $(GPU_SM_ARCH)0
CUDA_HOST_NOWRAP = --compiler-bindir /usr/bin
CUDA_HOST_WRAP = -I.
CUDA_HOST = $(CUDA_HOST_$(WRAP))
CUDA_FLAGS_DEV = --compile -arch=$(GPU_SM) -g -G -lineinfo -D DEBUG_GPU
CUDA_FLAGS_OPT = --compile -arch=$(GPU_SM) -O3 -lineinfo
CUDA_FLAGS_CUDA = $(CUDA_HOST) $(CUDA_FLAGS_$(BUILD_TYPE)) -D_FORCE_INLINES
ifeq ($(FOOL_CUDA),NO)
CUDA_FLAGS_PRE1 = $(CUDA_FLAGS_CUDA) -D$(EXA_OS)
else
CUDA_FLAGS_PRE1 = $(CUDA_FLAGS_CUDA) -D$(EXA_OS) -D__GNUC__=4
endif
ifeq ($(WITH_CUTT),YES)
CUDA_FLAGS_PRE2 = $(CUDA_FLAGS_PRE1) -D USE_CUTT
else
CUDA_FLAGS_PRE2 = $(CUDA_FLAGS_PRE1)
endif
ifeq ($(GPU_FINE_TIMING),YES)
CUDA_FLAGS = $(CUDA_FLAGS_PRE2) -D GPU_FINE_TIMING
else
CUDA_FLAGS = $(CUDA_FLAGS_PRE2)
endif
else
CUDA_FLAGS = -D_FORCE_INLINES
endif

#Accelerator support:
NO_ACCEL_CUDA = -D NO_AMD -D NO_PHI -D CUDA_ARCH=$(GPU_ARCH)
NO_ACCEL_NOCUDA = -D NO_AMD -D NO_PHI -D NO_GPU
NO_ACCEL = $(NO_ACCEL_$(GPU_CUDA))

#C FLAGS:
CFLAGS_DEV = -c -g $(NO_ACCEL) -D_DEBUG
CFLAGS_OPT = -c -O3 $(NO_ACCEL)
ifeq ($(BLASLIB),NONE)
CFLAGS = $(CFLAGS_$(BUILD_TYPE)) -D$(EXA_OS) -D NO_BLAS
else
CFLAGS = $(CFLAGS_$(BUILD_TYPE)) -D$(EXA_OS)
endif

#FORTRAN FLAGS:
FFLAGS_INTEL_DEV = -c -g -fpp -vec-threshold4 -qopenmp -mkl=parallel $(NO_ACCEL)
#FFLAGS_INTEL_DEV = -c -g -fpp -vec-threshold4 -openmp $(NO_ACCEL)
FFLAGS_INTEL_OPT = -c -O3 -fpp -vec-threshold4 -qopenmp -mkl=parallel $(NO_ACCEL)
#FFLAGS_INTEL_OPT = -c -O3 -fpp -vec-threshold4 -openmp $(NO_ACCEL)
FFLAGS_CRAY_DEV = -c -g $(NO_ACCEL)
FFLAGS_CRAY_OPT = -c -O3 $(NO_ACCEL)
FFLAGS_GNU_DEV = -c -fopenmp -fbacktrace -fcheck=bounds -fcheck=array-temps -fcheck=pointer -g $(NO_ACCEL)
FFLAGS_GNU_OPT = -c -fopenmp -O3 $(NO_ACCEL)
FFLAGS_PGI_DEV = -c -mp -Mcache_align -Mbounds -Mchkptr -Mstandard -g $(NO_ACCEL)
FFLAGS_PGI_OPT = -c -mp -Mcache_align -Mstandard -O3 $(NO_ACCEL)
FFLAGS_IBM_DEV = -c -qsmp=omp $(NO_ACCEL)
FFLAGS_IBM_OPT = -c -qsmp=omp -O3 $(NO_ACCEL)
ifeq ($(BLASLIB),NONE)
FFLAGS = $(FFLAGS_$(TOOLKIT)_$(BUILD_TYPE)) -D$(EXA_OS) -D NO_BLAS
else
FFLAGS = $(FFLAGS_$(TOOLKIT)_$(BUILD_TYPE)) -D$(EXA_OS)
endif

#THREADS:
LTHREAD_GNU   = -lgomp
LTHREAD_PGI   = -lpthread
LTHREAD_INTEL = -liomp5
LTHREAD_CRAY  = -L.
LTHREAD_IBM   = -L.
LTHREAD = $(LTHREAD_$(TOOLKIT))

#LINKING:
LFLAGS = $(LIB) $(LTHREAD) $(MPI_LINK) $(LA_LINK) $(CUDA_LINK)

OBJS =  ./OBJ/dil_basic.o ./OBJ/stsubs.o ./OBJ/combinatoric.o ./OBJ/symm_index.o ./OBJ/timers.o \
	./OBJ/tensor_algebra.o ./OBJ/tensor_algebra_cpu.o ./OBJ/tensor_algebra_cpu_phi.o ./OBJ/tensor_dil_omp.o \
	./OBJ/mem_manager.o ./OBJ/tensor_algebra_gpu_nvidia.o ./OBJ/talshf.o ./OBJ/talshc.o

$(NAME): lib$(NAME).a ./OBJ/test.o ./OBJ/main.o
	$(FCOMP) ./OBJ/main.o ./OBJ/test.o lib$(NAME).a $(LFLAGS) -o $(NAME).x

lib$(NAME).a: $(OBJS)
ifeq ($(WITH_CUTT),YES)
	mkdir -p tmp_obj__
	ar x $(PATH_CUTT)/lib/libcutt.a
	mv *.o ./tmp_obj__
	ar cr lib$(NAME).a $(OBJS) ./tmp_obj__/*.o
	rm -rf ./tmp_obj__
else
	ar cr lib$(NAME).a $(OBJS)
endif

./OBJ/dil_basic.o: dil_basic.F90
	mkdir -p ./OBJ
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
	$(CPPCOMP) $(INC) $(MPI_INC) $(CUDA_INC) $(CFLAGS) mem_manager.cpp -o ./OBJ/mem_manager.o

./OBJ/tensor_algebra_gpu_nvidia.o: tensor_algebra_gpu_nvidia.cu tensor_algebra.h
ifeq ($(GPU_CUDA),CUDA)
	$(CUDA_COMP) $(INC) $(MPI_INC) $(CUDA_INC) $(CUDA_FLAGS) --ptx --source-in-ptx tensor_algebra_gpu_nvidia.cu -o ./OBJ/tensor_algebra_gpu_nvidia.o
	$(CUDA_COMP) $(INC) $(MPI_INC) $(CUDA_INC) $(CUDA_FLAGS) tensor_algebra_gpu_nvidia.cu -o ./OBJ/tensor_algebra_gpu_nvidia.o
else
	cp tensor_algebra_gpu_nvidia.cu tensor_algebra_gpu_nvidia.cpp
	$(CPPCOMP) $(INC) $(MPI_INC) $(CUDA_INC) $(CFLAGS) tensor_algebra_gpu_nvidia.cpp -o ./OBJ/tensor_algebra_gpu_nvidia.o
	rm -f tensor_algebra_gpu_nvidia.cpp
endif

./OBJ/talshf.o: talshf.F90 ./OBJ/tensor_algebra_cpu_phi.o ./OBJ/tensor_algebra_gpu_nvidia.o ./OBJ/mem_manager.o
	$(FCOMP) $(INC) $(MPI_INC) $(CUDA_INC) $(FFLAGS) talshf.F90 -o ./OBJ/talshf.o

./OBJ/talshc.o: talshc.cpp talsh.h tensor_algebra.h ./OBJ/tensor_algebra_cpu_phi.o ./OBJ/tensor_algebra_gpu_nvidia.o ./OBJ/mem_manager.o
	$(CPPCOMP) $(INC) $(MPI_INC) $(CUDA_INC) $(CFLAGS) talshc.cpp -o ./OBJ/talshc.o

./OBJ/test.o: test.cpp talsh.h tensor_algebra.h lib$(NAME).a
	$(CPPCOMP) $(INC) $(MPI_INC) $(CUDA_INC) $(CFLAGS) test.cpp -o ./OBJ/test.o

./OBJ/main.o: main.F90 ./OBJ/test.o ./OBJ/talshf.o lib$(NAME).a
	$(FCOMP) $(INC) $(MPI_INC) $(CUDA_INC) $(FFLAGS) main.F90 -o ./OBJ/main.o


.PHONY: clean
clean:
	rm -f *.x *.a ./OBJ/* *.mod *.modmic *.ptx *.log
