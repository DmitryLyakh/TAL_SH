NAME = talsh

#ADJUST THE FOLLOWING ACCORDINGLY:
#Cross-compiling wrappers: [WRAP|NOWRAP]:
export WRAP ?= NOWRAP
#Compiler: [GNU|PGI|INTEL|CRAY|IBM]:
export TOOLKIT ?= GNU
#Optimization: [DEV|OPT]:
export BUILD_TYPE ?= OPT
#MPI Library: [NONE|MPICH|OPENMPI]:
export MPILIB ?= NONE
#BLAS: [ATLAS|MKL|ACML|ESSL|NONE]:
export BLASLIB ?= ATLAS
#Nvidia GPU via CUDA: [CUDA|NOCUDA]:
export GPU_CUDA ?= NOCUDA
#Nvidia GPU architecture (two digits):
export GPU_SM_ARCH ?= 35
#Operating system: [LINUX|NO_LINUX]:
export EXA_OS ?= LINUX

#ADJUST EXTRAS (optional):
#Fast GPU tensor transpose (cuTT library): [YES|NO]:
export WITH_CUTT ?= NO

#GPU fine timing [YES|NO]:
export GPU_FINE_TIMING ?= NO

#WORKAROUNDS (ignore if you do not experience problems):
#Fool CUDA 7.0 with GCC > 4.9: [YES|NO]:
export FOOL_CUDA ?= NO

#SET YOUR LOCAL PATHS (for unwrapped builds):

#MPI library (whichever you have, set one):
# Set this if you have MPICH or its derivative:
export PATH_MPICH ?= /usr/local/mpi/mpich/3.2
#  Only reset these if MPI files are spread in the system directories:
 export PATH_MPICH_INC ?= $(PATH_MPICH)/include
 export PATH_MPICH_LIB ?= $(PATH_MPICH)/lib
 export PATH_MPICH_BIN ?= $(PATH_MPICH)/bin
# Set this if you have OPENMPI or its derivative:
export PATH_OPENMPI ?= /usr/local/mpi/openmpi/2.0.1
#  Only reset these if MPI files are spread in the system directories:
 export PATH_OPENMPI_INC ?= $(PATH_OPENMPI)/include
 export PATH_OPENMPI_LIB ?= $(PATH_OPENMPI)/lib
 export PATH_OPENMPI_BIN ?= $(PATH_OPENMPI)/bin

#BLAS library (whichever you have, set one):
# Set this if you do not have a vendor provided BLAS:
export PATH_BLAS_ATLAS ?= /usr/lib
# Set this if you have vendor provided BLAS (choose):
#  MKL BLAS:
export PATH_BLAS_MKL ?= /opt/intel/mkl/lib/intel64
export PATH_BLAS_MKL_DEP ?= /opt/intel/compilers_and_libraries/linux/lib/intel64_lin
#  ACML BLAS:
export PATH_BLAS_ACML ?= /opt/acml/5.3.1/gfortran64_fma4_mp/lib
#  ESSL BLAS:
export PATH_BLAS_ESSL ?= /sw/summitdev/essl/5.5.0/lib64
export PATH_BLAS_ESSL_DEP ?= /sw/summitdev/xl/20161123/xlf/15.1.5/lib

# CUDA (only if you build with CUDA):
export PATH_CUDA ?= /usr/local/cuda
#  Only reset these if CUDA files are spread in the system directories:
 export PATH_CUDA_INC ?= $(PATH_CUDA)/include
 export PATH_CUDA_LIB ?= $(PATH_CUDA)/lib64
 export PATH_CUDA_BIN ?= $(PATH_CUDA)/bin
# cuTT path (if you use cuTT library):
export PATH_CUTT ?= /home/div/src/cutt

#YOU ARE DONE!


#=================
#Fortran compiler:
FC_GNU = gfortran
FC_PGI = pgf90
FC_INTEL = ifort
FC_CRAY = ftn
FC_IBM = xlf2008_r
FC_MPICH = $(PATH_MPICH_BIN)/mpif90
FC_OPENMPI = $(PATH_OPENMPI_BIN)/mpifort
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
CC_MPICH = $(PATH_MPICH_BIN)/mpicc
CC_OPENMPI = $(PATH_OPENMPI_BIN)/mpicc
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
CPP_MPICH = $(PATH_MPICH_BIN)/mpic++
ifeq ($(EXA_OS),LINUX)
CPP_OPENMPI = $(PATH_OPENMPI_BIN)/mpic++
else
CPP_OPENMPI = $(PATH_OPENMPI_BIN)/mpicxx
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
 LIB = $(LIB_$(WRAP)) -lstdc++
else
 LIB = $(LIB_$(WRAP)) -lstdc++
endif

#MPI INCLUDES:
MPI_INC_MPICH = -I$(PATH_MPICH_INC)
MPI_INC_OPENMPI = -I$(PATH_OPENMPI_INC)
ifeq ($(MPILIB),NONE)
MPI_INC_NOWRAP = -I.
else
MPI_INC_NOWRAP = $(MPI_INC_$(MPILIB))
endif
MPI_INC_WRAP = -I.
MPI_INC = $(MPI_INC_$(WRAP))

#MPI LIBS:
MPI_LINK_MPICH = -L$(PATH_MPICH_LIB)
MPI_LINK_OPENMPI = -L$(PATH_OPENMPI_LIB)
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
LA_LINK_MKL = -L$(PATH_BLAS_MKL) -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lpthread -lm -ldl
else
LA_LINK_MKL = -L$(PATH_BLAS_MKL) -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lpthread -lm -ldl -L$(PATH_BLAS_MKL_DEP) -liomp5
endif
LA_LINK_ACML = -L$(PATH_BLAS_ACML) -lacml_mp
LA_LINK_ESSL = -L$(PATH_BLAS_ESSL) -lesslsmp -L$(PATH_BLAS_ESSL_DEP) -lxlf90_r -lxlfmath
ifeq ($(BLASLIB),NONE)
LA_LINK_NOWRAP = -L.
else
LA_LINK_NOWRAP = $(LA_LINK_$(BLASLIB))
endif
ifeq ($(BLASLIB),MKL)
LA_LINK_WRAP = $(LA_LINK_MKL)
else
LA_LINK_WRAP = -L.
endif
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

#Platform independence:
PIC_FLAG_GNU = -fPIC
PIC_FLAG_PGI = -fpic
PIC_FLAG_INTEL = -fpic
PIC_FLAG_IBM = -qpic=large
PIC_FLAG_CRAY = -hpic
PIC_FLAG = $(PIC_FLAG_$(TOOLKIT))

#CUDA FLAGS:
ifeq ($(GPU_CUDA),CUDA)
GPU_SM = sm_$(GPU_SM_ARCH)
GPU_ARCH = $(GPU_SM_ARCH)0
CUDA_HOST_NOWRAP = --compiler-bindir /usr/bin
CUDA_HOST_WRAP = -I.
CUDA_HOST = $(CUDA_HOST_$(WRAP))
CUDA_FLAGS_DEV = --compile -arch=$(GPU_SM) -g -G -lineinfo -DDEBUG_GPU -w
CUDA_FLAGS_OPT = --compile -arch=$(GPU_SM) -O3 -lineinfo -w
CUDA_FLAGS_CUDA = $(CUDA_HOST) $(CUDA_FLAGS_$(BUILD_TYPE)) -D_FORCE_INLINES -Xcompiler $(PIC_FLAG)
ifeq ($(FOOL_CUDA),NO)
CUDA_FLAGS_PRE1 = $(CUDA_FLAGS_CUDA) -D$(EXA_OS)
else
CUDA_FLAGS_PRE1 = $(CUDA_FLAGS_CUDA) -D$(EXA_OS) -D__GNUC__=4
endif
ifeq ($(WITH_CUTT),YES)
CUDA_FLAGS_PRE2 = $(CUDA_FLAGS_PRE1) -DUSE_CUTT
else
CUDA_FLAGS_PRE2 = $(CUDA_FLAGS_PRE1)
endif
ifeq ($(GPU_FINE_TIMING),YES)
CUDA_FLAGS = $(CUDA_FLAGS_PRE2) -DGPU_FINE_TIMING
else
CUDA_FLAGS = $(CUDA_FLAGS_PRE2)
endif
else
CUDA_FLAGS = -D_FORCE_INLINES
endif

#Accelerator support:
ifeq ($(TOOLKIT),IBM)
DF := -WF,
else
DF :=
endif
ifeq ($(BLASLIB),NONE)
NO_BLAS = -DNO_BLAS
else
NO_BLAS :=
endif
ifeq ($(GPU_CUDA),CUDA)
NO_GPU = -DCUDA_ARCH=$(GPU_ARCH)
else
NO_GPU = -DNO_GPU
endif
NO_AMD = -DNO_AMD
NO_PHI = -DNO_PHI

#C FLAGS:
ifeq ($(TOOLKIT),PGI)
CFLAGS_DEV = -c -g -O0 -D_DEBUG -silent -w
CFLAGS_OPT = -c -O3 -silent -w -Mnovect
else
CFLAGS_DEV = -c -g -O0 -D_DEBUG
CFLAGS_OPT = -c -O3
endif
CFLAGS = $(CFLAGS_$(BUILD_TYPE)) $(NO_GPU) $(NO_AMD) $(NO_PHI) $(NO_BLAS) -D$(EXA_OS) $(PIC_FLAG)

#FORTRAN FLAGS:
FFLAGS_INTEL_DEV = -c -g -O0 -fpp -vec-threshold4 -qopenmp -mkl=parallel
#FFLAGS_INTEL_DEV = -c -g -fpp -vec-threshold4 -openmp
FFLAGS_INTEL_OPT = -c -O3 -fpp -vec-threshold4 -qopenmp -mkl=parallel
#FFLAGS_INTEL_OPT = -c -O3 -fpp -vec-threshold4 -openmp
FFLAGS_CRAY_DEV = -c -g
FFLAGS_CRAY_OPT = -c -O3
FFLAGS_GNU_DEV = -c -fopenmp -fbacktrace -fcheck=bounds -fcheck=array-temps -fcheck=pointer -g -O0
FFLAGS_GNU_OPT = -c -fopenmp -O3
FFLAGS_PGI_DEV = -c -mp -Mcache_align -Mbounds -Mchkptr -Mstandard -Mallocatable=03 -g -O0
FFLAGS_PGI_OPT = -c -mp -Mcache_align -Mstandard -Mallocatable=03 -O3
FFLAGS_IBM_DEV = -c -qsmp=omp -g -O0 -qkeepparm -qcheck -qsigtrap
FFLAGS_IBM_OPT = -c -qsmp=omp -O3
FFLAGS = $(FFLAGS_$(TOOLKIT)_$(BUILD_TYPE)) $(DF)$(NO_GPU) $(DF)$(NO_AMD) $(DF)$(NO_PHI) $(DF)$(NO_BLAS) $(DF)-D$(EXA_OS) $(PIC_FLAG)

#THREADS:
LTHREAD_GNU   = -lgomp
LTHREAD_PGI   = -mp -lpthread
LTHREAD_INTEL = -liomp5
LTHREAD_CRAY  = -L.
LTHREAD_IBM   = -lxlsmp
LTHREAD = $(LTHREAD_$(TOOLKIT))

#LINKING:
LFLAGS = $(MPI_LINK) $(LA_LINK) $(LTHREAD) $(CUDA_LINK) $(LIB)

OBJS =  ./OBJ/dil_basic.o ./OBJ/stsubs.o ./OBJ/combinatoric.o ./OBJ/symm_index.o ./OBJ/timers.o \
	./OBJ/tensor_algebra.o ./OBJ/tensor_algebra_cpu.o ./OBJ/tensor_algebra_cpu_phi.o ./OBJ/tensor_dil_omp.o \
	./OBJ/mem_manager.o ./OBJ/tensor_algebra_gpu_nvidia.o ./OBJ/talshf.o ./OBJ/talshc.o

$(NAME): lib$(NAME).a ./OBJ/test.o ./OBJ/main.o
	$(FCOMP) ./OBJ/main.o ./OBJ/test.o lib$(NAME).a $(LFLAGS) -o test_$(NAME).x

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

./OBJ/tensor_algebra_gpu_nvidia.o: tensor_algebra_gpu_nvidia.cu talsh_complex.h tensor_algebra.h
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
