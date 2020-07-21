NAME = talsh

#ADJUST THE FOLLOWING ENVIRONMENT VARIABLES ACCORDINGLY (choices are given)
#until you see "YOU ARE DONE!". The comments will guide you through (read them).
#Alternatively you can export all relevant environment variables such that this
#Makefile will pick their values, so you will not need to update anything here.
#However you will still need to read the meaning of those variables below.

#Cray cross-compiling wrappers (only for Cray): [WRAP|NOWRAP]:
export WRAP ?= NOWRAP
#Compiler: [GNU|INTEL|CRAY|IBM|PGI]:
export TOOLKIT ?= GNU
#Optimization: [DEV|OPT|PRF]:
export BUILD_TYPE ?= OPT
#MPI library base: [NONE]:
export MPILIB ?= NONE
#BLAS: [ATLAS|MKL|OPENBLAS|ACML|LIBSCI|ESSL|NONE]:
export BLASLIB ?= OPENBLAS
#NVIDIA GPU via CUDA: [CUDA|NOCUDA]:
export GPU_CUDA ?= NOCUDA
#NVIDIA GPU architecture (two digits, >=35):
export GPU_SM_ARCH ?= 35
#Operating system: [LINUX|NO_LINUX]:
export EXA_OS ?= LINUX


#ADJUST EXTRAS (optional):

#LAPACK: [YES|NO]:
export WITH_LAPACK ?= YES
#Fast GPU tensor transpose (cuTT library): [YES|NO]:
export WITH_CUTT ?= NO
#In-place GPU tensor contraction (cuTensor library): [YES|NO]
export WITH_CUTENSOR ?= NO

#GPU fine timing [YES|NO]:
export GPU_FINE_TIMING ?= YES

#The build is part of the ExaTN library build [YES|NO]:
export EXATN_SERVICE ?= NO


#SET YOUR LOCAL PATHS (for direct builds without Cray compiler wrappers):

#MPI library base (whichever you have, set one):
# Set this if you use MPICH or its derivative (e.g. Cray-MPICH):
export PATH_MPICH ?= /usr/local/mpi/mpich/3.2.1
#  Only reset these if MPICH files are spread in system directories:
 export PATH_MPICH_INC ?= $(PATH_MPICH)/include
 export PATH_MPICH_LIB ?= $(PATH_MPICH)/lib
 export PATH_MPICH_BIN ?= $(PATH_MPICH)/bin
# Set this if you use OPENMPI or its derivative (e.g. IBM Spectrum MPI):
export PATH_OPENMPI ?= /usr/local/mpi/openmpi/3.1.0
#  Only reset these if OPENMPI files are spread in system directories:
 export PATH_OPENMPI_INC ?= $(PATH_OPENMPI)/include
 export PATH_OPENMPI_LIB ?= $(PATH_OPENMPI)/lib
 export PATH_OPENMPI_BIN ?= $(PATH_OPENMPI)/bin

#BLAS library (whichever you have chosen above):
# Set this path if you have chosen ATLAS (any default Linux BLAS):
export PATH_BLAS_ATLAS ?= /usr/lib/x86_64-linux-gnu
# Set this path to Intel root directory if you have chosen Intel MKL:
export PATH_INTEL ?= /opt/intel
#  Only reset these if Intel MKL libraries are spread in system directories:
export PATH_BLAS_MKL ?= $(PATH_INTEL)/mkl/lib/intel64
export PATH_BLAS_MKL_DEP ?= $(PATH_INTEL)/compilers_and_libraries/linux/lib/intel64_lin
export PATH_BLAS_MKL_INC ?= $(PATH_INTEL)/mkl/include/intel64/lp64
# Set this path if you have chosen OpenBLAS:
export PATH_BLAS_OPENBLAS ?= /usr/local/blas/openblas/lib
# Set this path if you have chosen ACML:
export PATH_BLAS_ACML ?= /opt/acml/5.3.1/gfortran64_fma4_mp/lib
# Set this path if you have chosen Cray LibSci:
export PATH_BLAS_LIBSCI ?= /opt/cray/pe/libsci/19.06.1/CRAY/8.5/x86_64/lib
# Set this path if you have chosen ESSL (also set PATH_IBM_XL_CPP, PATH_IBM_XL_FOR, PATH_IBM_XL_SMP below):
export PATH_BLAS_ESSL ?= /sw/summit/essl/6.1.0-2/essl/6.1/lib64

#IBM XL (only set these if you use IBM XL compiler and/or ESSL library):
export PATH_IBM_XL_CPP ?= /sw/summit/xl/16.1.1-3/xlC/16.1.1/lib
export PATH_IBM_XL_FOR ?= /sw/summit/xl/16.1.1-3/xlf/16.1.1/lib
export PATH_IBM_XL_SMP ?= /sw/summit/xl/16.1.1-3/xlsmp/5.1.1/lib

#LAPACK (only set these if you have chosen WITH_LAPACK=YES above):
export PATH_LAPACK_LIB ?= $(PATH_BLAS_OPENBLAS)
export LAPACK_LIBS ?= -L.

#CUDA (set these only if you build with CUDA):
export PATH_CUDA ?= /usr/local/cuda
# Only reset these if CUDA files are spread in system directories:
 export PATH_CUDA_INC ?= $(PATH_CUDA)/include
 export PATH_CUDA_LIB ?= $(PATH_CUDA)/lib64
 export PATH_CUDA_BIN ?= $(PATH_CUDA)/bin
# Reset your CUDA Host compiler if needed:
 export CUDA_HOST_COMPILER ?= /usr/bin/g++
# cuTT path (only if you use cuTT library):
export PATH_CUTT ?= /home/dima/src/cutt
# cuTensor path (only if you use cuTensor library):
export PATH_CUTENSOR ?= /home/dima/src/cutensor

#YOU ARE DONE! MAKE IT!


#=======================
ifeq ($(BUILD_TYPE),PRF)
 COMP_PREF = scorep --thread=omp --cuda
else
 COMP_PREF =
endif
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
ifeq ($(EXA_TALSH_ONLY),YES)
FCOMP = $(CMAKE_Fortran_COMPILER)
else
FCOMP = $(COMP_PREF) $(FC_$(WRAP))
endif
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
ifeq ($(EXA_TALSH_ONLY),YES)
CCOMP = $(CMAKE_C_COMPILER)
else
CCOMP = $(COMP_PREF) $(CC_$(WRAP))
endif
#C++ compiler:
CPP_GNU = g++
CPP_PGI = pgc++
CPP_INTEL = icc
CPP_CRAY = CC
CPP_IBM = xlC_r
CPP_MPICH = $(PATH_MPICH_BIN)/mpicxx
CPP_OPENMPI = $(PATH_OPENMPI_BIN)/mpicxx
ifeq ($(MPILIB),NONE)
CPP_NOWRAP = $(CPP_$(TOOLKIT))
else
CPP_NOWRAP = $(CPP_$(MPILIB))
endif
CPP_WRAP = CC
ifeq ($(EXA_TALSH_ONLY),YES)
CPPCOMP = $(CMAKE_CXX_COMPILER)
else
CPPCOMP = $(COMP_PREF) $(CPP_$(WRAP))
endif
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
ifeq ($(TOOLKIT),IBM)
 LIB = $(LIB_$(WRAP)) -L$(PATH_IBM_XL_CPP) -libmc++ -lstdc++
else
ifeq ($(EXA_OS),LINUX)
 LIB = $(LIB_$(WRAP)) -lstdc++
else
ifeq ($(TOOLKIT),INTEL)
 LIB = $(LIB_$(WRAP)) -lc++
else
 LIB = $(LIB_$(WRAP)) -lstdc++
endif
endif
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
LA_LINK_ATLAS = -L$(PATH_BLAS_ATLAS) -lblas
LA_LINK_OPENBLAS = -L$(PATH_BLAS_OPENBLAS) -lopenblas
LA_LINK_LIBSCI = -L$(PATH_BLAS_LIBSCI) -lsci_cray_mp
ifeq ($(TOOLKIT),GNU)
LA_LINK_MKL = -L$(PATH_BLAS_MKL) -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lpthread -lm -ldl
else
LA_LINK_MKL = -L$(PATH_BLAS_MKL) -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lpthread -lm -ldl -L$(PATH_BLAS_MKL_DEP) -liomp5
endif
LA_LINK_ACML = -L$(PATH_BLAS_ACML) -lacml_mp
LA_LINK_ESSL = -L$(PATH_BLAS_ESSL) -lessl -L$(PATH_IBM_XL_FOR) -lxlf90_r -lxlfmath
#LA_LINK_ESSL = -L$(PATH_BLAS_ESSL) -lesslsmp -L$(PATH_IBM_XL_SMP) -lxlsmp -L$(PATH_IBM_XL_FOR) -lxlf90_r -lxlfmath
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
ifeq ($(WITH_LAPACK),YES)
LA_LINK = $(LA_LINK_$(WRAP)) -L$(PATH_LAPACK_LIB) $(LAPACK_LIBS)
else
LA_LINK = $(LA_LINK_$(WRAP))
endif
ifeq ($(BLASLIB),MKL)
LA_INC = -DUSE_MKL -I$(PATH_BLAS_MKL_INC)
else
LA_INC = -I.
endif
ifeq ($(WITH_LAPACK),YES)
LA_INC += -DWITH_LAPACK
endif

#CUDA INCLUDES:
ifeq ($(GPU_CUDA),CUDA)
CUDA_INC_NOWRAP = -I$(PATH_CUDA_INC)
CUDA_INC_WRAP = -I.
ifeq ($(WITH_CUTT),YES)
CUDA_INC_PRE1 = $(CUDA_INC_$(WRAP)) -I$(PATH_CUTT)/include
else
CUDA_INC_PRE1 = $(CUDA_INC_$(WRAP))
endif
ifeq ($(WITH_CUTENSOR),YES)
CUDA_INC = $(CUDA_INC_PRE1) -I$(PATH_CUTENSOR)/include
else
CUDA_INC = $(CUDA_INC_PRE1)
endif
else
CUDA_INC = -I.
endif

#CUDA LIBS:
ifeq ($(WITH_CUTENSOR),YES)
CUDA_LINK_NOWRAP = -L$(PATH_CUDA_LIB) -L$(PATH_CUTENSOR)/lib -lcutensor -lcublas -lcudart
CUDA_LINK_WRAP = -L$(PATH_CUTENSOR)/lib -lcutensor -lcublas -lcudart
else
CUDA_LINK_NOWRAP = -L$(PATH_CUDA_LIB) -lcublas -lcudart
CUDA_LINK_WRAP = -lcublas -lcudart
endif
CUDA_LINK_CUDA = $(CUDA_LINK_$(WRAP))
CUDA_LINK_NOCUDA = -L.
CUDA_LINK = $(CUDA_LINK_$(GPU_CUDA))

#Platform independence:
PIC_FLAG_GNU = -fPIC
PIC_FLAG_PGI = -fpic
PIC_FLAG_INTEL = -fpic
PIC_FLAG_IBM = -qpic=large
PIC_FLAG_CRAY = -fpic
PIC_FLAG = $(PIC_FLAG_$(TOOLKIT))
PIC_FLAG_CUDA = $(PIC_FLAG_GNU)

#CUDA FLAGS:
ifeq ($(GPU_CUDA),CUDA)
GPU_SM = sm_$(GPU_SM_ARCH)
GPU_ARCH = $(GPU_SM_ARCH)0

CUDA_HOST_NOWRAP = -I.
CUDA_HOST_WRAP = -I.
CUDA_HOST = $(CUDA_HOST_$(WRAP))
CUDA_FLAGS_DEV = --compile -arch=$(GPU_SM) -std=c++11 -g -G -DDEBUG_GPU -w
CUDA_FLAGS_OPT = --compile -arch=$(GPU_SM) -std=c++11 -O3 -lineinfo -w
CUDA_FLAGS_PRF = --compile -arch=$(GPU_SM) -std=c++11 -g -G -O3 -w
CUDA_FLAGS_CUDA = $(CUDA_HOST) $(CUDA_FLAGS_$(BUILD_TYPE)) -D_FORCE_INLINES -D$(EXA_OS) -Xcompiler $(PIC_FLAG_CUDA)
ifeq ($(WITH_CUTT),YES)
CUDA_FLAGS_PRE1 = $(CUDA_FLAGS_CUDA) -DUSE_CUTT
else
CUDA_FLAGS_PRE1 = $(CUDA_FLAGS_CUDA)
endif
ifeq ($(WITH_CUTENSOR),YES)
CUDA_FLAGS_PRE2 = $(CUDA_FLAGS_PRE1) -DUSE_CUTENSOR
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
CFLAGS_INTEL_DEV = -c -g -O0 -qopenmp -D_DEBUG
CFLAGS_INTEL_OPT = -c -O3 -qopenmp
CFLAGS_INTEL_PRF = -c -g -O3 -qopenmp
CFLAGS_CRAY_DEV = -c -g -O0 -fopenmp -D_DEBUG
CFLAGS_CRAY_OPT = -c -O3 -fopenmp
CFLAGS_CRAY_PRF = -c -g -O3 -fopenmp
CFLAGS_GNU_DEV = -c -g -O0 -fopenmp -D_DEBUG
CFLAGS_GNU_OPT = -c -O3 -fopenmp
CFLAGS_GNU_PRF = -c -g -O3 -fopenmp
CFLAGS_PGI_DEV = -c -g -O0 -D_DEBUG -silent -w
CFLAGS_PGI_OPT = -c -O3 -silent -w -Mnovect
CFLAGS_PGI_PRF = -c -g -O3 -silent -w -Mnovect
CFLAGS_IBM_DEV = -c -g -O0 -qsmp=omp -D_DEBUG
CFLAGS_IBM_OPT = -c -O3 -qsmp=omp
CFLAGS_IBM_PRF = -c -g -O3 -qsmp=omp
ifeq ($(WITH_CUTT),YES)
ifeq ($(WITH_CUTENSOR),YES)
CFLAGS = $(CFLAGS_$(TOOLKIT)_$(BUILD_TYPE)) $(NO_GPU) $(NO_AMD) $(NO_PHI) $(NO_BLAS) -D$(EXA_OS) $(PIC_FLAG) -DUSE_CUTT -DUSE_CUTENSOR
else
CFLAGS = $(CFLAGS_$(TOOLKIT)_$(BUILD_TYPE)) $(NO_GPU) $(NO_AMD) $(NO_PHI) $(NO_BLAS) -D$(EXA_OS) $(PIC_FLAG) -DUSE_CUTT
endif
else
ifeq ($(WITH_CUTENSOR),YES)
CFLAGS = $(CFLAGS_$(TOOLKIT)_$(BUILD_TYPE)) $(NO_GPU) $(NO_AMD) $(NO_PHI) $(NO_BLAS) -D$(EXA_OS) $(PIC_FLAG) -DUSE_CUTENSOR
else
CFLAGS = $(CFLAGS_$(TOOLKIT)_$(BUILD_TYPE)) $(NO_GPU) $(NO_AMD) $(NO_PHI) $(NO_BLAS) -D$(EXA_OS) $(PIC_FLAG)
endif
endif
ifeq ($(EXATN_SERVICE),YES)
CFLAGS += -DEXATN_SERVICE
endif

#CPP FLAGS:
CPPFLAGS = $(CFLAGS) -std=c++11

#FORTRAN FLAGS:
FFLAGS_INTEL_DEV = -c -g -O0 -fpp -vec-threshold4 -traceback -qopenmp -mkl=parallel $(LA_INC)
FFLAGS_INTEL_OPT = -c -O3 -fpp -vec-threshold4 -traceback -qopenmp -mkl=parallel $(LA_INC)
FFLAGS_INTEL_PRF = -c -g -O3 -fpp -vec-threshold4 -traceback -qopenmp -mkl=parallel $(LA_INC)
FFLAGS_CRAY_DEV = -c -g -fopenmp -J OBJ $(LA_INC)
FFLAGS_CRAY_OPT = -c -O3 -fopenmp -J OBJ $(LA_INC)
FFLAGS_CRAY_PRF = -c -g -O3 -fopenmp -J OBJ $(LA_INC)
FFLAGS_GNU_DEV = -c -fopenmp -g -Og -fbacktrace -fcheck=bounds -fcheck=array-temps -fcheck=pointer -ffpe-trap=invalid,zero,overflow $(LA_INC)
FFLAGS_GNU_OPT = -c -fopenmp -O3 $(LA_INC)
FFLAGS_GNU_PRF = -c -fopenmp -g -O3 $(LA_INC)
FFLAGS_PGI_DEV = -c -mp -Mcache_align -Mbounds -Mchkptr -Mstandard -Mallocatable=03 -g -O0 $(LA_INC)
FFLAGS_PGI_OPT = -c -mp -Mcache_align -Mstandard -Mallocatable=03 -O3 $(LA_INC)
FFLAGS_PGI_PRF = -c -mp -Mcache_align -Mstandard -Mallocatable=03 -g -O3 $(LA_INC)
FFLAGS_IBM_DEV = -c -qsmp=noopt -g9 -O0 -qfullpath -qkeepparm -qcheck -qsigtrap -qstackprotect=all
FFLAGS_IBM_OPT = -c -qsmp=omp -O3
FFLAGS_IBM_PRF = -c -qsmp=omp -g -O3
FFLAGS = $(FFLAGS_$(TOOLKIT)_$(BUILD_TYPE)) $(DF)$(NO_GPU) $(DF)$(NO_AMD) $(DF)$(NO_PHI) $(DF)$(NO_BLAS) $(DF)-D$(EXA_OS) $(PIC_FLAG)

#THREADS:
ifeq ($(BUILD_TYPE),DEV)
LTHREAD_GNU   = -fopenmp
else
LTHREAD_GNU   = -fopenmp
endif
LTHREAD_PGI   = -mp -lpthread
LTHREAD_INTEL = -liomp5
LTHREAD_CRAY  = -fopenmp
LTHREAD_IBM   = -lxlsmp
LTHREAD = $(LTHREAD_$(TOOLKIT))

#LINKING:
LFLAGS = $(MPI_LINK) $(LA_LINK) $(LTHREAD) $(CUDA_LINK) $(LIB)

OBJS =  ./OBJ/dil_basic.o ./OBJ/stsubs.o ./OBJ/combinatoric.o ./OBJ/symm_index.o ./OBJ/timer.o ./OBJ/timers.o ./OBJ/nvtx_profile.o \
	./OBJ/byte_packet.o ./OBJ/tensor_algebra.o ./OBJ/tensor_algebra_cpu.o ./OBJ/tensor_algebra_cpu_phi.o \
	./OBJ/tensor_dil_omp.o ./OBJ/mem_manager.o ./OBJ/tensor_algebra_gpu.o ./OBJ/tensor_algebra_gpu_nvidia.o \
	./OBJ/talshf.o ./OBJ/talshc.o ./OBJ/talsh_task.o ./OBJ/talshxx.o

$(NAME): lib$(NAME).a ./OBJ/test.o ./OBJ/main.o
	$(FCOMP) ./OBJ/main.o ./OBJ/test.o lib$(NAME).a $(LFLAGS) -o test_$(NAME).x

lib$(NAME).a: $(OBJS)
ifeq ($(WITH_CUTT),YES)
	mkdir -p tmp_obj__
	ar x $(PATH_CUTT)/lib/libcutt.a
	mv *.o ./tmp_obj__
	ar cr lib$(NAME).a $(OBJS) ./tmp_obj__/*.o
ifeq ($(EXA_OS),LINUX)
ifeq ($(TOOLKIT),IBM)
	$(CPPCOMP) -qmkshrobj -o lib$(NAME).so $(OBJS) ./tmp_obj__/*.o
else
	$(CPPCOMP) -shared -o lib$(NAME).so $(OBJS) ./tmp_obj__/*.o
endif
endif
	rm -rf ./tmp_obj__
else
	ar cr lib$(NAME).a $(OBJS)
ifeq ($(EXA_OS),LINUX)
ifeq ($(TOOLKIT),IBM)
	$(CPPCOMP) -qmkshrobj -o lib$(NAME).so $(OBJS)
else
	$(CPPCOMP) -shared -o lib$(NAME).so $(OBJS)
endif
endif
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

./OBJ/timer.o: timer.cpp timer.h
	$(CPPCOMP) $(INC) $(MPI_INC) $(CUDA_INC) $(CPPFLAGS) timer.cpp -o ./OBJ/timer.o

./OBJ/timers.o: timers.F90 ./OBJ/timer.o
	$(FCOMP) $(INC) $(MPI_INC) $(CUDA_INC) $(FFLAGS) timers.F90 -o ./OBJ/timers.o

./OBJ/nvtx_profile.o: nvtx_profile.c nvtx_profile.h
	$(CPPCOMP) $(INC) $(MPI_INC) $(CUDA_INC) $(CPPFLAGS) nvtx_profile.c -o ./OBJ/nvtx_profile.o

./OBJ/byte_packet.o: byte_packet.cpp byte_packet.h
	$(CPPCOMP) $(INC) $(MPI_INC) $(CUDA_INC) $(CPPFLAGS) byte_packet.cpp -o ./OBJ/byte_packet.o

./OBJ/tensor_algebra.o: tensor_algebra.F90 ./OBJ/dil_basic.o
	$(FCOMP) $(INC) $(MPI_INC) $(CUDA_INC) $(FFLAGS) tensor_algebra.F90 -o ./OBJ/tensor_algebra.o

./OBJ/tensor_algebra_cpu.o: tensor_algebra_cpu.F90 ./OBJ/tensor_algebra.o ./OBJ/stsubs.o ./OBJ/combinatoric.o ./OBJ/symm_index.o ./OBJ/timers.o
	$(FCOMP) $(INC) $(MPI_INC) $(CUDA_INC) $(FFLAGS) tensor_algebra_cpu.F90 -o ./OBJ/tensor_algebra_cpu.o

./OBJ/tensor_algebra_cpu_phi.o: tensor_algebra_cpu_phi.F90 ./OBJ/tensor_algebra_cpu.o
	$(FCOMP) $(INC) $(MPI_INC) $(CUDA_INC) $(FFLAGS) tensor_algebra_cpu_phi.F90 -o ./OBJ/tensor_algebra_cpu_phi.o

./OBJ/tensor_dil_omp.o: tensor_dil_omp.F90 ./OBJ/timers.o
	$(FCOMP) $(INC) $(MPI_INC) $(CUDA_INC) $(FFLAGS) tensor_dil_omp.F90 -o ./OBJ/tensor_dil_omp.o

./OBJ/mem_manager.o: mem_manager.cpp mem_manager.h tensor_algebra.h device_algebra.h
	$(CPPCOMP) $(INC) $(MPI_INC) $(CUDA_INC) $(CPPFLAGS) mem_manager.cpp -o ./OBJ/mem_manager.o

./OBJ/tensor_algebra_gpu.o: tensor_algebra_gpu.cpp tensor_algebra.h mem_manager.h
	$(CPPCOMP) $(INC) $(MPI_INC) $(CUDA_INC) $(CPPFLAGS) tensor_algebra_gpu.cpp -o ./OBJ/tensor_algebra_gpu.o

./OBJ/tensor_algebra_gpu_nvidia.o: tensor_algebra_gpu_nvidia.cu talsh_complex.h tensor_algebra.h device_algebra.h
ifeq ($(GPU_CUDA),CUDA)
	$(CUDA_COMP) -ccbin $(CUDA_HOST_COMPILER) $(INC) $(MPI_INC) $(CUDA_INC) $(CUDA_FLAGS) --ptx --source-in-ptx tensor_algebra_gpu_nvidia.cu -o ./OBJ/tensor_algebra_gpu_nvidia.ptx
	$(CUDA_COMP) -ccbin $(CUDA_HOST_COMPILER) $(INC) $(MPI_INC) $(CUDA_INC) $(CUDA_FLAGS) tensor_algebra_gpu_nvidia.cu -o ./OBJ/tensor_algebra_gpu_nvidia.o
else
	cp tensor_algebra_gpu_nvidia.cu tensor_algebra_gpu_nvidia.cpp
	$(CPPCOMP) $(INC) $(MPI_INC) $(CUDA_INC) $(CPPFLAGS) tensor_algebra_gpu_nvidia.cpp -o ./OBJ/tensor_algebra_gpu_nvidia.o
	rm -f tensor_algebra_gpu_nvidia.cpp
endif

./OBJ/talshf.o: talshf.F90 ./OBJ/tensor_algebra_cpu_phi.o ./OBJ/tensor_algebra_gpu_nvidia.o ./OBJ/mem_manager.o
	$(FCOMP) $(INC) $(MPI_INC) $(CUDA_INC) $(FFLAGS) talshf.F90 -o ./OBJ/talshf.o

./OBJ/talshc.o: talshc.cpp talsh.h talsh_complex.h tensor_algebra.h device_algebra.h ./OBJ/tensor_algebra_cpu_phi.o ./OBJ/tensor_algebra_gpu_nvidia.o ./OBJ/mem_manager.o
	$(CPPCOMP) $(INC) $(MPI_INC) $(CUDA_INC) $(CPPFLAGS) talshc.cpp -o ./OBJ/talshc.o

./OBJ/talsh_task.o: talsh_task.cpp talsh.h ./OBJ/talshc.o
	$(CPPCOMP) $(INC) $(MPI_INC) $(CUDA_INC) $(CPPFLAGS) talsh_task.cpp -o ./OBJ/talsh_task.o

./OBJ/talshxx.o: talshxx.cpp talshxx.hpp ./OBJ/talshc.o
	$(CPPCOMP) $(INC) $(MPI_INC) $(CUDA_INC) $(CPPFLAGS) talshxx.cpp -o ./OBJ/talshxx.o

./OBJ/test.o: test.cpp talshxx.hpp talsh_task.hpp talsh.h tensor_algebra.h device_algebra.h lib$(NAME).a
	$(CPPCOMP) $(INC) $(MPI_INC) $(CUDA_INC) $(CPPFLAGS) test.cpp -o ./OBJ/test.o

./OBJ/main.o: main.F90 ./OBJ/test.o ./OBJ/talshf.o lib$(NAME).a
	$(FCOMP) $(INC) $(MPI_INC) $(CUDA_INC) $(FFLAGS) main.F90 -o ./OBJ/main.o


.PHONY: clean
clean:
	rm -f *.x *.a *.so ./OBJ/* *.mod *.modmic *.ptx *.log
