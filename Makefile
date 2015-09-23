NAME = nvtal_test.x
#Cray cross-compiling wrappers: [YES|NO]:
WRAP = NO
#Compiler: [GNU|PGI|INTEL|CRAY]:
TOOLKIT = GNU
#Optimization: [DEV|OPT]:
TYPE = OPT
#MPI [MPICH|OPENMPI|NONE]:
MPILIB = NONE
#DONE.
#-----------------------------------------------
FC_GNU = gfortran
FC_INTEL = ifort
FC_PGI = pgf90
FC_YES = ftn
FC_NO = $(FC_$(TOOLKIT))
FC = $(FC_$(WRAP))
CC_GNU = gcc
CC_INTEL = icc
CC_PGI = pgcc
CC_YES = cc
CC_NO = $(CC_$(TOOLKIT))
CC = $(CC_$(WRAP))
CPP_GNU = gcc
CPP_INTEL = icc
CPP_PGI = pgcc
CPP_YES = CC
CPP_NO = $(CPP_$(TOOLKIT))
CPP = $(CPP_$(WRAP))
CUDA_C = nvcc
MPI_INC_MPICH = -I/usr/lib/mpich/include
MPI_INC_OPENMPI = -I/usr/local/include
MPI_INC_NONE = -I.
MPI_INC_YES = -I.
MPI_INC_NO = $(MPI_INC_$(MPILIB))
MPI_INC = $(MPI_INC_$(WRAP))
MPI_LINK_MPICH = -L/usr/lib
MPI_LINK_OPENMPI = -L/usr/local/lib
MPI_LINK_NONE = -L.
MPI_LINK_YES = -L.
MPI_LINK_NO = $(MPI_LINK_$(MPILIB))
MPI_LINK = $(MPI_LINK_$(WRAP))
CUDA_INC_YES = -I.
CUDA_INC_NO = -I/usr/local/cuda/include
CUDA_INC = $(CUDA_INC_$(WRAP))
CUDA_LINK_YES = -L. -lcudart -lcublas
CUDA_LINK_NO = -L/usr/local/cuda/lib64 -lcudart -lcublas
CUDA_LINK = $(CUDA_LINK_$(WRAP))

CUDA_FLAGS_DEV = --compile -arch=sm_35 -g -G -D DEBUG_GPU
CUDA_FLAGS_OPT = --compile -arch=sm_35 -O3
CUDA_FLAGS = $(CUDA_FLAGS_$(TYPE)) -D CUDA_ARCH=350
LA_LINK_MKL = -lmkl_core -lmkl_intel_thread -lmkl_intel_lp64 -lmkl_blas95_lp64 -lmkl_lapack95_lp64 -lrt
LA_LINK_ACML = -lacml_mp -L/opt/acml/5.3.1/gfortran64_fma4_mp/lib
LA_LINK_DEFAULT_YES = -L.
LA_LINK_DEFAULT_NO = -L/usr/lib/atlas-base/atlas -lblas -llapack
LA_LINK_DEFAULT = $(LA_LINK_DEFAULT_$(WRAP))
LA_LINK_INTEL = $(LA_LINK_DEFAULT)
LA_LINK_CRAY = $(LA_LINK_DEFAULT)
LA_LINK_GNU = $(LA_LINK_DEFAULT)
LA_LINK_PGI = $(LA_LINK_DEFAULT)
LA_LINK = $(LA_LINK_$(TOOLKIT))
CFLAGS_DEV = -c -D CUDA_ARCH=350 -g
CFLAGS_OPT = -c -D CUDA_ARCH=350 -O3
CFLAGS = $(CFLAGS_$(TYPE))
FFLAGS_INTEL_DEV = -c -g -fpp -vec-threshold4 -openmp
FFLAGS_INTEL_OPT = -c -O3 -fpp -vec-threshold4 -openmp
FFLAGS_CRAY_DEV = -c -g
FFLAGS_CRAY_OPT = -c -O3
FFLAGS_GNU_DEV = -c -fopenmp -fbacktrace -fcheck=bounds -fcheck=array-temps -fcheck=pointer -pg
FFLAGS_GNU_OPT = -c -fopenmp -O3
FFLAGS_PGI_DEV = -c -mp -Mcache_align -Mbounds -Mchkptr -Mstandard -pg
FFLAGS_PGI_OPT = -c -mp -Mcache_align -Mstandard -O3
FFLAGS = $(FFLAGS_$(TOOLKIT)_$(TYPE)) -D CUDA_ARCH=350 -D NO_PHI -D NO_AMD
LTHREAD_INTEL = -liomp5
LTHREAD_CRAY  = -L.
LTHREAD_GNU   = -lgomp
LTHREAD_PGI   = -lpthread
LTHREAD = $(LTHREAD_$(TOOLKIT))
LFLAGS = $(LTHREAD) $(MPI_LINK) $(LA_LINK) $(CUDA_LINK) -o

OBJS =  special.o c_proc_bufs.o tensor_algebra_gpu_nvidia.o main.o

$(NAME): $(OBJS)
	$(FC) $(OBJS) $(LFLAGS) $(NAME)

special.o: special.F90
	$(FC) $(MPI_INC) $(CUDA_INC) $(FFLAGS) special.F90

c_proc_bufs.o: c_proc_bufs.cu tensor_algebra.h
	$(CUDA_C) $(MPI_INC) $(CUDA_INC) $(CUDA_FLAGS) c_proc_bufs.cu

tensor_algebra_gpu_nvidia.o: tensor_algebra_gpu_nvidia.cu tensor_algebra.h
	$(CUDA_C) $(MPI_INC) $(CUDA_INC) $(CUDA_FLAGS) tensor_algebra_gpu_nvidia.cu

main.o: main.cu
	$(CUDA_C) $(MPI_INC) $(CUDA_INC) $(CUDA_FLAGS) main.cu

clean:
	rm *.o *.mod *.modmic *.ptx *.x
