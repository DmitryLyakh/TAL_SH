NAME = tal_nv_test.x
TOOLKIT = CRAY
TYPE = OPT

FC  = ftn
CC  = cc
CPP = CC
CUDA_C = nvcc
MPI_INC = -I.
CUDA_INC = -I.
CUDA_LINK = -lcudart -lcublas -L.

CUDA_FLAGS_DEV = --compile -arch=sm_35 -D CUDA_ARCH=350 -g -G -D DEBUG_GPU
CUDA_FLAGS_OPT = --compile -arch=sm_35 -D CUDA_ARCH=350 -O3
CUDA_FLAGS = $(CUDA_FLAGS_$(TYPE))
LA_LINK_MKL = -lmkl_core -lmkl_intel_thread -lmkl_intel_lp64 -lmkl_blas95_lp64 -lmkl_lapack95_lp64 -lrt
LA_LINK_ACML = -lacml_mp -L/opt/acml/5.3.1/gfortran64_fma4_mp/lib
LA_LINK_DEFAULT = -L.
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
FFLAGS_CRAY_DEV = -c -D CUDA_ARCH=350 -g
FFLAGS_CRAY_OPT = -c -D CUDA_ARCH=350 -O3
FFLAGS_GNU_DEV = -c -fopenmp -fbacktrace -fcheck=bounds -fcheck=array-temps -fcheck=pointer -pg
FFLAGS_GNU_OPT = -c -fopenmp -O3
FFLAGS_PGI_DEV = -c -mp -Mcache_align -Mbounds -Mchkptr -Mstandard -pg
FFLAGS_PGI_OPT = -c -mp -Mcache_align -Mstandard -O3
FFLAGS = $(FFLAGS_$(TOOLKIT)_$(TYPE)) -D NO_PHI -D NO_AMD
LTHREAD_INTEL = -liomp5
LTHREAD_CRAY  = -L.
LTHREAD_GNU   = -lgomp
LTHREAD_PGI   = -lpthread
LTHREAD = $(LTHREAD_$(TOOLKIT))
LFLAGS = $(LTHREAD) $(LA_LINK) $(CUDA_LINK) -o

OBJS = special.o c_proc_bufs.o dil_tal_nv.o main.o

$(NAME): $(OBJS)
	$(FC) $(OBJS) $(LFLAGS) $(NAME)

special.o: special.F90
	$(FC) $(MPI_INC) $(CUDA_INC) $(FFLAGS) special.F90

c_proc_bufs.o: c_proc_bufs.cu tensor_algebra.h
	$(CUDA_C) $(MPI_INC) $(CUDA_INC) $(CUDA_FLAGS) c_proc_bufs.cu

dil_tal_nv.o: dil_tal_nv.cu tensor_algebra.h
	$(CUDA_C) $(MPI_INC) $(CUDA_INC) $(CUDA_FLAGS) dil_tal_nv.cu

main.o: main.cu tensor_algebra.h
	$(CUDA_C) $(MPI_INC) $(CUDA_INC) $(CUDA_FLAGS) main.cu

clean:
	rm *.o *.mod *.modmic *.ptx *.x
