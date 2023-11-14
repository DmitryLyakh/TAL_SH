TAL-SH: Tensor Algebra Library for Shared Memory Computers:
        Nodes equipped with multicore CPU, NVIDIA GPU, AMD GPU,
        and Intel Xeon Phi. The library implements basic tensor
        algebra operations with interfaces to C, C++11, and Fortran 90+.
Author: Dmitry I. Lyakh (Liakh): quant4me@gmail.com

Copyright (C) 2014-2022 Dmitry I. Lyakh (Liakh)
Copyright (C) 2014-2022 Oak Ridge National Laboratory (UT-Battelle)

LICENSE: BSD 3-Clause

API reference manual: DOC/TALSH_manual.txt

BUILD: Modify the header of the Makefile accordingly and run make.
If your system has MODULES, look up relevant paths via "module show",
otherwise find the paths to BLAS and CUDA yourself. If you use ESSL
and/or IBM XL compiler, you will need IBM XL paths as well. On Cray
systems, you may also need to activate dynamic linking explicitly:
export XTPE_LINK_TYPE = dynamic
export CRAYPE_LINK_TYPE = dynamic

Typical build configuration examples:

GNU compiler, OpenBLAS:
```bash
$ export BLASLIB = OPENBLAS
$ export PATH_BLAS_OPENBLAS = /usr/local/blas/openblas/lib
$ export PATH_LAPACK_LIB = $PATH_BLAS_OPENBLAS
$ make
```

GNU compiler, MKL (Intel CPU or Intel Xeon Phi):
```bash
$ export BLASLIB = MKL
$ export PATH_INTEL = /opt/intel
$ export PATH_BLAS_MKL = $PATH_INTEL/mkl/lib/intel64
$ export PATH_LAPACK_LIB = $PATH_BLAS_MKL
$ make
```

GNU compiler, OpenBLAS, CUDA (NVIDIA GPU):
```bash
$ export BLASLIB = OPENBLAS
$ export PATH_BLAS_OPENBLAS = /usr/local/blas/openblas/lib
$ export PATH_LAPACK_LIB = $PATH_BLAS_OPENBLAS
$ export GPU_CUDA = CUDA
$ export GPU_SM_ARCH = 70
$ export PATH_CUDA = /usr/local/cuda
$ make
```

GNU compiler, OpenBLAS, ROCM (AMD GPU):
```bash
$ export BLASLIB = OPENBLAS
$ export PATH_BLAS_OPENBLAS = /usr/local/blas/openblas/lib
$ export PATH_LAPACK_LIB = $PATH_BLAS_OPENBLAS
$ export GPU_CUDA = CUDA
$ export USE_HIP = YES
$ export PATH_ROCM = /opt/rocm
$ make
```
Note: ROCM versions 5.6+ should be used. Earlier ROCM versions may result in code that compiles but may show runtime errors (e.g. HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION). A potential workaround (ROCM 5.3/5.4) is to reduce hipcc optimization level to -O1.

EXAMPLES: A number of examples is available in test.cpp and main.F90.
