TAL-SH: Tensor Algebra Library for Shared Memory Computers:
        Nodes equipped with multicore CPU, Nvidia GPU, and
        Intel Xeon Phi. The library implements basic tensor
        algebra operations with interfaces to C, C++11, and Fortran 90+.
Author: Dmitry I. Lyakh (Liakh): quant4me@gmail.com, liakhdi@ornl.gov

Copyright (C) 2014-2016 Dmitry I. Lyakh (Liakh)
Copyright (C) 2014-2016 Oak Ridge National Laboratory (UT-Battelle)

LICENSE: GNU Lesser General Public License v.3

API reference manual: DOC/TALSH_manual.txt

BUILD: Modify the header of the Makefile accordingly and run make.
If your system has MODULES, look up relevant paths via "module show",
otherwise find the paths to BLAS and CUDA yourself. If you use ESSL
and/or IBM XL compiler, you will need IBM XL paths as well.
