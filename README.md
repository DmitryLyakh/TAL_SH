TAL-SH: Tensor Algebra Library for Shared Memory Computers:
        Nodes equipped with multicore CPU, Nvidia GPU, and
        Intel Xeon Phi (in progress).
Author: Dmitry I. Lyakh (Liakh): quant4me@gmail.com, liakhdi@ornl.gov

Copyright (C) 2014-2016 Dmitry I. Lyakh (Liakh)
Copyright (C) 2014-2016 Oak Ridge National Laboratory (UT-Battelle)

LICENSE: GNU Lesser General Public License v.3

API reference manual: DOC/TALSH_manual.txt

BUILD
-----

Makefile based: Modify the header of the Makefile accordingly and run make.

CMake Build:
  cd TAL_SH
  mkdir build && cd build
  cmake .. -DTALSH_GPU=ON -DTALSH_GPU_ARCH=60 -DMPI_INCLUDE_PATH=/opt/mpich
  make && make install

  Note: The cmake variables TALSH_BLAS, TALSH_FINE_TIMING are optional. To use them:
  cmake .. -DTALSH_GPU=ON -DTALSH_GPU_ARCH=60 -DMPI_INCLUDE_PATH=/opt/mpich
           -DTALSH_BLAS=ON -DTALSH_FINE_TIMING=ON

