!BASIC FORTRAN PARAMETERS
!REVISION: 2017/03/01

!Copyright (C) 2014-2017 Dmitry I. Lyakh (Liakh)
!Copyright (C) 2014-2017 Oak Ridge National Laboratory (UT-Battelle)

!This file is part of ExaTensor.

!ExaTensor is free software: you can redistribute it and/or modify
!it under the terms of the GNU Lesser General Public License as published
!by the Free Software Foundation, either version 3 of the License, or
!(at your option) any later version.

!ExaTensor is distributed in the hope that it will be useful,
!but WITHOUT ANY WARRANTY; without even the implied warranty of
!MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
!GNU Lesser General Public License for more details.

!You should have received a copy of the GNU Lesser General Public License
!along with ExaTensor. If not, see <http://www.gnu.org/licenses/>.

       module dil_basic
        use, intrinsic:: ISO_C_BINDING
        implicit none
        public
!BASIC TYPE KINDS:
        integer(C_INT), parameter, public:: INTB=1   !byte integer size
        integer(C_INT), parameter, public:: INTS=2   !short integer size
        integer(C_INT), parameter, public:: INTD=4   !default integer size
        integer(C_INT), parameter, public:: INTL=8   !long integer size
        integer(C_INT), parameter, public:: REALS=4  !short real size
        integer(C_INT), parameter, public:: REALD=8  !default real size
        integer(C_INT), parameter, public:: REALL=16 !long real size
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: INTB,INTS,INTD,INTL,REALS,REALD,REALL
!DIR$ ATTRIBUTES ALIGN:128:: INTB,INTS,INTD,INTL,REALS,REALD,REALL
#endif

!BASIC ALIASES (keep consistent with tensor_algebra.h):
        integer(C_INT), parameter, public:: NOPE=0                  !"NO" answer
        integer(C_INT), parameter, public:: YEP=1                   !"YES" answer
        integer(C_INT), parameter, public:: SUCCESS=0               !success status
        integer(C_INT), parameter, public:: GENERIC_ERROR=-666      !generic error
        integer(C_INT), parameter, public:: TRY_LATER=-918273645    !resources are currently busy: KEEP THIS UNIQUE!
        integer(C_INT), parameter, public:: NOT_CLEAN=-192837465    !resource release did not go cleanly but you may continue: KEEP THIS UNIQUE!
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: NOPE,YEP,SUCCESS,GENERIC_ERROR,TRY_LATER,NOT_CLEAN
!DIR$ ATTRIBUTES ALIGN:128:: NOPE,YEP,SUCCESS,GENERIC_ERROR,TRY_LATER,NOT_CLEAN
#endif

!COMPARISON:
        integer(INTD), parameter, public:: CMP_EQ=0             !equivalence
        integer(INTD), parameter, public:: CMP_LT=-1            !less than
        integer(INTD), parameter, public:: CMP_GT=+1            !greater than
        integer(INTD), parameter, public:: CMP_IN=-2            !contained (in) -> less
        integer(INTD), parameter, public:: CMP_CN=+2            !contains -> greater
        integer(INTD), parameter, public:: CMP_OV=-5            !overlaps
        integer(INTD), parameter, public:: CMP_NC=-6            !not comparable
        integer(INTD), parameter, public:: CMP_ER=GENERIC_ERROR !error
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: CMP_EQ,CMP_LT,CMP_GT,CMP_IN,CMP_CN,CMP_OV,CMP_NC,CMP_ER
!DIR$ ATTRIBUTES ALIGN:128:: CMP_EQ,CMP_LT,CMP_GT,CMP_IN,CMP_CN,CMP_OV,CMP_NC,CMP_ER
#endif

!BASIC DATA KINDS (keep consistent with tensor_algebra.h):
        integer(C_INT), parameter, public:: NO_TYPE=0 !no type/kind
        integer(C_INT), parameter, public:: R4=4      !float tensor data kind
        integer(C_INT), parameter, public:: R8=8      !double tensor data kind
!       integer(C_INT), parameter, public:: R16=10    !long double tensor data kind
        integer(C_INT), parameter, public:: C4=12     !float complex tensor data kind
        integer(C_INT), parameter, public:: C8=16     !double complex tensor data kind
!       integer(C_INT), parameter, public:: C16=32    !long double complex tensor data kind
        real(4), parameter, public:: R4_=0.0
        real(8), parameter, public:: R8_=0d0
        complex(4), parameter, public:: C4_=(0.0,0.0)
        complex(8), parameter, public:: C8_=(0d0,0d0)
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: NO_TYPE,R4,R8,C4,C8,R4_,R8_,C4_,C8_
!DIR$ ATTRIBUTES ALIGN:128:: NO_TYPE,R4,R8,C4,C8,R4_,R8_,C4_,C8_
#endif

!DEVICE KINDS (keep consistent with tensor_algebra.h):
        integer(C_INT), parameter, public:: MAX_GPUS_PER_NODE=8   !max number of NVidia GPUs on a node
        integer(C_INT), parameter, public:: MAX_MICS_PER_NODE=8   !max number of Intel MICs on a node
        integer(C_INT), parameter, public:: MAX_AMDS_PER_NODE=8   !max number of AMD GPUs on a node
        integer(C_INT), parameter, public:: DEV_NULL=-1           !abstract null device
        integer(C_INT), parameter, public:: DEV_DEFAULT=DEV_NULL  !will allow runtime to choose the device
        integer(C_INT), parameter, public:: DEV_HOST=0            !multicore CPU Host (includes all self-hosted systems)
        integer(C_INT), parameter, public:: DEV_NVIDIA_GPU=1      !NVidia GPU
        integer(C_INT), parameter, public:: DEV_INTEL_MIC=2       !Intel Xeon Phi
        integer(C_INT), parameter, public:: DEV_AMD_GPU=3         !AMD GPU
        integer(C_INT), parameter, public:: DEV_MAX=1+MAX_GPUS_PER_NODE+MAX_MICS_PER_NODE+MAX_AMDS_PER_NODE

!BASIC NUMERIC:
        real(4), parameter, public:: EPS4=epsilon(1.0) !single precision epsilon
        real(8), parameter, public:: EPS8=epsilon(1d0) !double precision epsilon
        real(8), parameter, public:: ZERO_THRESH=1d-11 !numerical comparison threshold: should account for possible round-off errors
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: EPS4,EPS8,ZERO_THRESH
!DIR$ ATTRIBUTES ALIGN:128:: EPS4,EPS8,ZERO_THRESH
#endif
       end module dil_basic
