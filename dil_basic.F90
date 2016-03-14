!Copyright (C) 2014-2016 Dmitry I. Lyakh (Liakh)
!Copyright (C) 2014-2016 Oak Ridge National Laboratory (UT-Battelle)

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

!TENSOR DATA KINDS (keep consistent with tensor_algebra.h):
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

!BASIC ALIASES (keep consistent with tensor_algebra.h):
        integer(C_INT), parameter, public:: NOPE=0                  !"NO" answer
        integer(C_INT), parameter, public:: YEP=1                   !"YES" answer
        integer(C_INT), parameter, public:: SUCCESS=0               !generic success
        integer(C_INT), parameter, public:: GENERIC_ERROR=-666      !generic error
        integer(C_INT), parameter, public:: TRY_LATER=-918273645    !resources are currently busy: KEEP THIS UNIQUE!
        integer(C_INT), parameter, public:: NOT_CLEAN=-192837465    !resource release did not go cleanly but you may continue: KEEP THIS UNIQUE!
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: NOPE,YEP,SUCCESS,GENERIC_ERROR,TRY_LATER,NOT_CLEAN
!DIR$ ATTRIBUTES ALIGN:128:: NOPE,YEP,SUCCESS,GENERIC_ERROR,TRY_LATER,NOT_CLEAN
#endif
       end module dil_basic
