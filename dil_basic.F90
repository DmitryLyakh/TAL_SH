!BASIC FORTRAN PARAMETERS (Fortran-2003)
!REVISION: 2019/03/05

!Copyright (C) 2014-2019 Dmitry I. Lyakh (Liakh)
!Copyright (C) 2014-2019 Oak Ridge National Laboratory (UT-Battelle)

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
#ifndef NO_OMP
        use omp_lib
#endif
        implicit none
        public

!BASIC TYPE KINDS:
        integer(C_INT), parameter, public:: INTB=1   !byte integer size
        integer(C_INT), parameter, public:: INTS=2   !short integer size
        integer(C_INT), parameter, public:: INTD=4   !default integer size
        integer(C_INT), parameter, public:: INTL=8   !long integer size
        integer(C_INT), parameter, public:: REALH=2  !half real size
        integer(C_INT), parameter, public:: REALS=4  !short real size
        integer(C_INT), parameter, public:: REALD=8  !default real size
        integer(C_INT), parameter, public:: REALL=16 !long real size
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: INTB,INTS,INTD,INTL,REALH,REALS,REALD,REALL
!DIR$ ATTRIBUTES ALIGN:128:: INTB,INTS,INTD,INTL,REALH,REALS,REALD,REALL
#endif

!BASIC ALIASES (keep consistent with tensor_algebra.h):
        integer(C_INT), parameter, public:: NOPE=0                !"NO" answer
        integer(C_INT), parameter, public:: YEP=1                 !"YES" answer
        integer(C_INT), parameter, public:: SUCCESS=0             !success status
        integer(C_INT), parameter, public:: GENERIC_ERROR=-666    !generic error status
        integer(C_INT), parameter, public:: TRY_LATER=-918273645  !resources are currently busy: KEEP THIS UNIQUE!
        integer(C_INT), parameter, public:: NOT_CLEAN=-192837465  !resource release did not go cleanly but you may continue: KEEP THIS UNIQUE!
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: NOPE,YEP,SUCCESS,GENERIC_ERROR,TRY_LATER,NOT_CLEAN
!DIR$ ATTRIBUTES ALIGN:128:: NOPE,YEP,SUCCESS,GENERIC_ERROR,TRY_LATER,NOT_CLEAN
#endif

!BASIC NUMERIC DATA KINDS (keep consistent with tensor_algebra.h):
        integer(C_INT), parameter, public:: NO_TYPE=0 !no type/kind
        integer(C_INT), parameter, public:: R2=2      !half-precision float tensor data kind
        integer(C_INT), parameter, public:: R4=4      !single-precision float tensor data kind
        integer(C_INT), parameter, public:: R8=8      !double-precision float tensor data kind
!       integer(C_INT), parameter, public:: R16=10    !quadruple-precision float tensor data kind
        integer(C_INT), parameter, public:: C2=12     !half-precision complex float tensor data kind
        integer(C_INT), parameter, public:: C4=14     !single-precision complex float tensor data kind
        integer(C_INT), parameter, public:: C8=18     !double-precision complex float tensor data kind
!       integer(C_INT), parameter, public:: C16=20    !quadruple-precision complex float tensor data kind
        real(4), parameter, public:: R4_=0.0
        real(8), parameter, public:: R8_=0d0
        complex(4), parameter, public:: C4_=(0.0,0.0)
        complex(8), parameter, public:: C8_=(0d0,0d0)
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: NO_TYPE,R2,R4,R8,C2,C4,C8,R4_,R8_,C4_,C8_
!DIR$ ATTRIBUTES ALIGN:128:: NO_TYPE,R2,R4,R8,C2,C4,C8,R4_,R8_,C4_,C8_
#endif

!BASIC ERROR CLASSES:
        integer(INTD), parameter, public:: ERR_INVALID_ARGS=-1       !invalid procedure arguments
        integer(INTD), parameter, public:: ERR_INVALID_REQUEST=-2    !invalid request (in a context)
        integer(INTD), parameter, public:: ERR_INVALID_OBJECT=-3     !invalid object (in a context)
        integer(INTD), parameter, public:: ERR_CORRUPTED_DATA=-4     !corrupted data detected
        integer(INTD), parameter, public:: ERR_UNABLE_TO_COMPLETE=-5 !unable to complete the action/request
        integer(INTD), parameter, public:: ERR_MEM_ALLOC_FAIL=-6     !memory allocation failed
        integer(INTD), parameter, public:: ERR_MEM_FREE_FAIL=-7      !memory deallocation failed
        integer(INTD), parameter, public:: ERR_RESOURCE_UNAVAIL=-8   !resource shortage
        integer(INTD), parameter, public:: ERR_FILE_IO_FAIL=-9       !file I/O failed
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: ERR_INVALID_ARGS,ERR_INVALID_REQUEST,ERR_INVALID_OBJECT,ERR_CORRUPTED_DATA
!DIR$ ATTRIBUTES OFFLOAD:mic:: ERR_UNABLE_TO_COMPLETE,ERR_MEM_ALLOC_FAIL,ERR_MEM_FREE_FAIL
!DIR$ ATTRIBUTES OFFLOAD:mic:: ERR_RESOURCE_UNAVAIL,ERR_FILE_IO_FAIL
!DIR$ ATTRIBUTES ALIGN:128:: ERR_INVALID_ARGS,ERR_INVALID_REQUEST,ERR_INVALID_OBJECT,ERR_CORRUPTED_DATA
!DIR$ ATTRIBUTES ALIGN:128:: ERR_UNABLE_TO_COMPLETE,ERR_MEM_ALLOC_FAIL,ERR_MEM_FREE_FAIL
!DIR$ ATTRIBUTES ALIGN:128:: ERR_RESOURCE_UNAVAIL,ERR_FILE_IO_FAIL
#endif

!COMPARISON RESULTS:
        integer(INTD), parameter, public:: CMP_EQ=0             !equivalence
        integer(INTD), parameter, public:: CMP_LT=-1            !less than
        integer(INTD), parameter, public:: CMP_GT=+1            !greater than
        integer(INTD), parameter, public:: CMP_IN=-2            !contained (in) -> less
        integer(INTD), parameter, public:: CMP_CN=+2            !contains -> greater
        integer(INTD), parameter, public:: CMP_OV=-5            !overlap (whatever it means)
        integer(INTD), parameter, public:: CMP_NC=-6            !not comparable
        integer(INTD), parameter, public:: CMP_ER=GENERIC_ERROR !error
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: CMP_EQ,CMP_LT,CMP_GT,CMP_IN,CMP_CN,CMP_OV,CMP_NC,CMP_ER
!DIR$ ATTRIBUTES ALIGN:128:: CMP_EQ,CMP_LT,CMP_GT,CMP_IN,CMP_CN,CMP_OV,CMP_NC,CMP_ER
#endif

!DEVICE KINDS (keep consistent with tensor_algebra.h):
        integer(C_INT), parameter, public:: MAX_GPUS_PER_NODE=8   !max number of NVidia GPUs on a node
        integer(C_INT), parameter, public:: MAX_MICS_PER_NODE=8   !max number of Intel MICs on a node
        integer(C_INT), parameter, public:: MAX_AMDS_PER_NODE=8   !max number of AMD GPUs on a node
        integer(C_INT), parameter, public:: DEV_NULL=-1           !abstract null device
        integer(C_INT), parameter, public:: DEV_DEFAULT=DEV_NULL  !will allow runtime to choose the device
        integer(C_INT), parameter, public:: DEV_HOST=0            !multicore CPU Host (includes all self-hosted systems)
        integer(C_INT), parameter, public:: DEV_NVIDIA_GPU=1      !NVIDIA GPU
        integer(C_INT), parameter, public:: DEV_INTEL_MIC=2       !Intel Xeon Phi
        integer(C_INT), parameter, public:: DEV_AMD_GPU=3         !AMD GPU
        integer(C_INT), parameter, public:: DEV_MAX=1+MAX_GPUS_PER_NODE+MAX_MICS_PER_NODE+MAX_AMDS_PER_NODE

!BASIC NUMERIC CONSTANTS:
        real(4), parameter, public:: EPS4=epsilon(1.0) !single precision epsilon
        real(8), parameter, public:: EPS8=epsilon(1d0) !double precision epsilon
        real(8), parameter, public:: ZERO_THRESH=1d-11 !numerical comparison threshold: should account for possible round-off errors
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: EPS4,EPS8,ZERO_THRESH
!DIR$ ATTRIBUTES ALIGN:128:: EPS4,EPS8,ZERO_THRESH
#endif

!BASIC CONSTANTS:
        character(C_CHAR), parameter, public:: CHAR_NULL=achar(0) !null character

!OBJECT LOCK:
        type, public:: object_lock_t
#ifndef NO_OMP
         integer(omp_nest_lock_kind), private:: lock_omp=-1
         integer, private:: initialized=-1
#endif
         contains
          procedure, private:: ObjectLockCtorCopy
          generic, public:: assignment(=)=>ObjectLockCtorCopy
          procedure, public:: lock=>ObjectLockLock
          procedure, public:: unlock=>ObjectLockUnlock
          procedure, public:: clear=>ObjectLockClear
          final:: object_lock_dtor
        end type object_lock_t
        private ObjectLockCtorCopy
        private ObjectLockLock
        private ObjectLockUnlock
        private ObjectLockClear
        public object_lock_dtor
        type(object_lock_t), save, protected:: object_lock_null !empty object lock

       contains

![object_lock_t]==================================
        subroutine ObjectLockCtorCopy(this,source)
         implicit none
         class(object_lock_t), intent(out):: this
         class(object_lock_t), intent(in):: source
#ifndef NO_OMP
!$OMP ATOMIC WRITE
         this%initialized=-1
!$OMP FLUSH(this)
#endif
         return
        end subroutine ObjectLockCtorCopy
!--------------------------------------
        subroutine ObjectLockLock(this)
         implicit none
         class(object_lock_t), intent(inout):: this
         integer:: init_state
#ifndef NO_OMP
!$OMP ATOMIC CAPTURE
         init_state=this%initialized
         this%initialized=max(this%initialized,0)
!$OMP END ATOMIC
!$OMP FLUSH(this)
         if(init_state.lt.0) then
          !write(*,'("Initializing object lock ",i18," by thread ",i4)') this%lock_omp,omp_get_thread_num() !debug
          call omp_init_nest_lock(this%lock_omp)
!$OMP ATOMIC WRITE
          this%initialized=1
!$OMP FLUSH(this)
          !write(*,'("Locking/init object lock ",i18," by thread ",i4)') this%lock_omp,omp_get_thread_num() !debug
         elseif(init_state.eq.0) then
          !write(*,'("Locking/wait object lock ",i18," by thread ",i4)') this%lock_omp,omp_get_thread_num() !debug
          do while(init_state.eq.0)
!$OMP ATOMIC READ
           init_state=this%initialized
          enddo
         !else
          !write(*,'("Locking/ready object lock ",i18," by thread ",i4)') this%lock_omp,omp_get_thread_num() !debug
         endif
         call omp_set_nest_lock(this%lock_omp)
         !write(*,'("Lock ",i18," locked by thread ",i4)') this%lock_omp,omp_get_thread_num() !debug
#endif
         return
        end subroutine ObjectLockLock
!----------------------------------------
        subroutine ObjectLockUnlock(this)
         implicit none
         class(object_lock_t), intent(inout):: this
         integer:: init_state,ax,bx
#ifndef NO_OMP
!$OMP FLUSH(this)
!$OMP ATOMIC READ
         init_state=this%initialized
         if(init_state.gt.0) then
          !write(*,'("Unlocking object lock ",i18," by thread ",i4)') this%lock_omp,omp_get_thread_num() !debug
          call omp_unset_nest_lock(this%lock_omp)
          !write(*,'("Lock ",i18," unlocked by thread ",i4)') this%lock_omp,omp_get_thread_num() !debug
         else
          ax=0; bx=1; bx=bx/ax !crash
         endif
#endif
         return
        end subroutine ObjectLockUnlock
!---------------------------------------
        subroutine ObjectLockClear(this)
         implicit none
         class(object_lock_t), intent(inout):: this
#ifndef NO_OMP
!$OMP ATOMIC WRITE
         this%initialized=-1
!$OMP FLUSH(this)
#endif
         return
        end subroutine ObjectLockClear
!----------------------------------------
        subroutine object_lock_dtor(this)
         implicit none
         type(object_lock_t):: this
         integer:: init_state
#ifndef NO_OMP
         init_state=0
!$OMP FLUSH(this)
         do while(init_state.eq.0)
!$OMP ATOMIC READ
          init_state=this%initialized
         enddo
         if(init_state.gt.0) then
!$OMP ATOMIC WRITE
          this%initialized=-1
!$OMP FLUSH(this)
          !write(*,'("Lock ",i18," destroyed by thread ",i4)') this%lock_omp,omp_get_thread_num() !debug
          call omp_destroy_nest_lock(this%lock_omp)
         endif
#endif
         return
        end subroutine object_lock_dtor

       end module dil_basic
