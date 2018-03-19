!TALSH::Fortran API testing.

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

        program main
        use, intrinsic:: ISO_C_BINDING
        implicit none
        logical, parameter:: TEST_NVTAL=.FALSE.
        logical, parameter:: TEST_TALSH=.TRUE.
        logical, parameter:: TEST_NWCHEM=.TRUE.
        logical, parameter:: TEST_COMPLEX=.TRUE.
        logical, parameter:: BENCH_TALSH_RND=.FALSE.
        logical, parameter:: BENCH_TALSH_CUSTOM=.FALSE.

        interface

         subroutine test_talsh_c(ierr) bind(c)
          import
          integer(C_INT), intent(out):: ierr
         end subroutine test_talsh_c

         subroutine test_talsh_cxx(ierr) bind(c)
          import
          integer(C_INT), intent(out):: ierr
         end subroutine test_talsh_cxx

         subroutine test_nwchem_c(ierr) bind(c)
          import
          integer(C_INT), intent(out):: ierr
         end subroutine test_nwchem_c

#ifndef NO_GPU
         subroutine test_nvtal_c(ierr) bind(c)
          import
          integer(C_INT), intent(out):: ierr
         end subroutine test_nvtal_c
#endif
        end interface

        integer(C_INT):: ierr

!Test NV-TAL C/C++ API interface:
#ifndef NO_GPU
        if(TEST_NVTAL) then
         write(*,'("Testing NV-TAL C/C++ API ...")')
         call test_nvtal_c(ierr)
         write(*,'("Done: Status ",i5)') ierr
         if(ierr.ne.0) stop
         write(*,*)''
        endif
#endif
!Test TAL-SH C/C++ API interface:
        if(TEST_TALSH) then
         write(*,'("Testing TAL-SH C/C++ API ...")')
         call test_talsh_c(ierr)
         write(*,'("Done: Status ",i5)') ierr
         if(ierr.ne.0) stop
         write(*,*)''
         write(*,'("Testing TAL-SH C++11 API ...")')
         call test_talsh_cxx(ierr)
         write(*,'("Done: Status ",i5)') ierr
         if(ierr.ne.0) stop
         write(*,*)''
!Test TAL-SH Fortran API interface:
         write(*,'("Testing TAL-SH Fortran API ...")')
         call test_talsh_f(ierr)
         write(*,'("Done: Status ",i5)') ierr
         if(ierr.ne.0) stop
         write(*,*)''
        endif
!Test TAL-SH tensor contractions for NWChem:
        if(TEST_NWCHEM) then
         write(*,'("Testing TAL-SH tensor contractions for NWChem ...")')
         call test_nwchem_c(ierr)
         write(*,'("Done: Status ",i5)') ierr
         if(ierr.ne.0) stop
         write(*,*)''
        endif
!Test TAL-SH complex field operations:
        if(TEST_COMPLEX) then
         write(*,'("Testing TAL-SH complex tensor operations ...")')
         call test_talsh_cmplx_f(ierr)
         write(*,'("Done: Status ",i5)') ierr
         if(ierr.ne.0) stop
         write(*,*)''
        endif
!Benchmark tensor contraction performance:
 !Random test:
        if(BENCH_TALSH_RND) then
         write(*,'("Benchmarking tensor contraction performance (random) ...")')
         call benchmark_tensor_contractions_rnd(ierr)
         write(*,'("Done: Status ",i5)') ierr
         if(ierr.ne.0) stop
         write(*,*)''
        endif
 !Custom test:
        if(BENCH_TALSH_CUSTOM) then
         write(*,'("Benchmarking tensor contraction performance (custom) ...")')
         call benchmark_tensor_contractions_ctm(ierr)
         write(*,'("Done: Status ",i5)') ierr
         if(ierr.ne.0) stop
         write(*,*)''
        endif
        stop
        end program main
!------------------------------------
        subroutine test_talsh_f(ierr)
!Testing device-unified TAL-SH Fortran API.
        use, intrinsic:: ISO_C_BINDING
        use tensor_algebra
        use talsh
        use stsubs
        implicit none
        integer(C_SIZE_T), parameter:: BUF_SIZE=1_8*1024_8*1024_8*1024_8 !desired Host argument buffer size in bytes
        integer(C_INT), parameter:: MAX_TENSORS=24
        integer(C_INT), parameter:: MAX_TASKS=16
        integer(C_INT), parameter:: DIM_EXT=int(real(BUF_SIZE/8/(MAX_TENSORS+2),8)**(0.25d0)) !tensor dimension extent
        integer(C_SIZE_T):: host_buf_size
        integer(C_INT):: i,j,k,l,m,n,ierr,num_gpus,host_arg_max,sta
        integer(C_INT):: sts(MAX_TASKS)        !task statuses
        type(talsh_task_t):: tsks(MAX_TASKS)   !task handles
        type(talsh_tens_t):: tens(MAX_TENSORS) !tensors
        complex(8):: cval

        ierr=0
!Check GPU availability:
#ifndef NO_GPU
        write(*,'(1x,"Checking Nvidia GPU availability ... ")',ADVANCE='NO')
        ierr=cuda_get_device_count(num_gpus)
        write(*,'("Status ",i11,": Number of GPUs = ",i3)') ierr,num_gpus
        if(ierr.ne.TALSH_SUCCESS) then; ierr=1; return; endif
#else
        num_gpus=0
#endif
        if(MAX_TENSORS.lt.3*2*num_gpus) then; ierr=2; return; endif
!Initialize TALSH runtime:
        write(*,'(1x,"Initializing TALSH ... ")',ADVANCE='NO')
        host_buf_size=BUF_SIZE
#ifndef NO_GPU
        ierr=talsh_init(host_buf_size,host_arg_max,gpu_list=(/(i,i=0,num_gpus-1)/))
#else
        ierr=talsh_init(host_buf_size,host_arg_max)
#endif
        write(*,'("Status ",i11,": Size (Bytes) = ",i13,": Max args in HAB = ",i7)') ierr,host_buf_size,host_arg_max
        if(ierr.ne.TALSH_SUCCESS) then; ierr=3; return; endif

!Create tensors on Host and initialize them to a value:
        write(*,'(" Tensor dimension extent = ",i5)') DIM_EXT
        do i=1,MAX_TENSORS
         write(*,'(1x,"Constructing tensor block ",i2," ... ")',ADVANCE='NO') i
         select case(mod(i,3))
         case(1)
          cval=(0d0,0d0)
         case(2)
          cval=(1d-2,0d0)
         case(0)
          cval=(1d-3,0d0)
         end select
         ierr=talsh_tensor_construct(tens(i),R8,(/DIM_EXT,DIM_EXT,DIM_EXT,DIM_EXT/),init_val=cval)
         write(*,'("Status ",i11)') ierr; if(ierr.ne.TALSH_SUCCESS) then; ierr=4; return; endif
        enddo

        n=0 !number of tasks scheduled
!Stream tensor contractions on all GPUs:
#ifndef NO_GPU
        sts(:)=TALSH_TASK_EMPTY
        j=0; k=0
        sloop: do
         k=k+1
         do i=0,num_gpus-1
          if(j+3.gt.MAX_TENSORS) exit sloop
          n=n+1
          write(*,'(1x,"Scheduling tensor contraction ",i2," on GPU ",i2,"... ")',ADVANCE='NO') n,i
          ierr=talsh_tensor_contract('D(a,b,i,j)+=L(j,k,c,i)*R(c,b,k,a)',tens(j+1),tens(j+2),tens(j+3),&
                                    &dev_id=talsh_flat_dev_id(DEV_NVIDIA_GPU,i),copy_ctrl=COPY_TTT,talsh_task=tsks(n))
          if(ierr.ne.TRY_LATER) then
           write(*,'("Status ",i11)') ierr
           if(ierr.ne.TALSH_SUCCESS) then; ierr=5; return; endif
           sts(n)=TALSH_TASK_SCHEDULED
!          call talsh_task_print_info(tsks(n)) !debug
           j=j+3 !next set of tensor arguments
          else
           write(*,'("Deferred")')
           ierr=talsh_task_destruct(tsks(n))
           if(ierr.ne.TALSH_SUCCESS) then; print *,ierr; ierr=6; return; endif
           n=n-1
           call wait_delay(0.1) !seconds
          endif
         enddo
         if(k.gt.1) then
          do l=1,n
           if(sts(l).ne.TALSH_TASK_COMPLETED.and.sts(l).ne.TALSH_TASK_ERROR) then
            sta=talsh_task_complete(tsks(l),sts(l),ierr)
            if(ierr.ne.TALSH_SUCCESS) then; ierr=7; return; endif
            if(sta.eq.YEP) write(*,'("Task ",i3," completed with status ",i9)') l,sts(l)
           endif
          enddo
         endif
        enddo sloop
!Synchronize and compare the results:
        write(*,'(1x,"Waiting upon completion of tensor contractions on all GPUs ... ")',ADVANCE='NO')
        ierr=talsh_tasks_wait(n,tsks,sts)
        write(*,'("Status ",i11," Completion =",8(1x,i8))') ierr,sts(1:n)
        if(ierr.ne.TALSH_SUCCESS) then; ierr=8; return; endif
!Printing results:
        do i=1,MAX_TENSORS,3
         call talsh_tensor_print_info(tens(i))
         print *,'TENSOR ',i,' NORM1 = ',talshTensorImageNorm1_cpu(tens(i))
        enddo
#else
!Execute a tensor contraction on CPU:
        n=n+1
        write(*,'(1x,"Executing tensor contraction ",i2," on CPU ... ")',ADVANCE='NO') n
        ierr=talsh_tensor_contract('D(a,b,i,j)+=L(j,k,c,i)*R(c,b,k,a)',tens(1),tens(2),tens(3),&
                                  &dev_id=talsh_flat_dev_id(DEV_HOST,0),talsh_task=tsks(n))
        write(*,'("Status ",i11)') ierr; if(ierr.ne.TALSH_SUCCESS) then; ierr=9; return; endif
!       call talsh_task_print_info(tsks(n)) !debug
        write(*,'(1x,"Waiting upon completion of tensor contractions on CPU ... ")',ADVANCE='NO')
        ierr=talsh_tasks_wait(n,tsks,sts)
        write(*,'("Status ",i11," Completion =",8(1x,i8))') ierr,sts(1:n)
        if(ierr.ne.TALSH_SUCCESS) then; ierr=10; return; endif
#endif

!Destruct TAL-SH task handles:
        do i=n,1,-1
         write(*,'(1x,"Destructing TAL-SH task handle ",i2," ... ")',ADVANCE='NO') i
         ierr=talsh_task_destruct(tsks(i))
         write(*,'("Status ",i11)') ierr; if(ierr.ne.TALSH_SUCCESS) then; ierr=11; return; endif
        enddo

!Destroy tensors:
        do i=MAX_TENSORS,1,-1
         write(*,'(1x,"Destructing tensor block ",i2," ... ")',ADVANCE='NO') i
         ierr=talsh_tensor_destruct(tens(i))
         write(*,'("Status ",i11)') ierr; if(ierr.ne.TALSH_SUCCESS) then; ierr=12; return; endif
        enddo
!Print run-time statistics:
        ierr=talsh_stats()
        if(ierr.ne.TALSH_SUCCESS) then; ierr=13; return; endif
!Shutdown TALSH:
        write(*,'(1x,"Shutting down TALSH ... ")',ADVANCE='NO')
        ierr=talsh_shutdown()
        write(*,'("Status ",i11)') ierr
        if(ierr.ne.TALSH_SUCCESS) then; ierr=14; return; endif
        return
        end subroutine test_talsh_f
!------------------------------------------
        subroutine test_talsh_cmplx_f(ierr)
!Testing TAL-SH complex tensor operations.
        use, intrinsic:: ISO_C_BINDING
        use tensor_algebra
        use talsh
        use stsubs
        implicit none
        integer(C_SIZE_T), parameter:: BUF_SIZE=1_8*1024_8*1024_8*1024_8 !desired Host argument buffer size in bytes
        integer(C_INT), parameter:: DIM_EXT=31 !tensor dimension extent
        integer(C_SIZE_T):: host_buf_size,tens_vol
        integer(C_INT):: i,j,k,l,m,n,ierr,num_gpus,host_arg_max,sta,sts,shp(1:MAX_TENSOR_RANK)
        type(talsh_tens_t):: ltens,rtens,ctens,dtens,ptens,ntens
        type(talsh_task_t):: tsk0,tsk1
        type(C_PTR):: body_p
        complex(8), pointer, contiguous:: tens_body(:)
        complex(8):: cval,prod
        real(8):: cnrm,dnrm,rl,cx,alpha,beta

        ierr=0
!Check GPU availability:
#ifndef NO_GPU
        write(*,'(1x,"Checking Nvidia GPU availability ... ")',ADVANCE='NO')
        ierr=cuda_get_device_count(num_gpus)
        write(*,'("Status ",i11,": Number of GPUs = ",i3)') ierr,num_gpus
        if(ierr.ne.TALSH_SUCCESS) then; ierr=1; return; endif
#else
        num_gpus=0
#endif
!Initialize TALSH runtime:
        write(*,'(1x,"Initializing TALSH ... ")',ADVANCE='NO')
        host_buf_size=BUF_SIZE
#ifndef NO_GPU
        ierr=talsh_init(host_buf_size,host_arg_max,gpu_list=(/(i,i=0,num_gpus-1)/))
#else
        ierr=talsh_init(host_buf_size,host_arg_max)
#endif
        write(*,'("Status ",i11,": Size (Bytes) = ",i13,": Max args in HAB = ",i7)') ierr,host_buf_size,host_arg_max
        if(ierr.ne.TALSH_SUCCESS) then; ierr=2; return; endif

!Construct tensors:
        prod=(1d0,0d0)
        write(*,'(1x,"Constructing TAL-SH tensors: Statuses: ")',ADVANCE='NO')
        cval=(+1.234d-3,-3.985d-4); ierr=talsh_tensor_construct(ltens,C8,(/DIM_EXT,DIM_EXT,DIM_EXT,DIM_EXT/),init_val=cval)
        write(*,'(i11)',ADVANCE='NO') ierr; if(ierr.ne.TALSH_SUCCESS) then; ierr=3; return; endif
        prod=prod*cval
        cval=(-2.768d-4,+6.731d-3); ierr=talsh_tensor_construct(rtens,C8,(/DIM_EXT,DIM_EXT,DIM_EXT,DIM_EXT/),init_val=cval)
        write(*,'(i11)',ADVANCE='NO') ierr; if(ierr.ne.TALSH_SUCCESS) then; ierr=4; return; endif
        prod=prod*conjg(cval)
        cval=(0d0,0d0); ierr=talsh_tensor_construct(ctens,C8,(/DIM_EXT,DIM_EXT,DIM_EXT,DIM_EXT/),init_val=cval)
        write(*,'(i11)',ADVANCE='NO') ierr; if(ierr.ne.TALSH_SUCCESS) then; ierr=5; return; endif
        cval=(0d0,0d0); ierr=talsh_tensor_construct(dtens,C8,(/DIM_EXT,DIM_EXT,DIM_EXT,DIM_EXT/),init_val=cval)
        write(*,'(i11)',ADVANCE='NO') ierr; if(ierr.ne.TALSH_SUCCESS) then; ierr=6; return; endif
        cval=(+1d0,0d0); ierr=talsh_tensor_construct(ptens,C8,shp(1:0),init_val=cval)
        write(*,'(i11)',ADVANCE='NO') ierr; if(ierr.ne.TALSH_SUCCESS) then; ierr=7; return; endif
        cval=(-1d0,0d0); ierr=talsh_tensor_construct(ntens,C8,shp(1:0),init_val=cval)
        write(*,'(i11)',ADVANCE='NO') ierr; if(ierr.ne.TALSH_SUCCESS) then; ierr=8; return; endif
        write(*,'()')
        cnrm=talshTensorImageNorm1_cpu(ltens); dnrm=talshTensorImageNorm1_cpu(rtens)
        write(*,'(1x,"Initial tensor 1-norms: ",D25.14,1x,D25.14)') cnrm,dnrm
        ierr=talsh_tensor_get_scalar(ntens,cval); if(ierr.ne.TALSH_SUCCESS) then; ierr=9; return; endif
        write(*,*) cval

!Contract tensors:
        write(*,'(1x,"Scheduling two tensor contractions: Statuses: ")',ADVANCE='NO')
        ierr=talsh_tensor_contract('D(a,b,i,j)+=L+(j,k,c,i)*R(c,b,k,a)',ctens,ltens,rtens,scale=(5d-1,0d0),&
                                  &dev_id=talsh_flat_dev_id(DEV_HOST,0),talsh_task=tsk0)
        write(*,'(i11,1x)',ADVANCE='NO') ierr; if(ierr.ne.TALSH_SUCCESS) then; ierr=10; return; endif
        ierr=talsh_tensor_contract('D(a,b,i,j)+=L(j,k,c,i)*R+(c,b,k,a)',dtens,ltens,rtens,scale=(5d-1,0d0),&
                                  &dev_id=talsh_flat_dev_id(DEV_HOST,0),talsh_task=tsk1)
        write(*,'(i11)') ierr; if(ierr.ne.TALSH_SUCCESS) then; ierr=11; return; endif
        write(*,'(1x,"Waiting upon completion of tensor contraction 1 ... ")',ADVANCE='NO')
        ierr=talsh_task_wait(tsk0,sts); write(*,'("Status ",i11," Completion = ",i8)') ierr,sts
        if(ierr.ne.TALSH_SUCCESS) then; ierr=12; return; endif
        write(*,'(1x,"Waiting upon completion of tensor contraction 2 ... ")',ADVANCE='NO')
        ierr=talsh_task_wait(tsk1,sts); write(*,'("Status ",i11," Completion = ",i8)') ierr,sts
        if(ierr.ne.TALSH_SUCCESS) then; ierr=13; return; endif
!Destruct TAL-SH task handles:
        write(*,'(1x,"Destructing task handles: Statuses: ")',ADVANCE='NO')
        ierr=talsh_task_destruct(tsk1)
        write(*,'(i11)',ADVANCE='NO') ierr; if(ierr.ne.TALSH_SUCCESS) then; ierr=14; return; endif
        ierr=talsh_task_destruct(tsk0)
        write(*,'(i11)',ADVANCE='NO') ierr; if(ierr.ne.TALSH_SUCCESS) then; ierr=15; return; endif
        write(*,'()')
!Check 1-norms of the resulting tensors:
        cnrm=talshTensorImageNorm1_cpu(ctens); dnrm=talshTensorImageNorm1_cpu(dtens)
        write(*,'(1x,"Resulting tensor 1-norms: ",D25.14,1x,D25.14)') cnrm,dnrm
!Inspect individual tensor elements:
        prod=prod*dble(DIM_EXT)*dble(DIM_EXT) !LR+ value
        write(*,'(1x,"Getting access to ctens tensor body: ")',ADVANCE='NO')
        ierr=talsh_tensor_get_body_access(ctens,body_p,C8,0,DEV_HOST)
        tens_vol=talsh_tensor_volume(ctens)
        call c_f_pointer(body_p,tens_body,(/tens_vol/))
        write(*,'("Status ",i11,": Element value = (",(D20.14,1x,D20.14),")")') ierr,tens_body(lbound(tens_body,1))
        write(*,'(1x,"Getting access to dtens tensor body: ")',ADVANCE='NO')
        ierr=talsh_tensor_get_body_access(dtens,body_p,C8,0,DEV_HOST)
        tens_vol=talsh_tensor_volume(dtens)
        call c_f_pointer(body_p,tens_body,(/tens_vol/))
        write(*,'("Status ",i11,": Element value = (",(D20.14,1x,D20.14),")")') ierr,tens_body(lbound(tens_body,1))
        write(*,'(1x,"Reference value = (",(D20.14,1x,D20.14),")")') prod
        rl=dble(prod); cx=dimag(prod)
        alpha=(rl**2-cx**2)/(rl**2+cx**2); beta=-2d0*rl*cx/(rl**2+cx**2)

!Add the resulting tensors with a "+" sign:
        write(*,'(1x,"Adding the resulting tensors with a + sign: ")',ADVANCE='NO')
        ierr=talsh_tensor_contract('D(a,b,i,j)+=L(a,b,i,j)*R()',ctens,dtens,ptens,dev_id=talsh_flat_dev_id(DEV_HOST,0))
        write(*,'("Status ",i11)') ierr
        write(*,'(1x,"Getting access to ctens tensor body: ")',ADVANCE='NO')
        ierr=talsh_tensor_get_body_access(ctens,body_p,C8,0,DEV_HOST)
        tens_vol=talsh_tensor_volume(ctens)
        call c_f_pointer(body_p,tens_body,(/tens_vol/))
        write(*,'("Status ",i11,": Element value = (",(D20.14,1x,D20.14),")")') ierr,tens_body(lbound(tens_body,1))
!Redoing the same thing again, but via true tensor addition:
        write(*,'(1x,"Initializing the resulting tensor to zero: ")',ADVANCE='NO')
        ierr=talsh_tensor_init(ctens,(0d0,0d0),dev_id=talsh_flat_dev_id(DEV_HOST,0))
        write(*,'("Status ",i11)') ierr
        write(*,'(1x,"Adding the resulting tensors with a + sign: ")',ADVANCE='NO')
        ierr=talsh_tensor_add('D(a,b,i,j)+=L(a,b,i,j)',ctens,dtens,scale=dcmplx(alpha,beta),dev_id=talsh_flat_dev_id(DEV_HOST,0))
        write(*,'("Status ",i11)') ierr
        write(*,'(1x,"Getting access to ctens tensor body: ")',ADVANCE='NO')
        ierr=talsh_tensor_get_body_access(ctens,body_p,C8,0,DEV_HOST)
        tens_vol=talsh_tensor_volume(ctens)
        call c_f_pointer(body_p,tens_body,(/tens_vol/))
        write(*,'("Status ",i11,": Element value   = (",(D20.14,1x,D20.14),")")') ierr,tens_body(lbound(tens_body,1))
        write(*,'(1x,"Reference value = (",(D20.14,1x,D20.14),")")') prod*(5d-1,0d0)

!Destruct tensors:
        write(*,'(1x,"Destructing tensors: Statuses: ")',ADVANCE='NO')
        ierr=talsh_tensor_destruct(ntens)
        write(*,'(i11)',ADVANCE='NO') ierr; if(ierr.ne.TALSH_SUCCESS) then; ierr=16; return; endif
        ierr=talsh_tensor_destruct(ptens)
        write(*,'(i11)',ADVANCE='NO') ierr; if(ierr.ne.TALSH_SUCCESS) then; ierr=17; return; endif
        ierr=talsh_tensor_destruct(dtens)
        write(*,'(i11)',ADVANCE='NO') ierr; if(ierr.ne.TALSH_SUCCESS) then; ierr=18; return; endif
        ierr=talsh_tensor_destruct(ctens)
        write(*,'(i11)',ADVANCE='NO') ierr; if(ierr.ne.TALSH_SUCCESS) then; ierr=19; return; endif
        ierr=talsh_tensor_destruct(rtens)
        write(*,'(i11)',ADVANCE='NO') ierr; if(ierr.ne.TALSH_SUCCESS) then; ierr=20; return; endif
        ierr=talsh_tensor_destruct(ltens)
        write(*,'(i11)',ADVANCE='NO') ierr; if(ierr.ne.TALSH_SUCCESS) then; ierr=21; return; endif
        write(*,'()')

!Print run-time statistics:
        ierr=talsh_stats()
        if(ierr.ne.TALSH_SUCCESS) then; ierr=22; return; endif
!Shutdown TALSH:
        write(*,'(1x,"Shutting down TALSH ... ")',ADVANCE='NO')
        ierr=talsh_shutdown()
        write(*,'("Status ",i11)') ierr
        if(ierr.ne.TALSH_SUCCESS) then; ierr=23; return; endif
        return

        end subroutine test_talsh_cmplx_f
!---------------------------------------------------------
        subroutine benchmark_tensor_contractions_rnd(ierr)
!Benchmarks tensor contraction performance (random tensor contractions).
         use, intrinsic:: ISO_C_BINDING
         use tensor_algebra
         use talsh
         use stsubs
         use combinatoric
         implicit none
         integer(C_INT), intent(out):: ierr
         integer(C_SIZE_T), parameter:: BUF_SIZE=64_8*1024_8*1024_8 !desired Host argument buffer size in bytes, use >=5GiB for exhaustive benchmarking
         integer(C_INT), parameter:: TENS_DATA_KIND=R8 !tensor data kind (R4,R8,C4,C8)
         integer, parameter:: MAX_TENS_RANK=4          !max tensor rank (<= MAX_TENSOR_RANK), use 8 for exhaustive benchmarking
         integer, parameter:: MAX_GEN_DIMS=4           !max number of large dimensions per tensor
         integer, parameter:: SMALL_DIM_EXT=8          !extent of small dimensions
         integer, parameter:: MAX_REPEATS=64           !number of repeated tensor contractions that differ in index mapping
         logical, parameter:: DO_TWICE=.FALSE.         !if .TRUE., each tensor contraction will be executed twice
         real(C_DOUBLE), parameter:: CMP_ZERO=1D-4     !comparison threshold (relative)
         !----------------------------------------
         integer(C_INT):: i,j,k,l,m,n,num_gpus,host_arg_max,sts,rd,rl,rr,mnc,nc,ncl,ncs,nul,nus,cptrn(1:MAX_TENS_RANK*2)
         integer(C_INT):: pll(0:MAX_TENS_RANK),pls(0:MAX_TENS_RANK),prl(0:MAX_TENS_RANK),prs(0:MAX_TENS_RANK)
         integer(C_INT):: pdl(0:MAX_TENS_RANK),pds(0:MAX_TENS_RANK),nld,nll,nlr,mld,mll,mlr,ml
         integer(C_INT):: ddims(1:MAX_TENS_RANK),ldims(1:MAX_TENS_RANK),rdims(1:MAX_TENS_RANK)
         integer(C_SIZE_T):: host_buf_size,max_tens_vol,vd,vl,vr,words
         integer(8):: max_perm
         character(C_CHAR):: cptrn_sym(256)
         character(256):: str
         type(talsh_tens_t):: dtens,ltens,rtens
         type(talsh_task_t):: tsk
         complex(8):: cval
         real(C_DOUBLE):: flops,tm,tmc,tmi,tmo,tmm,gn1,cn1
         logical:: transpose,repeat

         ierr=0
         open(10,file='benchmark_tens_contr.txt',form='FORMATTED',status='UNKNOWN')
!Check Nvidia GPU availability:
#ifndef NO_GPU
         write(*,'(1x,"Checking Nvidia GPU availability ... ")',ADVANCE='NO')
         ierr=cuda_get_device_count(num_gpus)
         write(*,'("Status ",i11,": Number of GPUs = ",i3)') ierr,num_gpus
         if(ierr.ne.TALSH_SUCCESS) then; ierr=1; return; endif
#else
         num_gpus=0
#endif
!Initialize TALSH runtime:
         write(*,'(1x,"Initializing TALSH ... ")',ADVANCE='NO')
         host_buf_size=BUF_SIZE
#ifndef NO_GPU
         ierr=talsh_init(host_buf_size,host_arg_max,gpu_list=(/(i,i=0,num_gpus-1)/))
#else
         ierr=talsh_init(host_buf_size,host_arg_max)
#endif
         write(*,'("Status ",i11,": Size (Bytes) = ",i13,": Max args in HAB = ",i7)') ierr,host_buf_size,host_arg_max
         if(ierr.ne.TALSH_SUCCESS) then; ierr=2; return; endif
         if(talsh_valid_data_kind(TENS_DATA_KIND,n).eq.YEP) then
          max_tens_vol=host_buf_size/9_C_SIZE_T/int(n,C_SIZE_T) !max tensor volume
          write(*,'(" Max tensor volume (words) = ",i11," (word size = ",i2,")")') max_tens_vol,n
         else
          ierr=25; return
         endif

!Tensor contractions:
         n=0 !will be the total number of tensor contractions performed
         do rr=1,MAX_TENS_RANK !rank of the right tensor
          do rl=1,rr !rank of the left tensor
           mnc=(max(rl+rr-MAX_TENS_RANK,0)+1)/2 !min number of contracted indices
           do nc=mnc,rl !number of contracted indices
            rd=(rl+rr)-2*nc !number of uncontracted indices (rank of the destination tensor)
            do ncl=max(nc-max(rl-MAX_GEN_DIMS,0),0),min(nc,MAX_GEN_DIMS) !number of large contracted dimensions
             ncs=nc-ncl !number of small contracted dimensions
             nus=max(rl-MAX_GEN_DIMS,0)+max(rr-MAX_GEN_DIMS,0)-2*ncs !number of small uncontracted dimensions
             nul=rd-nus !number of large uncontracted dimensions
             if((nul+nus)+2*(ncl+ncs).ne.rl+rr) then; ierr=3; return; endif !error trap
 !Adjust large tensor dimensions:
             nld=nul; nll=min(rl,MAX_GEN_DIMS); nlr=min(rr,MAX_GEN_DIMS)
             ddims(1:nld)=1; ddims(nld+1:nul+nus)=SMALL_DIM_EXT
             ldims(1:nll)=1; ldims(nll+1:rl)=SMALL_DIM_EXT
             rdims(1:nlr)=1; rdims(nlr+1:rr)=SMALL_DIM_EXT
             vd=1_C_SIZE_T; do i=1,rd; vd=vd*ddims(i); enddo !preliminary volume of the destination tensor
             vl=1_C_SIZE_T; do i=1,rl; vl=vl*ldims(i); enddo !preliminary volume of the left tensor
             vr=1_C_SIZE_T; do i=1,rr; vr=vr*rdims(i); enddo !preliminary volume of the right tensor
             if(nld.gt.0) then; mld=int((dble(max_tens_vol)/dble(vd))**(1d0/dble(nld)))-1; else; mld=1; endif
             if(nll.gt.0) then; mll=int((dble(max_tens_vol)/dble(vl))**(1d0/dble(nll)))-1; else; mll=1; endif
             if(nlr.gt.0) then; mlr=int((dble(max_tens_vol)/dble(vr))**(1d0/dble(nlr)))-1; else; mlr=1; endif
             ml=max(mld,max(mll,mlr))
             if(nld.gt.0) ml=min(ml,mld)
             if(nll.gt.0) ml=min(ml,mll)
             if(nlr.gt.0) ml=min(ml,mlr)
             ddims(1:nld)=ml; ldims(1:nll)=ml; rdims(1:nlr)=ml
             vd=1_C_SIZE_T; do i=1,rd; vd=vd*ddims(i); enddo !final volume of the destination tensor
             vl=1_C_SIZE_T; do i=1,rl; vl=vl*ldims(i); enddo !final volume of the left tensor
             vr=1_C_SIZE_T; do i=1,rr; vr=vr*rdims(i); enddo !final volume of the right tensor
             words=vd+vl+vr !data size in words
             if(max(vd,max(vl,vr)).gt.max_tens_vol) then; ierr=4; return; endif
             flops=dsqrt(dble(vd)*dble(vl)*dble(vr))*2d0 !number of floating point operations (FMA = 2 Flops)
 !Benchmark different index mappings:
             max_perm=(int(noid(rl,nc),8)*int(noid(rr,nc),8)*factorial(rd)) !max number of possible permutations
             ploop: do m=1,int(min(max_perm,int(MAX_REPEATS,8)),C_INT) !explore different index mappings (permutations)
  !Select large contracted dimensions in the input tensors:
              if(ncl.gt.0) then
               call random_composition(.TRUE.,min(rl,MAX_GEN_DIMS),ncl,pll) !left large contracted indices
               call random_composition(.FALSE.,min(rr,MAX_GEN_DIMS),ncl,prl) !right large contracted indices
              endif
  !Select small contracted dimensions in the input tensors:
              if(ncs.gt.0) then
               call random_composition(.TRUE.,rl-MAX_GEN_DIMS,ncs,pls); pls(1:ncs)=pls(1:ncs)+MAX_GEN_DIMS !left small contracted indices
               call random_composition(.FALSE.,rr-MAX_GEN_DIMS,ncs,prs); prs(1:ncs)=prs(1:ncs)+MAX_GEN_DIMS !right small contracted indices
              endif
  !Map uncontracted indices onto the destination tensor dimensions:
              if(nul.gt.0) call random_permutation(nul,pdl,no_trivial=.TRUE.) !destination large index permutation
              if(nus.gt.0) call random_permutation(nus,pds,no_trivial=.TRUE.) !destination small index permutation
  !Determine the contraction pattern:
              cptrn(1:rl+rr)=0
              do i=1,ncl; cptrn(pll(i))=-prl(i); cptrn(rl+prl(i))=-pll(i); enddo !large contracted dimensions
              do i=1,ncs; cptrn(pls(i))=-prs(i); cptrn(rl+prs(i))=-pls(i); enddo !small contracted dimensions
              j=0; k=0 !j:large uncontracted counter; k:small uncontracted counter
              do i=1,rl+rr
               if(cptrn(i).eq.0) then !uncontracted dimension
                if(i.le.rl) then !left tensor
                 if(i.le.MAX_GEN_DIMS) then !large dimension
                  j=j+1; cptrn(i)=pdl(j)
                 else !small dimension
                  k=k+1; cptrn(i)=nul+pds(k)
                 endif
                else !right tensor
                 if(i-rl.le.MAX_GEN_DIMS) then !large dimension
                  j=j+1; cptrn(i)=pdl(j)
                 else !small dimension
                  k=k+1; cptrn(i)=nul+pds(k)
                 endif
                endif
               endif
              enddo
  !Check whether at least one tensor transpose is needed:
              transpose=.FALSE.
              do i=1,nc
               if(cptrn(i).ne.-i.or.cptrn(rl+i).ne.-i) then; transpose=.TRUE.; exit; endif
              enddo
              if(.not.transpose) then
               do i=1,rl-nc
                if(cptrn(nc+i).ne.i) then; transpose=.TRUE.; exit; endif
               enddo
              endif
              if(.not.transpose) then
               do i=1,rr-nc
                if(cptrn(rl+nc+i).ne.(rl-nc)+i) then; transpose=.TRUE.; exit; endif
               enddo
              endif
  !Contract tensors:
              if(transpose) then
               n=n+1 !tensor contraction number
               !if(n.ne.2933.and.n.ne.2934.and.n.ne.2935) cycle ploop !debug
   !Get the symbolic contraction pattern:
               call get_contr_pattern_sym(rl,rr,cptrn,cptrn_sym,l,ierr); if(ierr.ne.0) then; ierr=5; return; endif
               do i=1,l; str(i:i)=cptrn_sym(i); enddo
               write(*,'(2x,"Contraction ",i8,": (",16(1x,i8))',ADVANCE='NO') n,ddims(1:rd)
               write(*,'(") = (",16(1x,i8))',ADVANCE='NO') ldims(1:rl)
               write(*,'(") * (",16(1x,i8))',ADVANCE='NO') rdims(1:rr)
               call printl(6,'): '//str(1:l),adv=.FALSE.)
!              write(*,'(":",32(1x,i3))',ADVANCE='NO') cptrn(1:rl+rr) !contraction pattern
               repeat=DO_TWICE
   !Construct tensor blocks:
               cval=(1d-1,0d0); ierr=talsh_tensor_construct(dtens,TENS_DATA_KIND,ddims(1:rd),in_hab=YEP,init_val=cval)
               if(ierr.ne.TALSH_SUCCESS) then; ierr=6; return; endif
               cval=(1d-2,0d0); ierr=talsh_tensor_construct(ltens,TENS_DATA_KIND,ldims(1:rl),in_hab=YEP,init_val=cval)
               if(ierr.ne.TALSH_SUCCESS) then; ierr=7; return; endif
               cval=(1d-3,0d0); ierr=talsh_tensor_construct(rtens,TENS_DATA_KIND,rdims(1:rr),in_hab=YEP,init_val=cval)
               if(ierr.ne.TALSH_SUCCESS) then; ierr=8; return; endif
               dloop: do
#ifndef NO_GPU
   !Schedule tensor contraction on GPU:
                ierr=talsh_tensor_contract(str(1:l),dtens,ltens,rtens,&
                                          &dev_id=talsh_flat_dev_id(DEV_NVIDIA_GPU,0),copy_ctrl=COPY_TTT,talsh_task=tsk)
                if(ierr.ne.TALSH_SUCCESS) then; write(*,'("Error ",i11)') ierr; ierr=9; return; endif
   !Wait for GPU completion:
                ierr=talsh_task_wait(tsk,sts); if(ierr.ne.TALSH_SUCCESS.or.sts.ne.TALSH_TASK_COMPLETED) then; ierr=10; return; endif
                ierr=talsh_task_time(tsk,tm,tmc,tmi,tmo,tmm)
                if(ierr.ne.TALSH_SUCCESS) then; write(*,'("Error ",i11)') ierr; ierr=11; return; endif
                write(*,'(": ",D9.3)',ADVANCE='NO') flops/dble(words) !compute (arithmetic) intensity
                write(10,'(i8,4(1x,D9.3))',ADVANCE='NO') n,dble(vd),dble(vl),dble(vr),flops/dble(words)
                if(tmm.gt.0d0) then
                 write(*,'(1x,D9.3)',ADVANCE='NO') flops/tmm !GPU matrix multiplication Flop/s
                 write(10,'(1x,D9.3)',ADVANCE='NO') flops/tmm !GPU matrix multiplication Flop/s
                else
                 write(*,'(" ???")',ADVANCE='NO')
                 write(10,'(" ???")',ADVANCE='NO')
                endif
                if(tmc.gt.0d0) then
                 write(*,'(1x,D9.3)',ADVANCE='NO') flops/tmc !GPU tensor contraction Flop/s
                 write(10,'(1x,D9.3)',ADVANCE='NO') flops/tmc !GPU tensor contraction Flop/s
                else
                 write(*,'(" ???")',ADVANCE='NO')
                 write(10,'(" ???")',ADVANCE='NO')
                endif
                if(tm.gt.0d0) then
                 write(*,'(1x,D9.3)',ADVANCE='NO') flops/tm !Effective tensor contraction performance with data transfers, Flop/s
                 write(10,'(1x,D9.3)',ADVANCE='NO') flops/tm !Effective tensor contraction performance with data transfers, Flop/s
                else
                 write(*,'(" ???")',ADVANCE='NO')
                 write(10,'(" ???")',ADVANCE='NO')
                endif
                !write(*,'()')
                !write(10,'()')
   !Destruct task handle:
                ierr=talsh_task_destruct(tsk); if(ierr.ne.TALSH_SUCCESS) then; ierr=12; return; endif
   !Compute the destination tensor norm:
                gn1=talshTensorImageNorm1_cpu(dtens)!; write(*,'(1x,"Destination Norm1 (GPU) = ",D25.14)') gn1
   !Destruct/construct the destination tensor:
                ierr=talsh_tensor_destruct(dtens); if(ierr.ne.TALSH_SUCCESS) then; ierr=13; return; endif
                cval=(1d-1,0d0); ierr=talsh_tensor_construct(dtens,TENS_DATA_KIND,ddims(1:rd),init_val=cval)
                if(ierr.ne.TALSH_SUCCESS) then; ierr=14; return; endif
#endif
   !Run tensor contraction on CPU:
                !write(*,'("#DEBUG: Tensor contraction on CPU ...")') !debug
                ierr=talsh_tensor_contract(str(1:l),dtens,ltens,rtens,dev_id=talsh_flat_dev_id(DEV_HOST,0),talsh_task=tsk)
                if(ierr.ne.TALSH_SUCCESS) then; write(*,'("Error ",i11)') ierr; ierr=15; return; endif
                ierr=talsh_task_wait(tsk,sts); if(ierr.ne.TALSH_SUCCESS.or.sts.ne.TALSH_TASK_COMPLETED) then; ierr=16; return; endif
                ierr=talsh_task_time(tsk,tm,tmc,tmi,tmo,tmm)
                if(ierr.ne.TALSH_SUCCESS) then; write(*,'("Error ",i11)') ierr; ierr=17; return; endif
                if(tm.gt.0d0) then
                 write(*,'(1x,D9.3)') flops/tm !CPU tensor contraction Flop/s
                 write(10,'(1x,D9.3)') flops/tm !CPU tensor contraction Flop/s
                else
                 write(*,'(" ???")')
                 write(10,'(" ???")')
                endif
                cn1=talshTensorImageNorm1_cpu(dtens)!; write(*,'(1x,"Destination Norm1 (CPU) = ",D25.14)') cn1
#ifndef NO_GPU
                if(dabs(cn1-gn1)/max(cn1,gn1).gt.CMP_ZERO) then
                 write(*,'("FAILED: CPU/GPU result mismatch: 1-norms (CPU vs GPU): ",D25.14,2x,D25.14)') cn1,gn1
                 ierr=18; return
                !else
                 !write(*,'("#DEBUG(TALSH:main): Norms: ",D25.14,1x,D25.14)') cn1,gn1 !debug
                endif
#endif
                ierr=talsh_task_destruct(tsk); if(ierr.ne.TALSH_SUCCESS) then; ierr=19; return; endif
                if(repeat) then; repeat=.FALSE.; else; exit dloop; endif
               enddo dloop
   !Destruct tensor blocks:
               ierr=talsh_tensor_destruct(dtens); if(ierr.ne.TALSH_SUCCESS) then; ierr=20; return; endif
               ierr=talsh_tensor_destruct(ltens); if(ierr.ne.TALSH_SUCCESS) then; ierr=21; return; endif
               ierr=talsh_tensor_destruct(rtens); if(ierr.ne.TALSH_SUCCESS) then; ierr=22; return; endif
              endif
             enddo ploop !m
            enddo !ncl
           enddo !nc
          enddo !rl
         enddo !rr
         write(*,'(i7," tensor contractions processed.")') n

!Print run-time statistics:
         ierr=talsh_stats()
         if(ierr.ne.TALSH_SUCCESS) then; ierr=23; return; endif
!Shutdown TALSH:
         write(*,'(1x,"Shutting down TALSH ... ")',ADVANCE='NO')
         ierr=talsh_shutdown()
         write(*,'("Status ",i11)') ierr
         if(ierr.ne.TALSH_SUCCESS) then; ierr=24; return; endif
         close(10)
         return
        end subroutine benchmark_tensor_contractions_rnd
!---------------------------------------------------------
        subroutine benchmark_tensor_contractions_ctm(ierr)
!Benchmarks tensor contraction performance (custom tensor contractions).
         use, intrinsic:: ISO_C_BINDING
         use tensor_algebra
         use talsh
         use stsubs
         use combinatoric
         implicit none
         integer(C_INT), intent(out):: ierr
         integer(C_SIZE_T), parameter:: BUF_SIZE=1_8*1024_8*1024_8*1024_8 !desired Host argument buffer size in bytes
         integer(C_INT), parameter:: EXEC_DEVICE=DEV_HOST !device on which tensor contractions will be executed
         integer(C_INT), parameter:: TENS_DATA_KIND=R8 !tensor data kind
         real(C_DOUBLE), parameter:: CMP_ZERO=1d-4 !comparison threshold (relative)
         !----------------------------------------
         integer(C_INT):: i,sl,rd,rl,rr,num_gpus,host_arg_max,dev,sts
         integer(C_INT):: ddims(1:MAX_TENSOR_RANK),ldims(1:MAX_TENSOR_RANK),rdims(1:MAX_TENSOR_RANK)
         integer(C_SIZE_T):: host_buf_size,vd,vl,vr
         real(C_DOUBLE):: flops,words,tm,tmc,tmi,tmo,tmm,dn1
         complex(8):: cval
         character(512):: str
         type(talsh_tens_t):: dtens,ltens,rtens
         type(talsh_task_t):: tsk

         ierr=0
#ifndef NO_GPU
         write(*,'(1x,"Checking Nvidia GPU availability ... ")',ADVANCE='NO')
         ierr=cuda_get_device_count(num_gpus)
         write(*,'("Status ",i11,": Number of GPUs = ",i3)') ierr,num_gpus
         if(ierr.ne.TALSH_SUCCESS) then; write(*,'("Error ",i11)') ierr; ierr=1; return; endif
#else
         num_gpus=0
#endif
!Initialize TALSH runtime:
         write(*,'(1x,"Initializing TALSH ... ")',ADVANCE='NO')
         host_buf_size=BUF_SIZE
#ifndef NO_GPU
         ierr=talsh_init(host_buf_size,host_arg_max,gpu_list=(/(i,i=0,num_gpus-1)/))
         dev=EXEC_DEVICE
#else
         ierr=talsh_init(host_buf_size,host_arg_max)
         dev=DEV_HOST
#endif
         write(*,'("Status ",i11,": Size (Bytes) = ",i13,": Max args in HAB = ",i7)') ierr,host_buf_size,host_arg_max
         if(ierr.ne.TALSH_SUCCESS) then; write(*,'("Error ",i11)') ierr; ierr=2; return; endif

!Open the file containing tensor contractions:
         open(10,file='tensor_contractions.txt',form='FORMATTED',status='UNKNOWN')
!Execute tensor contractions:
         do
 !Read the tensor contraction specification:
          str=' '; read(10,'(A512)',end=100) str; sl=len_trim(str)
          read(10,*,end=100) rd,ddims(1:rd),rl,ldims(1:rl),rr,rdims(1:rr)
          call printl(6,' '//str(1:sl))
          write(*,'(2x,"Destination shape:",32(1x,i5))') ddims(1:rd)
          write(*,'(2x,"Left shape       :",32(1x,i5))') ldims(1:rl)
          write(*,'(2x,"Right shape      :",32(1x,i5))') rdims(1:rr)
 !Construct tensor blocks:
          cval=(0d0,0d0); ierr=talsh_tensor_construct(dtens,TENS_DATA_KIND,ddims(1:rd),in_hab=YEP,init_val=cval)
          if(ierr.ne.TALSH_SUCCESS) then; write(*,'("Error ",i11)') ierr; ierr=3; return; endif
          cval=(1d-2,0d0); ierr=talsh_tensor_construct(ltens,TENS_DATA_KIND,ldims(1:rl),in_hab=YEP,init_val=cval)
          if(ierr.ne.TALSH_SUCCESS) then; write(*,'("Error ",i11)') ierr; ierr=4; return; endif
          cval=(1d-3,0d0); ierr=talsh_tensor_construct(rtens,TENS_DATA_KIND,rdims(1:rr),in_hab=YEP,init_val=cval)
          if(ierr.ne.TALSH_SUCCESS) then; write(*,'("Error ",i11)') ierr; ierr=5; return; endif
          vd=talsh_tensor_volume(dtens)
          vl=talsh_tensor_volume(ltens)
          vr=talsh_tensor_volume(rtens)
          words=dble(vd)+dble(vl)+dble(vr)            !total data size in words
          flops=dsqrt(dble(vd)*dble(vl)*dble(vr))*2d0 !number of floating point operations

 !Schedule the tensor contraction:
          ierr=talsh_tensor_contract(str(1:sl),dtens,ltens,rtens,dev_id=dev,copy_ctrl=COPY_TTT,talsh_task=tsk)
          if(ierr.eq.TALSH_SUCCESS) then
  !Wait for completion:
           ierr=talsh_task_wait(tsk,sts)
           if(ierr.ne.TALSH_SUCCESS.or.sts.ne.TALSH_TASK_COMPLETED) then; ierr=7; return; endif
           ierr=talsh_task_time(tsk,tm,tmc,tmi,tmo,tmm)
           if(ierr.ne.TALSH_SUCCESS) then; write(*,'("Error ",i11)') ierr; ierr=8; return; endif
           write(*,'(3x,"Total GFlops = ",F12.4,": Compute intensity = ",F12.4)')&
                &flops/dble(1024*1024*1024),flops/words
           if(dev.eq.DEV_HOST) then
            write(*,'(3x,"Timings (total):",F8.4,": GFlop/s = ",F12.4)')&
                 &tm,flops/tm/dble(1024*1024*1024)
           else
            write(*,'(3x,"Timings (total,compute,mm):",3(F8.4),": GFlop/s = ",F12.4,": Overhead = ",F8.2,"%")')&
                 &tm,tmc,tmm,flops/tmc/dble(1024*1024*1024),max(tmc/tmm-1d0,0d0)*1d2
           endif
  !Compute the destination tensor norm:
           dn1=talshTensorImageNorm1_cpu(dtens); write(*,'(3x,"Destination Norm1 = ",D25.14)') dn1
          else
           write(*,'("Error ",i11)') ierr
           if(ierr.ne.DEVICE_UNABLE) then; ierr=6; return; endif
          endif
 !Destruct task handle:
          ierr=talsh_task_destruct(tsk)
          if(ierr.ne.TALSH_SUCCESS) then; write(*,'("Error ",i11)') ierr; ierr=9; return; endif
 !Destruct tensor blocks:
          ierr=talsh_tensor_destruct(rtens)
          if(ierr.ne.TALSH_SUCCESS) then; write(*,'("Error ",i11)') ierr; ierr=10; return; endif
          ierr=talsh_tensor_destruct(ltens)
          if(ierr.ne.TALSH_SUCCESS) then; write(*,'("Error ",i11)') ierr; ierr=11; return; endif
          ierr=talsh_tensor_destruct(dtens)
          if(ierr.ne.TALSH_SUCCESS) then; write(*,'("Error ",i11)') ierr; ierr=12; return; endif
         enddo !next tensor contraction
100      close(10)

!Print run-time statistics:
         ierr=talsh_stats()
         if(ierr.ne.TALSH_SUCCESS) then; write(*,'("Error ",i11)') ierr; ierr=13; return; endif
!Shutdown TALSH:
         write(*,'(1x,"Shutting down TALSH ... ")',ADVANCE='NO')
         ierr=talsh_shutdown()
         write(*,'("Status ",i11)') ierr
         if(ierr.ne.TALSH_SUCCESS) then; write(*,'("Error ",i11)') ierr; ierr=14; return; endif
         return
        end subroutine benchmark_tensor_contractions_ctm
