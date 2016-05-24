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
#ifndef NO_GPU
        interface
         subroutine test_nvtal_c(ierr) bind(c)
          import
          integer(C_INT), intent(out):: ierr
         end subroutine test_nvtal_c
        end interface
#endif
        integer(C_INT):: ierr

!Test NV-TAL C/C++ API interface:
#ifndef NO_GPU
        write(*,'("Testing NV-TAL C/C++ API ...")')
        call test_nvtal_c(ierr)
        write(*,'("Done: Status ",i5)') ierr
        if(ierr.ne.0) stop
        write(*,*)''
#endif
!Test TAL-SH Fortran API interface:
        write(*,'("Testing TAL-SH Fortran API ...")')
        call test_talsh_f(ierr)
        write(*,'("Done: Status ",i5)') ierr
        if(ierr.ne.0) stop
        stop
        end program main
!------------------------------------
        subroutine test_talsh_f(ierr)
!Testing device-unified TAL-SH Fortran API.
        use, intrinsic:: ISO_C_BINDING
        use tensor_algebra
        use talsh
        implicit none
        integer(C_SIZE_T), parameter:: BUF_SIZE=1024*1024*1024 !desired Host argument buffer size in bytes
        integer(C_INT), parameter:: DIM_EXT=41 !tensor dimension extent
        integer(C_SIZE_T):: host_buf_size
        integer(C_INT):: i,n,ierr,host_arg_max,dev_gpu,dev_cpu,sts(3)
        type(talsh_tens_t):: tens(9) !three tensors for CPU, six for GPU
        type(talsh_task_t):: tsks(3) !three tasks (tensor contractions, three tensors per tensor contraction)

        interface
         real(C_DOUBLE) function talshTensorImageNorm1_cpu(talsh_tens) bind(c,name='talshTensorImageNorm1_cpu')
          import
          implicit none
          type(talsh_tens_t), intent(in):: talsh_tens
         end function talshTensorImageNorm1_cpu
        end interface

        ierr=0
!Init TALSH:
        write(*,'(1x,"Initializing TALSH ... ")',ADVANCE='NO')
        host_buf_size=BUF_SIZE
#ifndef NO_GPU
        ierr=talsh_init(host_buf_size,host_arg_max,gpu_list=(/0/))
#else
        ierr=talsh_init(host_buf_size,host_arg_max)
#endif
        write(*,'("Status ",i11,": Size (Bytes) = ",i13,": Max args in HAB = ",i7)') ierr,host_buf_size,host_arg_max
        if(ierr.ne.TALSH_SUCCESS) then; ierr=1; return; endif

!Create nine rank-4 tensors on Host and initialize them to value:
 !Tensor block 1:
        write(*,'(1x,"Constructing tensor block 1 ... ")',ADVANCE='NO')
        ierr=talsh_tensor_construct(tens(1),R8,(/DIM_EXT,DIM_EXT,DIM_EXT,DIM_EXT/),init_val=(0d0,0d0))
        write(*,'("Status ",i11)') ierr; if(ierr.ne.TALSH_SUCCESS) then; ierr=1; return; endif
 !Tensor block 2:
        write(*,'(1x,"Constructing tensor block 2 ... ")',ADVANCE='NO')
        ierr=talsh_tensor_construct(tens(2),R8,(/DIM_EXT,DIM_EXT,DIM_EXT,DIM_EXT/),init_val=(1d-2,0d0))
        write(*,'("Status ",i11)') ierr; if(ierr.ne.TALSH_SUCCESS) then; ierr=1; return; endif
 !Tensor block 3:
        write(*,'(1x,"Constructing tensor block 3 ... ")',ADVANCE='NO')
        ierr=talsh_tensor_construct(tens(3),R8,(/DIM_EXT,DIM_EXT,DIM_EXT,DIM_EXT/),init_val=(1d-3,0d0))
        write(*,'("Status ",i11)') ierr; if(ierr.ne.TALSH_SUCCESS) then; ierr=1; return; endif
 !Tensor block 4:
        write(*,'(1x,"Constructing tensor block 4 ... ")',ADVANCE='NO')
        ierr=talsh_tensor_construct(tens(4),R8,(/DIM_EXT,DIM_EXT,DIM_EXT,DIM_EXT/),init_val=(0d0,0d0))
        write(*,'("Status ",i11)') ierr; if(ierr.ne.TALSH_SUCCESS) then; ierr=1; return; endif
 !Tensor block 5:
        write(*,'(1x,"Constructing tensor block 5 ... ")',ADVANCE='NO')
        ierr=talsh_tensor_construct(tens(5),R8,(/DIM_EXT,DIM_EXT,DIM_EXT,DIM_EXT/),init_val=(1d-2,0d0))
        write(*,'("Status ",i11)') ierr; if(ierr.ne.TALSH_SUCCESS) then; ierr=1; return; endif
 !Tensor block 6:
        write(*,'(1x,"Constructing tensor block 6 ... ")',ADVANCE='NO')
        ierr=talsh_tensor_construct(tens(6),R8,(/DIM_EXT,DIM_EXT,DIM_EXT,DIM_EXT/),init_val=(1d-3,0d0))
        write(*,'("Status ",i11)') ierr; if(ierr.ne.TALSH_SUCCESS) then; ierr=1; return; endif
 !Tensor block 7:
        write(*,'(1x,"Constructing tensor block 7 ... ")',ADVANCE='NO')
        ierr=talsh_tensor_construct(tens(7),R8,(/DIM_EXT,DIM_EXT,DIM_EXT,DIM_EXT/),init_val=(0d0,0d0))
        write(*,'("Status ",i11)') ierr; if(ierr.ne.TALSH_SUCCESS) then; ierr=1; return; endif
 !Tensor block 8:
        write(*,'(1x,"Constructing tensor block 8 ... ")',ADVANCE='NO')
        ierr=talsh_tensor_construct(tens(8),R8,(/DIM_EXT,DIM_EXT,DIM_EXT,DIM_EXT/),init_val=(1d-2,0d0))
        write(*,'("Status ",i11)') ierr; if(ierr.ne.TALSH_SUCCESS) then; ierr=1; return; endif
 !Tensor block 9:
        write(*,'(1x,"Constructing tensor block 9 ... ")',ADVANCE='NO')
        ierr=talsh_tensor_construct(tens(9),R8,(/DIM_EXT,DIM_EXT,DIM_EXT,DIM_EXT/),init_val=(1d-3,0d0))
        write(*,'("Status ",i11)') ierr; if(ierr.ne.TALSH_SUCCESS) then; ierr=1; return; endif

!Choose execution devices:
        dev_cpu=talsh_flat_dev_id(DEV_HOST,0) !Host CPU
#ifndef NO_GPU
        dev_gpu=talsh_flat_dev_id(DEV_NVIDIA_GPU,0) !Nvidia GPU #0
#else
        dev_gpu=dev_cpu !fall back to CPU when no GPU in use
#endif

!Data transfers:
 !Copy tensor block 2 to GPU:
        write(*,'(1x,"Cloning tensor block 2 to GPU ... ")',ADVANCE='NO')
        ierr=talsh_tensor_place(tens(2),dev_gpu,copy_ctrl=COPY_K,talsh_task=tsks(1))
        write(*,'("Status ",i11)') ierr; if(ierr.ne.TALSH_SUCCESS) then; ierr=1; return; endif
        write(*,'(1x,"Waiting upon completion of the data transfer ... ")',ADVANCE='NO')
        ierr=talsh_task_wait(tsks(1),sts(1))
        write(*,'("Status ",i11," Completion = ",i11)') ierr,sts(1); if(ierr.ne.TALSH_SUCCESS) then; ierr=1; return; endif
 !Print tensor info:
        call talsh_tensor_print_info(tens(2))
 !Destruct (clean) TAL-SH task:
        write(*,'(1x,"Destructing TAL-SH task 1 ... ")',ADVANCE='NO')
        ierr=talsh_task_destruct(tsks(1))
        write(*,'("Status ",i11)') ierr; if(ierr.ne.TALSH_SUCCESS) then; ierr=1; return; endif
#ifndef NO_GPU
 !Discard CPU copy:
        write(*,'(1x,"Discarding tensor block 2 from CPU ... ")',ADVANCE='NO')
        ierr=talsh_tensor_discard(tens(2),0,DEV_HOST)
        write(*,'("Status ",i11)') ierr; if(ierr.ne.TALSH_SUCCESS) then; ierr=1; return; endif
 !Print tensor info:
        !call talsh_tensor_print_info(tens(2))
 !Move the GPU copy back to CPU:
        write(*,'(1x,"Moving tensor block 2 from GPU to CPU ... ")',ADVANCE='NO')
        ierr=talsh_tensor_place(tens(2),dev_cpu,copy_ctrl=COPY_M,talsh_task=tsks(1))
        write(*,'("Status ",i11)') ierr; if(ierr.ne.TALSH_SUCCESS) then; ierr=1; return; endif
        write(*,'(1x,"Waiting upon completion of the data transfer ... ")',ADVANCE='NO')
        ierr=talsh_task_wait(tsks(1),sts(1))
        write(*,'("Status ",i11," Completion = ",i11)') ierr,sts(1); if(ierr.ne.TALSH_SUCCESS) then; ierr=1; return; endif
 !Print tensor info:
        call talsh_tensor_print_info(tens(2))
 !Destruct (clean) TAL-SH task:
        write(*,'(1x,"Destructing TAL-SH task 1 ... ")',ADVANCE='NO')
        ierr=talsh_task_destruct(tsks(1))
        write(*,'("Status ",i11)') ierr; if(ierr.ne.TALSH_SUCCESS) then; ierr=1; return; endif
#endif

        n=0 !number of tasks scheduled
!Schedule two tensor contractions on GPU:
        write(*,'(1x,"Scheduling tensor contraction 1 on GPU ... ")',ADVANCE='NO')
        n=n+1
        ierr=talsh_tensor_contract('D(a,b,i,j)+=L(j,c,k,a)*R(c,b,k,i)',tens(1),tens(2),tens(3),&
                                   &copy_ctrl=COPY_TTT,dev_id=dev_gpu,talsh_task=tsks(n))
        write(*,'("Status ",i11)') ierr; if(ierr.ne.TALSH_SUCCESS) then; ierr=1; return; endif
        !call talsh_task_print_info(tsks(n)) !debug
        write(*,'(1x,"Scheduling tensor contraction 2 on GPU ... ")',ADVANCE='NO')
        n=n+1
        ierr=talsh_tensor_contract('D(a,b,i,j)+=L(j,c,k,a)*R(c,b,k,i)',tens(4),tens(5),tens(6),&
                                   &copy_ctrl=COPY_TTT,dev_id=dev_gpu,talsh_task=tsks(n))
        write(*,'("Status ",i11)') ierr; if(ierr.ne.TALSH_SUCCESS) then; ierr=1; return; endif
        !call talsh_task_print_info(tsks(n)) !debug
!Execute a tensor contraction on CPU (while the previous two are running on GPU):
        write(*,'(1x,"Executing tensor contraction 3 on CPU ... ")',ADVANCE='NO')
        n=n+1
        ierr=talsh_tensor_contract('D(a,b,i,j)+=L(j,c,k,a)*R(c,b,k,i)',tens(7),tens(8),tens(9),dev_id=dev_cpu,talsh_task=tsks(n))
        write(*,'("Status ",i11)') ierr; if(ierr.ne.TALSH_SUCCESS) then; ierr=1; return; endif
        !call talsh_task_print_info(tsks(n)) !debug
!Synchronize and compare the results:
        write(*,'(1x,"Waiting upon completion of tensor contractions on GPU ... ")',ADVANCE='NO')
        ierr=talsh_tasks_wait(n,tsks,sts)
        write(*,'("Status ",i11," Completion = ",8(i8))') ierr,sts(1:n); if(ierr.ne.TALSH_SUCCESS) then; ierr=1; return; endif
!Printing results:
        call talsh_tensor_print_info(tens(1)); print *,'TENSOR 1 NORM1 = ',talshTensorImageNorm1_cpu(tens(1))
        call talsh_tensor_print_info(tens(4)); print *,'TENSOR 4 NORM1 = ',talshTensorImageNorm1_cpu(tens(4))
        call talsh_tensor_print_info(tens(7)); print *,'TENSOR 7 NORM1 = ',talshTensorImageNorm1_cpu(tens(7))

!Destruct TAL-SH task handles:
        do i=3,1,-1
         write(*,'(1x,"Destructing task handle ",i2," ... ")',ADVANCE='NO') i
         ierr=talsh_task_destruct(tsks(i))
         write(*,'("Status ",i11)') ierr; if(ierr.ne.TALSH_SUCCESS) then; ierr=1; return; endif
        enddo

!Destroy tensors:
        do i=9,1,-1
         write(*,'(1x,"Destructing tensor block ",i2," ... ")',ADVANCE='NO') i
         ierr=talsh_tensor_destruct(tens(i))
         write(*,'("Status ",i11)') ierr; if(ierr.ne.TALSH_SUCCESS) then; ierr=1; return; endif
        enddo
!Print run-time statistics:
        ierr=talsh_stats()
        if(ierr.ne.TALSH_SUCCESS) then; ierr=1; return; endif
!Shutdown TALSH:
        write(*,'(1x,"Shutting down TALSH ... ")',ADVANCE='NO')
        ierr=talsh_shutdown()
        write(*,'("Status ",i11)') ierr
        if(ierr.ne.TALSH_SUCCESS) then; ierr=1; return; endif
        return
        end subroutine test_talsh_f
