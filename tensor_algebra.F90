!ExaTensor::TAL-SH: Parameters, types, C function interfaces:
!Keep consistent with "tensor_algebra.h"!
!REVISION: 2017/01/24

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

        module tensor_algebra
        use dil_basic !contains ISO_C_BINDING: basic parameters
        implicit none
        public
!TENSOR ALGEBRA LIMITS (keep consistent with tensor_algebra.h):
        integer(C_INT), parameter, public:: MAX_TENSOR_RANK=32    !max allowed tensor rank (max number of indices in a tensor)
        integer(C_INT), parameter, public:: MAX_TENSOR_OPERANDS=4 !max number of tensor operands in a tensor operation
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: MAX_TENSOR_RANK,MAX_TENSOR_OPERANDS
!DIR$ ATTRIBUTES ALIGN:128:: MAX_TENSOR_RANK,MAX_TENSOR_OPERANDS
#endif

!MEMORY ALLOCATION POLICY (keep consistent with tensor_algebra.h):
        integer(C_INT), parameter, public:: MEM_ALLOC_REGULAR=0 !all large memory allocations are done via the regular Fortran allocator
        integer(C_INT), parameter, public:: MEM_ALLOC_TMP_BUF=1 !large temporary memory allocations are done via the Host argument buffer
        integer(C_INT), parameter, public:: MEM_ALLOC_ALL_BUF=2 !all large memory allocations are done via the Host argument buffer
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: MEM_ALLOC_REGULAR,MEM_ALLOC_TMP_BUF,MEM_ALLOC_ALL_BUF
!DIR$ ATTRIBUTES ALIGN:128:: MEM_ALLOC_REGULAR,MEM_ALLOC_TMP_BUF,MEM_ALLOC_ALL_BUF
#endif

!ALIASES (keep consistent with tensor_algebra.h):
        integer(C_INT), parameter, public:: BLAS_ON=0                   !enables BLAS
        integer(C_INT), parameter, public:: BLAS_OFF=1                  !disables BLAS
        integer(C_INT), parameter, public:: EFF_TRN_OFF=0               !disables efficient tensor transpose algorithm
        integer(C_INT), parameter, public:: EFF_TRN_ON=1                !enables efficient tensor transpose algorithm
        integer(C_INT), parameter, public:: DEVICE_UNABLE=-546372819    !device is unsuitable for the given task: KEEP THIS UNIQUE!

        integer(C_INT), parameter, public:: EVERYTHING=0                !everything (source, destination, temporary)
        integer(C_INT), parameter, public:: SOURCE=1                    !source
        integer(C_INT), parameter, public:: DESTINATION=2               !destination
        integer(C_INT), parameter, public:: TEMPORARY=3                 !temporary
        integer(C_INT), parameter, public:: DEV_OFF=0                   !device status "Disabled"
        integer(C_INT), parameter, public:: DEV_ON=1                    !device status "Enabled"
        integer(C_INT), parameter, public:: DEV_ON_BLAS=2               !device status "Enabled with vendor provided BLAS"
        integer(C_INT), parameter, public:: GPU_OFF=0                   !GPU status "Disabled"
        integer(C_INT), parameter, public:: GPU_ON=1                    !GPU status "Enabled"
        integer(C_INT), parameter, public:: GPU_ON_BLAS=2               !GPU status "Enabled with vendor provided BLAS"
        integer(C_INT), parameter, public:: NO_COPY_BACK=0              !keeps the tensor-result on Accelerator without updating Host
        integer(C_INT), parameter, public:: COPY_BACK=1                 !tensor-result will be copied back from Accelerator to Host (default)
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: BLAS_ON,BLAS_OFF,EFF_TRN_OFF,EFF_TRN_ON,DEVICE_UNABLE,NO_COPY_BACK,COPY_BACK
!DIR$ ATTRIBUTES OFFLOAD:mic:: EVERYTHING,SOURCE,DESTINATION,TEMPORARY,DEV_OFF,DEV_ON,DEV_ON_BLAS
!DIR$ ATTRIBUTES ALIGN:128:: BLAS_ON,BLAS_OFF,EFF_TRN_OFF,EFF_TRN_ON,DEVICE_UNABLE,NO_COPY_BACK,COPY_BACK
!DIR$ ATTRIBUTES ALIGN:128:: EVERYTHING,SOURCE,DESTINATION,TEMPORARY,DEV_OFF,DEV_ON,DEV_ON_BLAS
#endif
!Coherence (copy) control parameters (Senior bits: Destination -> Left -> Right: Junior bits):
        integer(C_INT), parameter, public:: COPY_D=0
        integer(C_INT), parameter, public:: COPY_M=1
        integer(C_INT), parameter, public:: COPY_T=2
        integer(C_INT), parameter, public:: COPY_K=3
        integer(C_INT), parameter, public:: COPY_DD=0
        integer(C_INT), parameter, public:: COPY_DM=1
        integer(C_INT), parameter, public:: COPY_DT=2
        integer(C_INT), parameter, public:: COPY_DK=3
        integer(C_INT), parameter, public:: COPY_MD=4
        integer(C_INT), parameter, public:: COPY_MM=5
        integer(C_INT), parameter, public:: COPY_MT=6
        integer(C_INT), parameter, public:: COPY_MK=7
        integer(C_INT), parameter, public:: COPY_TD=8
        integer(C_INT), parameter, public:: COPY_TM=9
        integer(C_INT), parameter, public:: COPY_TT=10
        integer(C_INT), parameter, public:: COPY_TK=11
        integer(C_INT), parameter, public:: COPY_KD=12
        integer(C_INT), parameter, public:: COPY_KM=13
        integer(C_INT), parameter, public:: COPY_KT=14
        integer(C_INT), parameter, public:: COPY_KK=15
        integer(C_INT), parameter, public:: COPY_DDD=0
        integer(C_INT), parameter, public:: COPY_DDM=1
        integer(C_INT), parameter, public:: COPY_DDT=2
        integer(C_INT), parameter, public:: COPY_DDK=3
        integer(C_INT), parameter, public:: COPY_DMD=4
        integer(C_INT), parameter, public:: COPY_DMM=5
        integer(C_INT), parameter, public:: COPY_DMT=6
        integer(C_INT), parameter, public:: COPY_DMK=7
        integer(C_INT), parameter, public:: COPY_DTD=8
        integer(C_INT), parameter, public:: COPY_DTM=9
        integer(C_INT), parameter, public:: COPY_DTT=10
        integer(C_INT), parameter, public:: COPY_DTK=11
        integer(C_INT), parameter, public:: COPY_DKD=12
        integer(C_INT), parameter, public:: COPY_DKM=13
        integer(C_INT), parameter, public:: COPY_DKT=14
        integer(C_INT), parameter, public:: COPY_DKK=15
        integer(C_INT), parameter, public:: COPY_MDD=16
        integer(C_INT), parameter, public:: COPY_MDM=17
        integer(C_INT), parameter, public:: COPY_MDT=18
        integer(C_INT), parameter, public:: COPY_MDK=19
        integer(C_INT), parameter, public:: COPY_MMD=20
        integer(C_INT), parameter, public:: COPY_MMM=21
        integer(C_INT), parameter, public:: COPY_MMT=22
        integer(C_INT), parameter, public:: COPY_MMK=23
        integer(C_INT), parameter, public:: COPY_MTD=24
        integer(C_INT), parameter, public:: COPY_MTM=25
        integer(C_INT), parameter, public:: COPY_MTT=26
        integer(C_INT), parameter, public:: COPY_MTK=27
        integer(C_INT), parameter, public:: COPY_MKD=28
        integer(C_INT), parameter, public:: COPY_MKM=29
        integer(C_INT), parameter, public:: COPY_MKT=30
        integer(C_INT), parameter, public:: COPY_MKK=31
        integer(C_INT), parameter, public:: COPY_TDD=32
        integer(C_INT), parameter, public:: COPY_TDM=33
        integer(C_INT), parameter, public:: COPY_TDT=34
        integer(C_INT), parameter, public:: COPY_TDK=35
        integer(C_INT), parameter, public:: COPY_TMD=36
        integer(C_INT), parameter, public:: COPY_TMM=37
        integer(C_INT), parameter, public:: COPY_TMT=38
        integer(C_INT), parameter, public:: COPY_TMK=39
        integer(C_INT), parameter, public:: COPY_TTD=40
        integer(C_INT), parameter, public:: COPY_TTM=41
        integer(C_INT), parameter, public:: COPY_TTT=42
        integer(C_INT), parameter, public:: COPY_TTK=43
        integer(C_INT), parameter, public:: COPY_TKD=44
        integer(C_INT), parameter, public:: COPY_TKM=45
        integer(C_INT), parameter, public:: COPY_TKT=46
        integer(C_INT), parameter, public:: COPY_TKK=47
        integer(C_INT), parameter, public:: COPY_KDD=48
        integer(C_INT), parameter, public:: COPY_KDM=49
        integer(C_INT), parameter, public:: COPY_KDT=50
        integer(C_INT), parameter, public:: COPY_KDK=51
        integer(C_INT), parameter, public:: COPY_KMD=52
        integer(C_INT), parameter, public:: COPY_KMM=53
        integer(C_INT), parameter, public:: COPY_KMT=54
        integer(C_INT), parameter, public:: COPY_KMK=55
        integer(C_INT), parameter, public:: COPY_KTD=56
        integer(C_INT), parameter, public:: COPY_KTM=57
        integer(C_INT), parameter, public:: COPY_KTT=58
        integer(C_INT), parameter, public:: COPY_KTK=59
        integer(C_INT), parameter, public:: COPY_KKD=60
        integer(C_INT), parameter, public:: COPY_KKM=61
        integer(C_INT), parameter, public:: COPY_KKT=62
        integer(C_INT), parameter, public:: COPY_KKK=63

!CUDA TASK STATUS (keep consistent with tensor_algebra.h):
        integer(C_INT), parameter, public:: CUDA_TASK_ERROR=-1
        integer(C_INT), parameter, public:: CUDA_TASK_EMPTY=0
        integer(C_INT), parameter, public:: CUDA_TASK_SCHEDULED=1
        integer(C_INT), parameter, public:: CUDA_TASK_STARTED=2
        integer(C_INT), parameter, public:: CUDA_TASK_INPUT_THERE=3
        integer(C_INT), parameter, public:: CUDA_TASK_OUTPUT_THERE=4
        integer(C_INT), parameter, public:: CUDA_TASK_COMPLETED=5

!TENSOR BLOCK STORAGE LAYOUT:
        integer(C_INT), parameter, public:: NOT_ALLOCATED=0   !tensor block has not been allocated/initialized
        integer(C_INT), parameter, public:: SCALAR_TENSOR=1   !scalar (rank-0 tensor)
        integer(C_INT), parameter, public:: DIMENSION_LED=2   !dense tensor block (column-major storage by default): no symmetry restrictions
        integer(C_INT), parameter, public:: BRICKED_DENSE=3   !dense tensor block (bricked storage): no symmetry restrictions
        integer(C_INT), parameter, public:: BRICKED_ORDERED=4 !symmetrically packed tensor block (bricked storage): symmetry restrictions apply
        integer(C_INT), parameter, public:: SPARSE_LIST=5     !sparse tensor block: symmetry restrictions do not apply!
        integer(C_INT), parameter, public:: COMPRESSED=6      !compressed tensor block: symmetry restrictions do not apply!
        logical, parameter, public:: FORTRAN_LIKE=.true.
        logical, parameter, public:: C_LIKE=.false.
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: NOT_ALLOCATED,SCALAR_TENSOR,DIMENSION_LED,BRICKED_DENSE,BRICKED_ORDERED,SPARSE_LIST,COMPRESSED
!DIR$ ATTRIBUTES OFFLOAD:mic:: FORTRAN_LIKE,C_LIKE
!DIR$ ATTRIBUTES ALIGN:128:: NOT_ALLOCATED,SCALAR_TENSOR,DIMENSION_LED,BRICKED_DENSE,BRICKED_ORDERED,SPARSE_LIST,COMPRESSED
!DIR$ ATTRIBUTES ALIGN:128:: FORTRAN_LIKE,C_LIKE
#endif

!INTEROPERABLE TYPES (keep consistent with tensor_algebra.h):
 !TAL-SH tensor shape:
        type, public, bind(C):: talsh_tens_shape_t
         integer(C_INT):: num_dim=-1   !tensor rank (number of dimensions): >=0; -1:empty
         type(C_PTR):: dims=C_NULL_PTR !tensor dimension extents
         type(C_PTR):: divs=C_NULL_PTR !tensor dimension dividers
         type(C_PTR):: grps=C_NULL_PTR !tensor dimension groups
        end type talsh_tens_shape_t

!EXTERNAL INTERFACES (keep consistent with tensor_algebra.h):
 !User-defined tensor block initialization:
        abstract interface
         subroutine talsh_tens_init_i(tens_body_p,data_kind,tens_rank,tens_dims,ierr) bind(c)
          import
          type(C_PTR), value:: tens_body_p             !in: pointer to the tensor elements storage
          integer(C_INT), value:: data_kind            !in: data kind: {R4,R8,C4,C8}
          integer(C_INT), value:: tens_rank            !in: tensor block rank
          integer(C_INT), intent(in):: tens_dims(1:*)  !in: tensor block dimension extents
          integer(C_INT), intent(out):: ierr           !out: error code (0:success)
         end subroutine talsh_tens_init_i
        end interface
        public talsh_tens_init_i

!C FUNCTION INTERFACES (for Fortran):
        interface
 !Device management:
         integer(C_INT) function encode_device_id(dev_kind,dev_num) bind(c,name='encode_device_id')
          import
          implicit none
          integer(C_INT), intent(in), value:: dev_kind
          integer(C_INT), intent(in), value:: dev_num
         end function encode_device_id
 !Argument buffer memory management:
  !Initialize all argument buffers on Host and all Devices (GPU constant+global, MICs, etc):
         integer(C_INT) function arg_buf_allocate(host_mem,arg_max,gpu_beg,gpu_end) bind(c,name='arg_buf_allocate')
          import
          implicit none
          integer(C_SIZE_T), intent(inout):: host_mem
          integer(C_INT), intent(out):: arg_max
          integer(C_INT), value, intent(in):: gpu_beg
          integer(C_INT), value, intent(in):: gpu_end
         end function arg_buf_allocate
  !Deallocate all argument buffers on Host and all Devices:
         integer(C_INT) function arg_buf_deallocate(gpu_beg,gpu_end) bind(c,name='arg_buf_deallocate')
          import
          implicit none
          integer(C_INT), value, intent(in):: gpu_beg
          integer(C_INT), value, intent(in):: gpu_end
         end function arg_buf_deallocate
  !Check whether the Host argument buffer is clean:
         integer(C_INT) function arg_buf_clean_host() bind(c,name='arg_buf_clean_host')
          import
          implicit none
         end function arg_buf_clean_host
#ifndef NO_GPU
  !Check whether a GPU argument buffer is clean:
         integer(C_INT) function arg_buf_clean_gpu(gpu_num) bind(c,name='arg_buf_clean_gpu')
          import
          implicit none
          integer(C_INT), value, intent(in):: gpu_num
         end function arg_buf_clean_gpu
#endif
  !Get the buffer block sizes for each level of the Host argument buffer:
         integer(C_INT) function get_blck_buf_sizes_host(blck_sizes) bind(c,name='get_blck_buf_sizes_host')
          import
          implicit none
          integer(C_SIZE_T), intent(out):: blck_sizes(*)
         end function get_blck_buf_sizes_host
#ifndef NO_GPU
  !Get the buffer block sizes for each level of the GPU argument buffer:
         integer(C_INT) function get_blck_buf_sizes_gpu(gpu_num,blck_sizes) bind(c,name='get_blck_buf_sizes_gpu')
          import
          implicit none
          integer(C_INT), value, intent(in):: gpu_num
          integer(C_SIZE_T), intent(out):: blck_sizes(*)
         end function get_blck_buf_sizes_gpu
#endif
  !Get a free argument entry in the Host argument buffer:
         integer(C_INT) function get_buf_entry_host(bsize,entry_ptr,entry_num) bind(c,name='get_buf_entry_host')
          import
          implicit none
          integer(C_SIZE_T), value, intent(in):: bsize
          type(C_PTR), intent(out):: entry_ptr
          integer(C_INT), intent(out):: entry_num
         end function get_buf_entry_host
  !Free an argument entry in the Host argument buffer:
         integer(C_INT) function free_buf_entry_host(entry_num) bind(c,name='free_buf_entry_host')
          import
          implicit none
          integer(C_INT), value, intent(in):: entry_num
         end function free_buf_entry_host
#ifndef NO_GPU
  !Get a free argument entry in the GPU argument buffer:
         integer(C_INT) function get_buf_entry_gpu(gpu_num,bsize,entry_ptr,entry_num) bind(c,name='get_buf_entry_gpu')
          import
          implicit none
          integer(C_INT), value, intent(in):: gpu_num
          integer(C_SIZE_T), value, intent(in):: bsize
          type(C_PTR), intent(out):: entry_ptr
          integer(C_INT), intent(out):: entry_num
         end function get_buf_entry_gpu
  !Free an argument entry in the GPU argument buffer:
         integer(C_INT) function free_buf_entry_gpu(gpu_num,entry_num) bind(c,name='free_buf_entry_gpu')
          import
          implicit none
          integer(C_INT), value, intent(in):: gpu_num
          integer(C_INT), value, intent(in):: entry_num
         end function free_buf_entry_gpu
#endif
  !Get the argument buffer entry number for pointers located in the TAL-SH argument buffer:
         integer(C_INT) function get_buf_entry_from_address(dev_id,addr) bind(c,name='get_buf_entry_from_address')
          import
          implicit none
          integer(C_INT), value, intent(in):: dev_id
          type(C_PTR), value, intent(in):: addr
         end function get_buf_entry_from_address
  !Query the free buffer space in bytes on a given device:
         integer(C_INT) function mem_free_left(dev_id,free_mem) bind(c,name='mem_free_left')
          import
          implicit none
          integer(C_INT), value, intent(in):: dev_id
          integer(C_SIZE_T), intent(out):: free_mem
         end function mem_free_left
  !Print the current status of the argument buffer on a given device:
         integer(C_INT) function mem_print_stats(dev_id) bind(c,name='mem_print_stats')
          import
          implicit none
          integer(C_INT), value, intent(in):: dev_id
         end function mem_print_stats
  !Allocate pinned memory on Host:
         integer(C_INT) function host_mem_alloc_pin(cptr,bsize) bind(c,name='host_mem_alloc_pin')
          import
          implicit none
          type(C_PTR), intent(out):: cptr
          integer(C_SIZE_T), value, intent(in):: bsize !bytes
         end function host_mem_alloc_pin
  !Free pinned memory on Host:
         integer(C_INT) function host_mem_free_pin(cptr) bind(c,name='host_mem_free_pin')
          import
          implicit none
          type(C_PTR), value:: cptr
         end function host_mem_free_pin
  !Register (pin) Host memory:
         integer(C_INT) function host_mem_register(cptr,bsize) bind(c,name='host_mem_register')
          import
          implicit none
          type(C_PTR), value:: cptr
          integer(C_SIZE_T), value, intent(in):: bsize !bytes
         end function host_mem_register
  !Unregister (unpin) Host memory:
         integer(C_INT) function host_mem_unregister(cptr) bind(c,name='host_mem_unregister')
          import
          implicit none
          type(C_PTR), value:: cptr
         end function host_mem_unregister
#ifndef NO_GPU
 !NV-TAL debugging:
  !Get the current GPU error count:
         integer(C_INT) function gpu_get_error_count() bind(c,name='gpu_get_error_count')
          import
          implicit none
         end function gpu_get_error_count
  !Get the current GPU debug dump:
         integer(C_INT) function gpu_get_debug_dump(dump) bind(c,name='gpu_get_debug_dump')
          import
          implicit none
          integer(C_INT), intent(out):: dump(*)
         end function gpu_get_debug_dump
 !NV-TAL query/action API:
  !Check whether GPU belongs to the current process:
         integer(C_INT) function gpu_is_mine(gpu_num) bind(c,name='gpu_is_mine')
          import
          implicit none
          integer(C_INT), value, intent(in):: gpu_num
         end function gpu_is_mine
  !Returns the ID of the least busy NVidia GPU (which belongs to the process):
         integer(C_INT) function gpu_busy_least() bind(c,name='gpu_busy_least')
          import
          implicit none
         end function gpu_busy_least
  !Activate a specific GPU (only if it belongs to the process):
         integer(C_INT) function gpu_activate(gpu_num) bind(c,name='gpu_activate')
          import
          implicit none
          integer(C_INT), value, intent(in):: gpu_num
         end function gpu_activate
 !NV-TAL internal control:
  !Set the width of the NVidia GPU shared memory bank:
         integer(C_INT) function gpu_set_shmem_width(width) bind(c,name='gpu_set_shmem_width')
          import
          implicit none
          integer(C_INT), value, intent(in):: width
         end function gpu_set_shmem_width
  !Set the tensor transpose algorithm:
         subroutine gpu_set_transpose_algorithm(alg) bind(c,name='gpu_set_transpose_algorithm')
          import
          implicit none
          integer(C_INT), value, intent(in):: alg
         end subroutine gpu_set_transpose_algorithm
  !Set the matrix multiplication algorithm:
         subroutine gpu_set_matmult_algorithm(alg) bind(c,name='gpu_set_matmult_algorithm')
          import
          implicit none
          integer(C_INT), value, intent(in):: alg
         end subroutine gpu_set_matmult_algorithm
 !NV-TAL CUDA task API:
  !Create a CUDA task:
         integer(C_INT) function cuda_task_create(cuda_task) bind(c,name='cuda_task_create')
          import
          implicit none
          type(C_PTR), intent(out):: cuda_task
         end function cuda_task_create
  !Clean a CUDA task:
         integer(C_INT) function cuda_task_clean(cuda_task) bind(c,name='cuda_task_clean')
          import
          implicit none
          type(C_PTR), value:: cuda_task
         end function cuda_task_clean
  !Destroy a CUDA task:
         integer(C_INT) function cuda_task_destroy(cuda_task) bind(c,name='cuda_task_destroy')
          import
          implicit none
          type(C_PTR), value:: cuda_task
         end function cuda_task_destroy
  !Get the GPU ID the CUDA task is scheduled on:
         integer(C_INT) function cuda_task_gpu_id(cuda_task) bind(c,name='cuda_task_gpu_id')
          import
          implicit none
          type(C_PTR), value, intent(in):: cuda_task
         end function cuda_task_gpu_id
  !Get the CUDA task status:
         integer(C_INT) function cuda_task_status(cuda_task) bind(c,name='cuda_task_status')
          import
          implicit none
          type(C_PTR), value:: cuda_task
         end function cuda_task_status
  !Query CUDA task completion:
         integer(C_INT) function cuda_task_complete(cuda_task) bind(c,name='cuda_task_completed')
          import
          implicit none
          type(C_PTR), value:: cuda_task
         end function cuda_task_complete
  !Wait on completion of a CUDA task:
         integer(C_INT) function cuda_task_wait(cuda_task) bind(c,name='cuda_task_wait')
          import
          implicit none
          type(C_PTR), value:: cuda_task
         end function cuda_task_wait
  !Get the task timing in seconds:
         real(C_FLOAT) function cuda_task_time_(cuda_task,in_copy,out_copy,comp,mmul) bind(c,name='cuda_task_time_')
          import
          implicit none
          type(C_PTR), value, intent(in):: cuda_task
          real(C_FLOAT), intent(out):: in_copy
          real(C_FLOAT), intent(out):: out_copy
          real(C_FLOAT), intent(out):: comp
          real(C_FLOAT), intent(out):: mmul
         end function cuda_task_time_
#endif
        end interface

        end module tensor_algebra
