!ExaTensor::TAL-SH: Device-unified user-level API:
!REVISION: 2019/04/06

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
!--------------------------------------------------------------------------------
       module talsh
        use tensor_algebra_cpu_phi !device-specific tensor algebra API + basic
        implicit none
        private
!EXTERNAL PUBLIC:
        public tensor_shape_t         !CP-TAL tensor shape (Fortran)
        public tensor_block_t         !CP-TAL tensor block (Fortran)
        public talsh_tens_signature_t !TAL-SH tensor signature
        public talsh_tens_shape_t     !TAL-SH tensor shape
        public talsh_tens_data_t      !TAL-SH tensor data descriptor
        public MAX_SHAPE_STR_LEN      !max length of a shape-defining string
        public MAX_TENSOR_RANK        !max tensor rank
        public mem_allocate           !universal memory allocator
        public mem_free               !universal memory deallocator
        public tensor_shape_rank      !
        public get_contr_pattern_dig  !
        public get_contr_pattern_sym  !
        public contr_pattern_rnd      !
        public prof_push,prof_pop     !profiling
!PARAMETERS:
 !Generic:
        integer(INTD), private:: CONS_OUT=6 !default output device for this module
        integer(INTD), private:: DEBUG=0    !debugging mode for this module
        logical, private:: VERBOSE=.TRUE.   !verbosity for errors
 !Errors (keep consistent with "talsh.h"):
        integer(C_INT), parameter, public:: TALSH_SUCCESS=0                   !success
        integer(C_INT), parameter, public:: TALSH_FAILURE=-666                !generic failure
        integer(C_INT), parameter, public:: TALSH_NOT_AVAILABLE=-888          !information or feature not avaiable (in principle)
        integer(C_INT), parameter, public:: TALSH_NOT_IMPLEMENTED=-999        !feature not implemented yet
        integer(C_INT), parameter, public:: TALSH_NOT_INITIALIZED=1000000     !TALSH library has not been initialized yet
        integer(C_INT), parameter, public:: TALSH_ALREADY_INITIALIZED=1000001 !TALSH library has already been initialized
        integer(C_INT), parameter, public:: TALSH_INVALID_ARGS=1000002        !invalid arguments passed to a procedure
        integer(C_INT), parameter, public:: TALSH_INTEGER_OVERFLOW=1000003    !integer overflow occurred
        integer(C_INT), parameter, public:: TALSH_OBJECT_NOT_EMPTY=1000004    !object is not empty while expected so
        integer(C_INT), parameter, public:: TALSH_OBJECT_IS_EMPTY=1000005     !object is empty while not expected so
        integer(C_INT), parameter, public:: TALSH_IN_PROGRESS=1000006         !TAL-SH operation is still in progress (not finished)
        integer(C_INT), parameter, public:: TALSH_NOT_ALLOWED=1000007         !request is not allowed by TAL-SH
        integer(C_INT), parameter, public:: TALSH_LIMIT_EXCEEDED=1000008      !internal limit exceeded
        integer(C_INT), parameter, public:: TALSH_NOT_FOUND=1000009           !requested object not found
 !TAL-SH task status:
        integer(C_INT), parameter, public:: TALSH_TASK_ERROR=1999999
        integer(C_INT), parameter, public:: TALSH_TASK_EMPTY=2000000
        integer(C_INT), parameter, public:: TALSH_TASK_SCHEDULED=2000001
        integer(C_INT), parameter, public:: TALSH_TASK_STARTED=2000002
        integer(C_INT), parameter, public:: TALSH_TASK_INPUT_READY=2000003
        integer(C_INT), parameter, public:: TALSH_TASK_OUTPUT_READY=2000004
        integer(C_INT), parameter, public:: TALSH_TASK_COMPLETED=2000005
 !Host argument buffer:
        integer(C_SIZE_T), parameter, private:: HAB_SIZE_DEFAULT=16777216 !default size of the Host argument buffer in bytes
 !CP-TAL:
        integer(C_INT), parameter, private:: CPTAL_MAX_TMP_FTENS=192 !max number of simultaneously existing temporary Fortran tensors for CP-TAL
!DERIVED TYPES:
 !TAL-SH tensor block:
        type, public, bind(C):: talsh_tens_t
         type(C_PTR):: shape_p=C_NULL_PTR   !shape of the tensor block
         type(C_PTR):: dev_rsc=C_NULL_PTR   !list of device resources occupied by the tensor block body on each device
         type(C_PTR):: data_kind=C_NULL_PTR !list of data kinds for each device location occupied by the tensor body {R4,R8,C4,C8}
         type(C_PTR):: avail=C_NULL_PTR     !list of the data availability flags for each device location occupied by the tensor body
         integer(C_INT):: dev_rsc_len=0     !capacity of .dev_rsc[], .data_kind[], .avail[]
         integer(C_INT):: ndev=0            !number of devices the tensor block body resides on: ndev <= dev_rsc_len
        end type talsh_tens_t
 !Tensor operation argument (auxiliary type):
        type, bind(C):: talshTensArg_t
         type(C_PTR):: tens_p               !pointer to a tensor block
         integer(C_INT):: source_image      !specific body image of that tensor block participating in the operation
        end type talshTensArg_t
 !TAL-SH task handle:
        type, public, bind(C):: talsh_task_t
         type(C_PTR):: task_p=C_NULL_PTR    !pointer to the corresponding device-specific task object
         integer(C_INT):: task_error=-1     !-1:undefined(task in progress or empty); 0:successfully completed; >0: error code
         integer(C_INT):: dev_kind=DEV_NULL !device kind (DEV_NULL: uninitialized)
         integer(C_INT):: data_kind=NO_TYPE !data kind {R4,R8,C4,C8}, NO_TYPE: uninitialized
         integer(C_INT):: coherence=-1      !coherence control (-1:undefined)
         integer(C_INT):: num_args=0        !number of tensor arguments participating in the tensor operation
         type(talshTensArg_t):: tens_args(MAX_TENSOR_OPERANDS) !tensor arguments
         real(C_DOUBLE):: data_vol=0d0      !total data volume (information)
         real(C_DOUBLE):: flops=0d0         !number of floating point operations (information)
         real(C_DOUBLE):: exec_time=0d0     !execution time in seconds (information)
        end type talsh_task_t
!GLOBALS:
 !Temporary Fortran tensors for CP-TAL:
        integer(INTD), private:: ftens_len=0
        type(tensor_block_t), target, private:: ftensor(1:CPTAL_MAX_TMP_FTENS)

!INTERFACES FOR EXTERNAL C/C++ FUNCTIONS:
        interface
 !TAL-SH helper functions:
  !Check the validity of a data kind and get its size in bytes:
         integer(C_INT) function talsh_valid_data_kind(datk,datk_size) bind(c,name='talshValidDataKind')
          import
          implicit none
          integer(C_INT), intent(in), value:: datk
          integer(C_INT), intent(out):: datk_size
         end function talsh_valid_data_kind
 !TAL-SH control C/C++ API:
  !Initialize TAL-SH:
         integer(C_INT) function talshInit(host_buf_size,host_arg_max,ngpus,gpu_list,nmics,mic_list,namds,amd_list)&
                                          &bind(c,name='talshInit')
          import
          implicit none
          integer(C_SIZE_T), intent(inout):: host_buf_size
          integer(C_INT), intent(out):: host_arg_max
          integer(C_INT), value, intent(in):: ngpus
          integer(C_INT), intent(in):: gpu_list(*)
          integer(C_INT), value, intent(in):: nmics
          integer(C_INT), intent(in):: mic_list(*)
          integer(C_INT), value, intent(in):: namds
          integer(C_INT), intent(in):: amd_list(*)
         end function talshInit
  !Shutdown TAL-SH:
         integer(C_INT) function talshShutdown() bind(c,name='talshShutdown')
          import
          implicit none
         end function talshShutdown
  !Get on-node device count for a specific device kind:
         integer(C_INT) function talsh_device_count(dev_kind,dev_count) bind(c,name='talshDeviceCount')
          import
          implicit none
          integer(C_INT), intent(in), value:: dev_kind
          integer(C_INT), intent(out):: dev_count
         end function talsh_device_count
  !Get the flat device Id:
         integer(C_INT) function talshFlatDevId(dev_kind,dev_num) bind(c,name='talshFlatDevId')
          import
          implicit none
          integer(C_INT), value, intent(in):: dev_kind
          integer(C_INT), value, intent(in):: dev_num
         end function talshFlatDevId
  !Get the kind-specific device Id:
         integer(C_INT) function talshKindDevId(dev_id,dev_kind) bind(c,name='talshKindDevId')
          import
          implicit none
          integer(C_INT), value, intent(in):: dev_id
          integer(C_INT), intent(out):: dev_kind
         end function talshKindDevId
  !Query the state of a device:
         integer(C_INT) function talshDeviceState_(dev_num,dev_kind) bind(c,name='talshDeviceState_')
          import
          implicit none
          integer(C_INT), value, intent(in):: dev_num
          integer(C_INT), value, intent(in):: dev_kind
         end function talshDeviceState_
  !Find the least busy device:
         integer(C_INT) function talshDeviceBusyLeast_(dev_kind) bind(c,name='talshDeviceBusyLeast_')
          import
          implicit none
          integer(C_INT), value, intent(in):: dev_kind
         end function talshDeviceBusyLeast_
  !Query the device memory size in bytes:
         integer(C_SIZE_T) function talshDeviceMemorySize_(dev_num,dev_kind) bind(c,name='talshDeviceMemorySize_')
          import
          implicit none
          integer(C_INT), value, intent(in):: dev_num
          integer(C_INT), value, intent(in):: dev_kind
         end function talshDeviceMemorySize_
  !Query the device argument buffer size in bytes:
         integer(C_SIZE_T) function talshDeviceBufferSize_(dev_num,dev_kind) bind(c,name='talshDeviceBufferSize_')
          import
          implicit none
          integer(C_INT), value, intent(in):: dev_num
          integer(C_INT), value, intent(in):: dev_kind
         end function talshDeviceBufferSize_
  !Query the device max tensor size in bytes:
         integer(C_SIZE_T) function talshDeviceTensorSize_(dev_num,dev_kind) bind(c,name='talshDeviceTensorSize_')
          import
          implicit none
          integer(C_INT), value, intent(in):: dev_num
          integer(C_INT), value, intent(in):: dev_kind
         end function talshDeviceTensorSize_
  !Print run-time TAL-SH statistics for chosen devices:
         integer(C_INT) function talshStats_(dev_id,dev_kind) bind(c,name='talshStats_')
          import
          implicit none
          integer(C_INT), value, intent(in):: dev_id
          integer(C_INT), value, intent(in):: dev_kind
         end function talshStats_
 !TAL-SH tensor block C/C++ API:
  !Check whether a tensor block is empty (only be called on defined tensor blocks!):
         integer(C_INT) function talshTensorIsEmpty(tens_block) bind(c,name='talshTensorIsEmpty')
          import
          implicit none
          type(C_PTR), value, intent(in):: tens_block
         end function talshTensorIsEmpty
  !Construct a tensor block:
         integer(C_INT) function talshTensorConstruct_(tens_block,data_kind,tens_rank,tens_dims,dev_id,&
                        ext_mem,in_hab,init_method,init_val_real,init_val_imag) bind(c,name='talshTensorConstruct_')
          import
          implicit none
          type(talsh_tens_t):: tens_block
          integer(C_INT), value, intent(in):: data_kind
          integer(C_INT), value, intent(in):: tens_rank
          integer(C_INT), intent(in):: tens_dims(*)
          integer(C_INT), value, intent(in):: dev_id
          type(C_PTR), value:: ext_mem
          integer(C_INT), value, intent(in):: in_hab
          type(C_FUNPTR), value, intent(in):: init_method
          real(C_DOUBLE), value, intent(in):: init_val_real
          real(C_DOUBLE), value, intent(in):: init_val_imag
         end function talshTensorConstruct_
  !Destruct a tensor block:
         integer(C_INT) function talshTensorDestruct(tens_block) bind(c,name='talshTensorDestruct')
          import
          implicit none
          type(talsh_tens_t):: tens_block
         end function talshTensorDestruct
  !Get the rank of the tensor block:
         integer(C_INT) function talshTensorRank(tens_block) bind(c,name='talshTensorRank')
          import
          implicit none
          type(talsh_tens_t), intent(in):: tens_block
         end function talshTensorRank
  !Get the volume of the tensor block:
         integer(C_SIZE_T) function talshTensorVolume(tens_block) bind(c,name='talshTensorVolume')
          import
          implicit none
          type(talsh_tens_t), intent(in):: tens_block
         end function talshTensorVolume
  !Get the shape of the tensor block:
         integer(C_INT) function talshTensorShape(tens_block,tens_shape) bind(c,name='talshTensorShape')
          import
          implicit none
          type(talsh_tens_t), intent(in):: tens_block
          type(talsh_tens_shape_t), intent(inout):: tens_shape
         end function talshTensorShape
  !Get the data kind of each tensor image:
         integer(C_INT) function talshTensorDataKind(tens_block,num_images,data_kinds) bind(c,name='talshTensorDataKind')
          import
          implicit none
          type(talsh_tens_t), intent(in):: tens_block
          integer(C_INT), intent(out):: num_images
          integer(C_INT), intent(inout):: data_kinds(*)
         end function talshTensorDataKind
  !Query the presence of the tensor block on device(s):
         integer(C_INT) function talshTensorPresence_(tens_block,ncopies,copies,data_kinds,dev_kind,dev_id)&
                                 &bind(c,name='talshTensorPresence_')
          import
          implicit none
          type(talsh_tens_t), intent(in):: tens_block
          integer(C_INT), intent(out):: ncopies
          integer(C_INT), intent(inout):: copies(*)
          integer(C_INT), intent(inout):: data_kinds(*)
          integer(C_INT), value, intent(in):: dev_kind
          integer(C_INT), value, intent(in):: dev_id
         end function talshTensorPresence_
  !Get access to the tensor body image for a subsequent initialization:
         integer(C_INT) function talshTensorGetBodyAccess_(tens_block,body_p,data_kind,dev_id,dev_kind)&
                                 &bind(c,name='talshTensorGetBodyAccess_')
          import
          implicit none
          type(talsh_tens_t), intent(inout):: tens_block
          type(C_PTR), intent(inout):: body_p
          integer(C_INT), intent(in), value:: data_kind
          integer(C_INT), intent(in), value:: dev_id
          integer(C_INT), intent(in), value:: dev_kind
         end function talshTensorGetBodyAccess_
  !Get the scalar value of the rank-0 tensor:
         integer(C_INT) function talshTensorGetScalar_(tens_block,scalar_real,scalar_imag)&
                                 &bind(c,name='talshTensorGetScalar_')
          import
          implicit none
          type(talsh_tens_t), intent(inout):: tens_block
          real(C_DOUBLE), intent(out):: scalar_real
          real(C_DOUBLE), intent(out):: scalar_imag
         end function talshTensorGetScalar_
  !Print information about a TAL-SH tensor:
         subroutine talsh_tensor_print_info(tens_block) bind(c,name='talshTensorPrintInfo')
          import
          implicit none
          type(talsh_tens_t), intent(in):: tens_block
         end subroutine talsh_tensor_print_info
  !Print tensor elements larger by absolute value than some threshold:
         subroutine talsh_tensor_print_body(tens_block,thresh) bind(c,name='talshTensorPrintBody')
          import
          implicit none
          type(talsh_tens_t), intent(in):: tens_block
          real(C_DOUBLE), intent(in), value:: thresh
         end subroutine talsh_tensor_print_body
  ![DEBUG]: Compute the 1-norm of a tensor on Host CPU:
         real(C_DOUBLE) function talshTensorImageNorm1_cpu(talsh_tens) bind(c,name='talshTensorImageNorm1_cpu')
          import
          implicit none
          type(talsh_tens_t), intent(in):: talsh_tens
         end function talshTensorImageNorm1_cpu
 !TAL-SH task C/C++ API:
  !Clean an uninitialized TAL-SH task before the use:
         integer(C_INT) function talsh_task_clean(talsh_task) bind(c,name='talshTaskClean')
          import
          implicit none
          type(talsh_task_t), intent(inout):: talsh_task
         end function talsh_task_clean
  !Destruct a TAL-SH task:
         integer(C_INT) function talshTaskDestruct(talsh_task) bind(c,name='talshTaskDestruct')
          import
          implicit none
          type(talsh_task_t), intent(inout):: talsh_task
         end function talshTaskDestruct
  !Get the device id the TAL-SH task is scheduled on:
         integer(C_INT) function talshTaskDevId_(talsh_task,dev_kind) bind(c,name='talshTaskDevId_')
          import
          implicit none
          type(talsh_task_t), intent(inout):: talsh_task
          type(C_PTR), value:: dev_kind
         end function talshTaskDevId_
  !Get the TAL-SH task status:
         integer(C_INT) function talshTaskStatus(talsh_task) bind(c,name='talshTaskStatus')
          import
          implicit none
          type(talsh_task_t), intent(inout):: talsh_task
         end function talshTaskStatus
  !Check whether a TAL-SH task has completed:
         integer(C_INT) function talshTaskComplete(talsh_task,stats,ierr) bind(c,name='talshTaskComplete')
          import
          implicit none
          type(talsh_task_t), intent(inout):: talsh_task
          integer(C_INT), intent(out):: stats
          integer(C_INT), intent(out):: ierr
         end function talshTaskComplete
  !Wait upon completion of a TAL-SH task:
         integer(C_INT) function talshTaskWait(talsh_task,stats) bind(c,name='talshTaskWait')
          import
          implicit none
          type(talsh_task_t), intent(inout):: talsh_task
          integer(C_INT), intent(out):: stats
         end function talshTaskWait
  !Wait upon completion of multiple TAL-SH tasks:
         integer(C_INT) function talshTasksWait(ntasks,talsh_tasks,stats) bind(c,name='talshTasksWait')
          import
          implicit none
          integer(C_INT), value, intent(in):: ntasks
          type(talsh_task_t), intent(inout):: talsh_tasks(*)
          integer(C_INT), intent(out):: stats(*)
         end function talshTasksWait
  !Get the TAL-SH task timings:
         integer(C_INT) function talshTaskTime_(talsh_task,total,comput,input,output,mmul) bind(c,name='talshTaskTime_')
          import
          implicit none
          type(talsh_task_t), intent(inout):: talsh_task
          real(C_DOUBLE), intent(out):: total
          real(C_DOUBLE), intent(out):: comput
          real(C_DOUBLE), intent(out):: input
          real(C_DOUBLE), intent(out):: output
          real(C_DOUBLE), intent(out):: mmul
         end function talshTaskTime_
  !Print TAL-SH task info:
         subroutine talsh_task_print_info(talsh_task) bind(c,name='talshTaskPrint')
          import
          implicit none
          type(talsh_task_t), intent(in):: talsh_task
         end subroutine talsh_task_print_info
 !TAL-SH tensor operations C/C++ API:
  !Place a tensor block on a specific device:
         integer(C_INT) function talshTensorPlace_(tens,dev_id,dev_kind,dev_mem,copy_ctrl,talsh_task)&
                        &bind(c,name='talshTensorPlace_')
          import
          implicit none
          type(talsh_tens_t), intent(inout):: tens
          integer(C_INT), value, intent(in):: dev_id
          integer(C_INT), value, intent(in):: dev_kind
          type(C_PTR), value:: dev_mem
          integer(C_INT), value, intent(in):: copy_ctrl
          type(talsh_task_t), intent(inout):: talsh_task
         end function talshTensorPlace_
  !Discard a tensor block on a specific device:
         integer(C_INT) function talshTensorDiscard_(tens,dev_id,dev_kind) bind(c,name='talshTensorDiscard_')
          import
          implicit none
          type(talsh_tens_t), intent(inout):: tens
          integer(C_INT), value, intent(in):: dev_id
          integer(C_INT), value, intent(in):: dev_kind
         end function talshTensorDiscard_
  !Discard a tensor block on all devices except a specific device:
         integer(C_INT) function talshTensorDiscardOther_(tens,dev_id,dev_kind) bind(c,name='talshTensorDiscardOther_')
          import
          implicit none
          type(talsh_tens_t), intent(inout):: tens
          integer(C_INT), value, intent(in):: dev_id
          integer(C_INT), value, intent(in):: dev_kind
         end function talshTensorDiscardOther_
  !Tensor initialization:
         integer(C_INT) function talshTensorInit_(dtens,val_real,val_imag,dev_id,dev_kind,copy_ctrl,talsh_task)&
                                                 &bind(c,name='talshTensorInit_')
          import
          implicit none
          type(talsh_tens_t), intent(inout):: dtens
          real(C_DOUBLE), value, intent(in):: val_real
          real(C_DOUBLE), value, intent(in):: val_imag
          integer(C_INT), value, intent(in):: dev_id
          integer(C_INT), value, intent(in):: dev_kind
          integer(C_INT), value, intent(in):: copy_ctrl
          type(talsh_task_t), intent(inout):: talsh_task
         end function talshTensorInit_
  !Tensor slicing:
         integer(C_INT) function talshTensorSlice_(dtens,ltens,offsets,dev_id,dev_kind,copy_ctrl,talsh_task)&
                                                  &bind(c,name='talshTensorSlice_')
          import
          implicit none
          type(talsh_tens_t), intent(inout):: dtens
          type(talsh_tens_t), intent(inout):: ltens
          integer(C_INT), intent(in):: offsets(*)
          integer(C_INT), value, intent(in):: dev_id
          integer(C_INT), value, intent(in):: dev_kind
          integer(C_INT), value, intent(in):: copy_ctrl
          type(talsh_task_t), intent(inout):: talsh_task
         end function talshTensorSlice_
  !Tensor insertion:
         integer(C_INT) function talshTensorInsert_(dtens,ltens,offsets,dev_id,dev_kind,copy_ctrl,talsh_task)&
                                                   &bind(c,name='talshTensorInsert_')
          import
          implicit none
          type(talsh_tens_t), intent(inout):: dtens
          type(talsh_tens_t), intent(inout):: ltens
          integer(C_INT), intent(in):: offsets(*)
          integer(C_INT), value, intent(in):: dev_id
          integer(C_INT), value, intent(in):: dev_kind
          integer(C_INT), value, intent(in):: copy_ctrl
          type(talsh_task_t), intent(inout):: talsh_task
         end function talshTensorInsert_
  !Tensor copy:
         integer(C_INT) function talshTensorCopy_(dtens,ltens,permutation,dev_id,dev_kind,copy_ctrl,talsh_task)&
                                                 &bind(c,name='talshTensorCopy_')
          import
          implicit none
          type(talsh_tens_t), intent(inout):: dtens
          type(talsh_tens_t), intent(inout):: ltens
          integer(C_INT), intent(in):: permutation(*)
          integer(C_INT), value, intent(in):: dev_id
          integer(C_INT), value, intent(in):: dev_kind
          integer(C_INT), value, intent(in):: copy_ctrl
          type(talsh_task_t), intent(inout):: talsh_task
         end function talshTensorCopy_
  !Tensor addition:
         integer(C_INT) function talshTensorAdd_(cptrn,dtens,ltens,scale_real,scale_imag,dev_id,dev_kind,&
                                                &copy_ctrl,talsh_task) bind(c,name='talshTensorAdd_')
          import
          implicit none
          character(C_CHAR), intent(in):: cptrn(*)
          type(talsh_tens_t), intent(inout):: dtens
          type(talsh_tens_t), intent(inout):: ltens
          real(C_DOUBLE), value, intent(in):: scale_real
          real(C_DOUBLE), value, intent(in):: scale_imag
          integer(C_INT), value, intent(in):: dev_id
          integer(C_INT), value, intent(in):: dev_kind
          integer(C_INT), value, intent(in):: copy_ctrl
          type(talsh_task_t), intent(inout):: talsh_task
         end function talshTensorAdd_
  !Tensor contraction:
         integer(C_INT) function talshTensorContract_(cptrn,dtens,ltens,rtens,scale_real,scale_imag,dev_id,dev_kind,&
                                                     &copy_ctrl,accumulative,talsh_task) bind(c,name='talshTensorContract_')
          import
          implicit none
          character(C_CHAR), intent(in):: cptrn(*)
          type(talsh_tens_t), intent(inout):: dtens
          type(talsh_tens_t), intent(inout):: ltens
          type(talsh_tens_t), intent(inout):: rtens
          real(C_DOUBLE), value, intent(in):: scale_real
          real(C_DOUBLE), value, intent(in):: scale_imag
          integer(C_INT), value, intent(in):: dev_id
          integer(C_INT), value, intent(in):: dev_kind
          integer(C_INT), value, intent(in):: copy_ctrl
          integer(C_INT), value, intent(in):: accumulative
          type(talsh_task_t), intent(inout):: talsh_task
         end function talshTensorContract_
 !Internal TAL-SH C/C++ API:
  !Obtains the information on a specific tensor body image:
         integer(C_INT) function talsh_tensor_image_info(talsh_tens,image_id,dev_id,data_kind,gmem_p,buf_entry)&
                  & bind(c,name='talsh_tensor_image_info')
          import
          implicit none
          type(talsh_tens_t), intent(in):: talsh_tens
          integer(C_INT), value, intent(in):: image_id
          integer(C_INT), intent(out):: dev_id
          integer(C_INT), intent(out):: data_kind
          type(C_PTR), intent(out):: gmem_p
          integer(C_INT), intent(out):: buf_entry
         end function talsh_tensor_image_info
 !CUDA runtime:
  !Get on-node GPU device count:
         integer(C_INT) function gpu_get_device_count(dev_count) bind(c,name='gpu_get_device_count')
          import
          implicit none
          integer(C_INT), intent(out):: dev_count
         end function gpu_get_device_count
        end interface
!INTERFACES FOR OVERLOADED FOTRAN FUNCTIONS:
        interface talsh_tensor_construct
         module procedure talsh_tensor_construct_num
         module procedure talsh_tensor_construct_sym
         module procedure talsh_tensor_construct_shp
        end interface talsh_tensor_construct
!VISIBILITY:
 !TAL-SH helper API:
        public talsh_valid_data_kind
 !TAL-SH control API:
        public talsh_init
        public talsh_shutdown
        public talsh_device_count
        public talsh_flat_dev_id
        public talsh_kind_dev_id
        public talsh_device_state
        public talsh_device_busy_least
        public talsh_device_memory_size
        public talsh_device_tensor_size
        public talsh_stats
 !TAL-SH tensor block API:
        public talsh_tensor_is_empty
        public talsh_tensor_construct
        private talsh_tensor_construct_num
        private talsh_tensor_construct_sym
        private talsh_tensor_construct_shp
        public talsh_tensor_destruct
        public talsh_tensor_rank
        public talsh_tensor_volume
        public talsh_tensor_dimensions
        public talsh_tensor_shape
        public talsh_tensor_data_kind
        public talsh_tensor_presence
        public talsh_tensor_get_body_access
        public talsh_tensor_get_scalar
        public talsh_tensor_print_info
        public talsh_tensor_print_body
        public talshTensorImageNorm1_cpu
 !TAL-SH task API:
       !private talsh_task_clean
        public talsh_task_destruct
        public talsh_task_dev_id
        public talsh_task_status
        public talsh_task_complete
        public talsh_task_wait
        public talsh_tasks_wait
        public talsh_task_time
        public talsh_task_print_info
 !TAL-SH tensor operations API:
        public talsh_tensor_place
        public talsh_tensor_discard
        public talsh_tensor_discard_other
        public talsh_tensor_init
!       public talsh_tensor_scale
!       public talsh_tensor_norm1
!       public talsh_tensor_norm2
        public talsh_tensor_slice
        public talsh_tensor_insert
!       public talsh_tensor_copy
        public talsh_tensor_add
        public talsh_tensor_contract

       contains
!INTERNAL FUNCTIONS:
!--------------------------------------------------------------------------------------------------------
        integer(C_INT) function talsh_get_contr_ptrn_str2dig(c_str,dig_ptrn,drank,lrank,rrank,conj_bits)&
                       &bind(c,name='talsh_get_contr_ptrn_str2dig')
         implicit none
         character(C_CHAR), intent(in):: c_str(1:*)  !in: C-string (NULL terminated) containing the mnemonic contraction pattern
         integer(C_INT), intent(out):: dig_ptrn(1:*) !out: digitial tensor contraction pattern
         integer(C_INT), intent(out):: drank         !out: destination tensor rank
         integer(C_INT), intent(out):: lrank         !out: left tensor rank
         integer(C_INT), intent(out):: rrank         !out: right tensor rank
         integer(C_INT), intent(out):: conj_bits     !out: argument complex conjugation flags (Bit 0 -> Destination, Bit 1 - > Left, Bit 2 -> Right)
         integer, parameter:: MAX_CONTR_STR_LEN=1024 !max length of the tensor contraction string
         integer:: dgp(MAX_TENSOR_RANK*2),dgl,csl,ierr
         character(MAX_CONTR_STR_LEN):: contr_str
         integer(INTD):: i,star_pos

         talsh_get_contr_ptrn_str2dig=0
         drank=-1; lrank=-1; rrank=-1; conj_bits=0
!Convert C-string to a Fortran string:
         csl=1; star_pos=0
         do while(iachar(c_str(csl)).ne.0)
          if(csl.gt.MAX_CONTR_STR_LEN) then
           talsh_get_contr_ptrn_str2dig=-1; return
          endif
          if(c_str(csl).eq.'*'.and.star_pos.eq.0) star_pos=csl
          contr_str(csl:csl)=c_str(csl); csl=csl+1
         enddo
         csl=csl-1
!Add a fake third argument in case of addition:
         if(star_pos.gt.0) then !check for possible scalar multiplication
          i=index(contr_str(star_pos:csl),'(')+star_pos-1
          if(i.lt.star_pos) then
           csl=star_pos-1; star_pos=0 !remove scalar multiplication
           if(csl.le.0) then; talsh_get_contr_ptrn_str2dig=-2; return; endif
           if(contr_str(csl:csl).ne.')') then; talsh_get_contr_ptrn_str2dig=-3; return; endif
          endif
         endif
         if(star_pos.eq.0) then !this is not a contraction (less than three arguments)
          i=len('*R()'); contr_str(csl+1:csl+i)='*R()'; csl=csl+i !add a fake third argument (cast as a contraction)
         endif
!Call converter from CP-TAL:
         if(csl.gt.0) then
          call get_contr_pattern_dig(contr_str(1:csl),drank,lrank,rrank,dgp,ierr,conj_bits)
          if(ierr.eq.0) then
           dgl=lrank+rrank; if(dgl.gt.0) dig_ptrn(1:dgl)=dgp(1:dgl)
          else
           talsh_get_contr_ptrn_str2dig=ierr; return
          endif
         endif
         return
        end function talsh_get_contr_ptrn_str2dig
!------------------------------------------
        subroutine get_f_tensor(ftens,ierr)
         implicit none
         type(tensor_block_t), intent(out), pointer:: ftens
         integer(INTD), intent(out):: ierr

         ierr=0
!$OMP CRITICAL (CPTAL_TMP_FTENS)
         if(ftens_len.lt.CPTAL_MAX_TMP_FTENS) then
          ftens_len=ftens_len+1
          ftens=>ftensor(ftens_len)
         else
          ftens=>NULL(); ierr=-1
         endif
!$OMP END CRITICAL (CPTAL_TMP_FTENS)
         return
        end subroutine get_f_tensor
!---------------------------------------------
        subroutine return_f_tensor(ftens,ierr)
         implicit none
         type(tensor_block_t), intent(in), pointer:: ftens
         integer(INTD), intent(out):: ierr
         type(tensor_block_t), pointer:: ft
         integer(INTD):: i

         ierr=0
!$OMP CRITICAL (CPTAL_TMP_FTENS)
         if(associated(ftens)) then
          do i=ftens_len,1,-1
           ft=>ftensor(i)
           if(associated(ft,ftens)) then; exit; else; ft=>NULL(); endif
          enddo
          if(associated(ft).and.(i.ge.1.and.i.le.ftens_len)) then
           if(i.ne.ftens_len) ftensor(i)=ftensor(ftens_len) !move tensor_block_t (it has no allocatable components)
           ftens_len=ftens_len-1
          else
           ierr=-2
          endif
         else
          ierr=-1
         endif
!$OMP END CRITICAL (CPTAL_TMP_FTENS)
         return
        end subroutine return_f_tensor
!------------------------------------------------------------------------------------------------------------------
        integer(C_INT) function talsh_tensor_f_assoc(talsh_tens,image_id,tensF) bind(c,name='talsh_tensor_f_assoc')
!Returns a C pointer <tensF> to a <tensor_block_t> object instantiated with the tensor body image <image_id>.
!A return status TALSH_NOT_ALLOWED indicates that the requested tensor body image
!is no longer available (to be discarded by runtime).
         implicit none
         type(talsh_tens_t), intent(in):: talsh_tens        !in: TAL-SH tensor
         integer(C_INT), value, intent(in):: image_id       !in: tensor body image id
         type(C_PTR), intent(out):: tensF                   !out: C pointer to <tensor_block_t> associated with the TAL-SH tensor image
         type(tensor_block_t), pointer:: ftens
         type(talsh_tens_shape_t), pointer:: tens_shape
         type(tensor_shape_t):: tshape
         integer(C_INT), pointer, contiguous:: dims(:),divs(:),grps(:)
         integer(C_INT):: devid,dtk,buf_entry,errc
         type(C_PTR):: gmem_p
         integer(INTD):: n,ierr

         talsh_tensor_f_assoc=TALSH_SUCCESS
         if(.not.talsh_tensor_is_empty(talsh_tens)) then
          if(image_id.ge.0.and.image_id.lt.talsh_tens%ndev) then
           if(c_associated(talsh_tens%dev_rsc).and.c_associated(talsh_tens%data_kind).and.c_associated(talsh_tens%avail).and.&
             &talsh_tens%ndev.gt.0.and.talsh_tens%ndev.le.talsh_tens%dev_rsc_len) then
            call c_f_pointer(talsh_tens%shape_p,tens_shape)
            n=tens_shape%num_dim
            if(n.ge.0) then
             call get_f_tensor(ftens,ierr)
             if(ierr.eq.0) then
              if(n.gt.0) then
               if(c_associated(tens_shape%dims)) then
                call c_f_pointer(tens_shape%dims,dims,shape=(/n/))
               else
                dims=>NULL()
               endif
               if(c_associated(tens_shape%divs)) then
                call c_f_pointer(tens_shape%divs,divs,shape=(/n/))
               else
                divs=>NULL()
               endif
               if(c_associated(tens_shape%grps)) then
                call c_f_pointer(tens_shape%grps,grps,shape=(/n/))
               else
                grps=>NULL()
               endif
              else
               dims=>NULL(); divs=>NULL(); grps=>NULL()
              endif
              call tensor_shape_assoc(tshape,ierr,dims,divs,grps)
              if(ierr.eq.0) then
               errc=talsh_tensor_image_info(talsh_tens,image_id,devid,dtk,gmem_p,buf_entry)
               if(errc.eq.0) then
                call tensor_block_assoc(ftens,tshape,dtk,gmem_p,errc)
                if(errc.ne.0) talsh_tensor_f_assoc=TALSH_FAILURE
               else
                if(errc.eq.TALSH_NOT_ALLOWED) then
                 talsh_tensor_f_assoc=TALSH_NOT_ALLOWED !requested image is not available (to be discarded)
                else
                 talsh_tensor_f_assoc=TALSH_FAILURE
                endif
               endif
              else
               talsh_tensor_f_assoc=TALSH_FAILURE
              endif
              if(talsh_tensor_f_assoc.eq.TALSH_SUCCESS) then
               tensF=c_loc(ftens)
              else
               call tensor_block_destroy(ftens,ierr)
               call return_f_tensor(ftens,ierr); if(ierr.ne.0) talsh_tensor_f_assoc=TALSH_FAILURE
               tensF=C_NULL_PTR
              endif
             else
              talsh_tensor_f_assoc=TRY_LATER
             endif
            else
             talsh_tensor_f_assoc=TALSH_FAILURE
            endif
           else
            talsh_tensor_f_assoc=TALSH_FAILURE
           endif
          else
           talsh_tensor_f_assoc=TALSH_INVALID_ARGS
          endif
         else
          talsh_tensor_f_assoc=TALSH_OBJECT_IS_EMPTY
         endif
         return
        end function talsh_tensor_f_assoc
!------------------------------------------------------------------------------------------------
        integer(C_INT) function talsh_tensor_f_dissoc(tensF) bind(c,name='talsh_tensor_f_dissoc')
!Destroys a temporary <tensor_block_t> object associated with a specific image of some TAL-SH tensor.
         implicit none
         type(C_PTR), value:: tensF !in: C pointer to a dynamically allocated <tensor_block_t> object by <talsh_tensor_f_assoc()>
         type(tensor_block_t), pointer:: ftens
         integer:: ierr

         talsh_tensor_f_dissoc=TALSH_SUCCESS
         if(c_associated(tensF)) then
          call c_f_pointer(tensF,ftens)
          if(.not.tensor_block_is_empty(ftens,ierr)) then
           if(ierr.eq.0) then
            call tensor_block_destroy(ftens,ierr)
            if(ierr.ne.0) then
             if(ierr.eq.NOT_CLEAN) then
              talsh_tensor_f_dissoc=NOT_CLEAN
             else
              talsh_tensor_f_dissoc=TALSH_FAILURE
             endif
            endif
            call return_f_tensor(ftens,ierr)
            if(ierr.ne.0.and.talsh_tensor_f_dissoc.eq.TALSH_SUCCESS) talsh_tensor_f_dissoc=TALSH_FAILURE
           else
            talsh_tensor_f_dissoc=TALSH_FAILURE
           endif
          else
           talsh_tensor_f_dissoc=TALSH_OBJECT_IS_EMPTY
          endif
         else
          talsh_tensor_f_dissoc=TALSH_OBJECT_IS_EMPTY
         endif
         return
        end function talsh_tensor_f_dissoc
!----------------------------------------------------------------------------
        integer(C_INT) function talsh_update_f_scalar(tensF,data_kind,gmem_p) bind(c,name='talsh_update_f_scalar')
!Updates the given memory location <gmem_p> with the value of a scalar tensor.
!The memory location is the (single-element) body of a scalar tensor of type <talsh_tens_t>.
         implicit none
         type(C_PTR), value:: tensF                    !in: C pointer to <tensor_block_t>
         integer(C_INT), intent(in), value:: data_kind !in: data kind
         type(C_PTR), value:: gmem_p                   !in: C pointer to the single-element body of a <talsh_tens_t> image
         type(tensor_block_t), pointer:: ftens
         integer:: ierr
         real(4), pointer:: r4p
         real(8), pointer:: r8p
         complex(4), pointer:: c4p
         complex(8), pointer:: c8p
         complex(8):: val

         talsh_update_f_scalar=TALSH_SUCCESS
         if(c_associated(tensF)) then
          call c_f_pointer(tensF,ftens)
          if(.not.tensor_block_is_empty(ftens,ierr)) then
           if(ierr.eq.0) then
            if(c_associated(gmem_p)) then
             val=tensor_block_scalar_value(ftens)
             select case(data_kind)
             case(R4)
              call c_f_pointer(gmem_p,r4p); r4p=real(val,4); r4p=>NULL()
             case(R8)
              call c_f_pointer(gmem_p,r8p); r8p=real(val,8); r8p=>NULL()
             case(C4)
              call c_f_pointer(gmem_p,c4p); c4p=cmplx(real(val),imag(val),4); c4p=>NULL()
             case(C8)
              call c_f_pointer(gmem_p,c8p); c8p=val; c8p=>NULL()
             case default
              talsh_update_f_scalar=TALSH_INVALID_ARGS
             end select
            else
             talsh_update_f_scalar=TALSH_INVALID_ARGS
            endif
           else
            talsh_update_f_scalar=TALSH_FAILURE
           endif
          else
           talsh_update_f_scalar=TALSH_OBJECT_IS_EMPTY
          endif
         else
          talsh_update_f_scalar=TALSH_OBJECT_IS_EMPTY
         endif
         return
        end function talsh_update_f_scalar
!---------------------------------------------------------------------------------------------------------------------
        subroutine talsh_set_mem_alloc_policy_host(mem_policy,fallback,ierr) bind(c,name='talshSetMemAllocPolicyHost')
!Wrapper for CP-TAL set_mem_alloc_policy() for C/C++.
         implicit none
         integer(C_INT), intent(in), value:: mem_policy !in: CPU memory allocation policy for CP-TAL
         integer(C_INT), intent(in), value:: fallback   !in: fallback to regular allocation
         integer(C_INT), intent(out):: ierr             !out: error code
         integer:: mem_pol,errc
         logical:: fb

         mem_pol=mem_policy; fb=(fallback.ne.0)
         call set_mem_alloc_policy(mem_pol,errc,fb); ierr=errc
         return
        end subroutine talsh_set_mem_alloc_policy_host
!-----------------------------------------------------
!FORTRAN TAL-SH API DEFINITIONS:
 !TAL-SH control API:
!----------------------------------------------------------------------------------------------
        function talsh_init(host_buf_size,host_arg_max,gpu_list,mic_list,amd_list) result(ierr)
         implicit none
         integer(C_INT):: ierr                                      !out: error code (0:success)
         integer(C_SIZE_T), intent(inout), optional:: host_buf_size !inout: desired size in bytes of the Host Argument Buffer (HAB).
                                                                    !       It will be replaced by the actual size.
         integer(C_INT), intent(out), optional:: host_arg_max       !out: max number of arguments the HAB can contain
         integer(C_INT), intent(in), optional:: gpu_list(1:)        !in: list of NVidia GPU's to use
         integer(C_INT), intent(in), optional:: mic_list(1:)        !in: list of Intel Xeon Phi's to use
         integer(C_INT), intent(in), optional:: amd_list(1:)        !in: list of AMD GPU's to use
         integer(C_INT):: ngpus,gpus(MAX_GPUS_PER_NODE)
         integer(C_INT):: nmics,mics(MAX_MICS_PER_NODE)
         integer(C_INT):: namds,amds(MAX_AMDS_PER_NODE)
         integer(C_SIZE_T):: hbuf_size
         integer(C_INT):: harg_max

         if(present(host_buf_size)) then; hbuf_size=host_buf_size; else; hbuf_size=HAB_SIZE_DEFAULT; endif
         if(present(gpu_list)) then; ngpus=size(gpu_list); gpus(1:ngpus)=gpu_list(1:ngpus); else; ngpus=0; endif
         if(present(mic_list)) then; nmics=size(mic_list); mics(1:nmics)=mic_list(1:nmics); else; nmics=0; endif
         if(present(amd_list)) then; namds=size(amd_list); amds(1:namds)=amd_list(1:namds); else; namds=0; endif
         ierr=talshInit(hbuf_size,harg_max,ngpus,gpus,nmics,mics,namds,amds)
         if(present(host_arg_max)) host_arg_max=harg_max
         if(present(host_buf_size)) host_buf_size=hbuf_size
         return
        end function talsh_init
!---------------------------------------------
        function talsh_shutdown() result(ierr)
         implicit none
         integer(C_INT):: ierr !out: error code (0:success)
         ierr=talshShutdown()
         return
        end function talsh_shutdown
!----------------------------------------------------------------
        function talsh_flat_dev_id(dev_kind,dev_num) result(res)
         implicit none
         integer(C_INT):: res                  !out: flat device Id [0..DEV_MAX-1]; Failure: DEV_MAX
         integer(C_INT), intent(in):: dev_kind !in: device kind
         integer(C_INT), intent(in):: dev_num  !in: device Id within its kind (0..MAX)
         res=talshFlatDevId(dev_kind,dev_num)
         return
        end function talsh_flat_dev_id
!--------------------------------------------------------------
        function talsh_kind_dev_id(dev_id,dev_kind) result(res)
         implicit none
         integer(C_INT):: res                   !out: kind-specific device Id [0..MAX]; Failure: DEV_NULL (negative)
         integer(C_INT), intent(in):: dev_id    !in: flat device Id
         integer(C_INT), intent(out):: dev_kind !out: device kind
         res=talshKindDevId(dev_id,dev_kind)
         return
        end function talsh_kind_dev_id
!----------------------------------------------------------------------
        function talsh_device_state(dev_num,dev_kind) result(dev_state)
         implicit none
         integer(C_INT):: dev_state                      !out: device state (Success:[DEV_OFF,DEV_ON,DEV_ON_BLAS])
         integer(C_INT), intent(in):: dev_num            !in: either a flat or kind specific (when <dev_kind> is present) device id
         integer(C_INT), intent(in), optional:: dev_kind !in: device kind (note that it changes the meaning of the <dev_num> argument)
         integer(C_INT):: devk

         if(present(dev_kind)) then; devk=dev_kind; else; devk=DEV_NULL; endif
         dev_state=talshDeviceState_(dev_num,devk)
         return
        end function talsh_device_state
!----------------------------------------------------------------
        function talsh_device_busy_least(dev_kind) result(dev_id)
         implicit none
         integer(C_INT):: dev_id                         !out: either a flat or kind specific device id
         integer(C_INT), intent(in), optional:: dev_kind !in: device kind (if absent, <dev_id> will return the flat device id)
         integer(C_INT):: devk

         if(present(dev_kind)) then; devk=dev_kind; else; devk=DEV_NULL; endif
         dev_id=talshDeviceBusyLeast_(devk)
         return
        end function talsh_device_busy_least
!---------------------------------------------------------------------------
        function talsh_device_memory_size(dev_num,dev_kind) result(mem_size)
         implicit none
         integer(C_SIZE_T):: mem_size                    !out: device memory size in bytes
         integer(C_INT), intent(in):: dev_num            !in: either a flat or kind specific (when <dev_kind> is present) device id
         integer(C_INT), intent(in), optional:: dev_kind !in: device kind (note that it changes the meaning of the <dev_num> argument)
         integer(C_INT):: devk

         if(present(dev_kind)) then; devk=dev_kind; else; devk=DEV_NULL; endif
         mem_size=talshDeviceMemorySize_(dev_num,devk)
         return
        end function talsh_device_memory_size
!----------------------------------------------------------------------------
        function talsh_device_tensor_size(dev_num,dev_kind) result(tens_size)
         implicit none
         integer(C_SIZE_T):: tens_size                   !out: max tensor size in bytes on a given device
         integer(C_INT), intent(in):: dev_num            !in: either a flat or kind specific (when <dev_kind> is present) device id
         integer(C_INT), intent(in), optional:: dev_kind !in: device kind (note that it changes the meaning of the <dev_num> argument)
         integer(C_INT):: devk

         if(present(dev_kind)) then; devk=dev_kind; else; devk=DEV_NULL; endif
         tens_size=talshDeviceTensorSize_(dev_num,devk)
         return
        end function talsh_device_tensor_size
!---------------------------------------------------------
        function talsh_stats(dev_id,dev_kind) result(ierr)
         implicit none
         integer(C_INT):: ierr                           !out: error code (0:success)
         integer(C_INT), intent(in), optional:: dev_id   !in: device id (either flat or kind specific device id, see below)
         integer(C_INT), intent(in), optional:: dev_kind !in: device kind (if present, <dev_id> will be interpreted as kind specific)
         integer(C_INT):: devn,devk

         if(present(dev_id)) then; devn=dev_id; else; devn=-1; endif
         if(present(dev_kind)) then; devk=dev_kind; else; devk=DEV_NULL; endif
         ierr=talshStats_(devn,devk)
         return
        end function talsh_stats
!-------------------------------------------------------------
        function talsh_tensor_is_empty(tens_block) result(res)
         implicit none
         logical:: res                                       !out: .TRUE. if the tensor block is empty, .FALSE. otherwise
         type(talsh_tens_t), intent(in), target:: tens_block !in: tensor block

         res=.FALSE.
         if(talshTensorIsEmpty(c_loc(tens_block)).eq.YEP) res=.TRUE.
         return
        end function talsh_tensor_is_empty
!-----------------------------------------------------------------------------------------------------------------------------------
        function talsh_tensor_construct_num(tens_block,data_kind,tens_shape,dev_id,ext_mem,in_hab,init_method,init_val) result(ierr)
         implicit none
         integer(C_INT):: ierr                                !out: error code (0:success)
         type(talsh_tens_t), intent(inout):: tens_block       !inout: constructed tensor block (must be empty on entrance)
         integer(C_INT), intent(in):: data_kind               !in: data kind: {R4,R8,C4,C8,NO_TYPE}
         integer(C_INT), intent(in):: tens_shape(1:)          !in: tensor shape (length = tensor rank)
         integer(C_INT), intent(in), optional:: dev_id        !in: flat device ID on which the tensor block will reside
         type(C_PTR), intent(in), optional:: ext_mem          !in: pointer to externally provided memory for tensor elements
         integer(C_INT), intent(in), optional:: in_hab        !in: if >=0, <ext_mem> points to the HAB entry #<in_hab>
         procedure(talsh_tens_init_i), optional:: init_method !in: user-defined initialization method (<init_val> must be absent)
         complex(8), intent(in), optional:: init_val          !in: initialization value (will be typecast to <data_kind>, defaults to 0)
         type(C_PTR):: tens_body_p
         integer(C_INT):: devid,hab_entry,tens_rank
         integer(C_INT), target:: tens_dims(1:MAX_TENSOR_RANK)
         type(C_FUNPTR):: init_method_p
         real(C_DOUBLE):: val_real,val_imag

         ierr=TALSH_SUCCESS
         tens_rank=size(tens_shape) !tens_shape(1:) must have the exact volume = tensor rank
         if(tens_rank.ge.0.and.tens_rank.le.MAX_TENSOR_RANK) then
          if(tens_rank.gt.0) tens_dims(1:tens_rank)=tens_shape(1:tens_rank)
          if(present(dev_id)) then; devid=dev_id; else; devid=talshFlatDevId(DEV_HOST,0); endif
          if(present(ext_mem)) then; tens_body_p=ext_mem; else; tens_body_p=C_NULL_PTR; endif
          if(present(in_hab)) then; if(in_hab.ge.0) then; hab_entry=in_hab; else; hab_entry=-1; endif; else; hab_entry=-1; endif
          if(present(init_method)) then; init_method_p=c_funloc(init_method); else; init_method_p=C_NULL_FUNPTR; endif
          val_real=0d0; val_imag=0d0; if(present(init_val)) then; val_real=real(init_val); val_imag=aimag(init_val); endif
          ierr=talshTensorConstruct_(tens_block,data_kind,tens_rank,tens_dims,devid,&
                                     tens_body_p,hab_entry,init_method_p,val_real,val_imag)
         else
          ierr=TALSH_INVALID_ARGS
         endif
         return
        end function talsh_tensor_construct_num
!-----------------------------------------------------------------------------------------------------------------------------------
        function talsh_tensor_construct_sym(tens_block,data_kind,tens_shape,dev_id,ext_mem,in_hab,init_method,init_val) result(ierr)
         implicit none
         integer(C_INT):: ierr                                !out: error code (0:success)
         type(talsh_tens_t), intent(inout):: tens_block       !inout: constructed tensor block (must be empty on entrance)
         integer(C_INT), intent(in):: data_kind               !in: data kind: {R4,R8,C4,C8,NO_TYPE}
         character(*), intent(in):: tens_shape                !in: tensor shape (symbolic)
         integer(C_INT), intent(in), optional:: dev_id        !in: flat device ID on which the tensor block will reside
         type(C_PTR), intent(in), optional:: ext_mem          !in: pointer to externally provided memory for tensor elements
         integer(C_INT), intent(in), optional:: in_hab        !in: if >=0, <ext_mem> points to the HAB entry #<in_hab>
         procedure(talsh_tens_init_i), optional:: init_method !in: user-defined initialization method (<init_val> must be absent)
         complex(8), intent(in), optional:: init_val          !in: initialization value (will be typecast to <data_kind>, defaults to 0)
         type(tensor_block_t):: ftens
         type(C_PTR):: extmem
         integer(C_INT):: devid,inhab,trank,tshape(1:MAX_TENSOR_RANK)
         integer:: errc

         call tensor_block_shape_create(ftens,tens_shape,errc)
         trank=ftens%tensor_shape%num_dim
         if(errc.eq.0.and.trank.ge.0.and.trank.le.MAX_TENSOR_RANK) then
          tshape(1:trank)=ftens%tensor_shape%dim_extent(1:trank)
          if(present(dev_id)) then; devid=dev_id; else; devid=talshFlatDevId(DEV_HOST,0); endif
          if(present(ext_mem)) then; extmem=ext_mem; else; extmem=C_NULL_PTR; endif
          if(present(in_hab)) then; inhab=in_hab; else; inhab=-1; endif
          if(present(init_method)) then
           ierr=talsh_tensor_construct_num(tens_block,data_kind,tshape(1:trank),devid,extmem,inhab,init_method)
          else
           if(present(init_val)) then
            ierr=talsh_tensor_construct_num(tens_block,data_kind,tshape(1:trank),devid,extmem,inhab,init_val=init_val)
           else
            ierr=talsh_tensor_construct_num(tens_block,data_kind,tshape(1:trank),devid,extmem,inhab)
           endif
          endif
         else
          ierr=TALSH_INVALID_ARGS
         endif
         call tensor_block_destroy(ftens,errc)
         return
        end function talsh_tensor_construct_sym
!-----------------------------------------------------------------------------------------------------------------------------------
        function talsh_tensor_construct_shp(tens_block,data_kind,tens_shape,dev_id,ext_mem,in_hab,init_method,init_val) result(ierr)
         implicit none
         integer(C_INT):: ierr                                !out: error code (0:success)
         type(talsh_tens_t), intent(inout):: tens_block       !inout: constructed tensor block (must be empty on entrance)
         integer(C_INT), intent(in):: data_kind               !in: data kind: {R4,R8,C4,C8,NO_TYPE}
         type(talsh_tens_shape_t), intent(in):: tens_shape    !in: tensor shape
         integer(C_INT), intent(in), optional:: dev_id        !in: flat device ID on which the tensor block will reside
         type(C_PTR), intent(in), optional:: ext_mem          !in: pointer to externally provided memory for tensor elements
         integer(C_INT), intent(in), optional:: in_hab        !in: if >=0, <ext_mem> points to the HAB entry #<in_hab>
         procedure(talsh_tens_init_i), optional:: init_method !in: user-defined initialization method (<init_val> must be absent)
         complex(8), intent(in), optional:: init_val          !in: initialization value (will be typecast to <data_kind>, defaults to 0)
         type(C_PTR):: extmem
         integer(C_INT):: devid,inhab,trank
         integer(C_INT), pointer:: tshape(:)
         integer:: errc

         trank=tens_shape%num_dim
         if(trank.ge.0.and.trank.le.MAX_TENSOR_RANK) then
          call c_f_pointer(tens_shape%dims,tshape,shape=(/trank/))
          if(present(dev_id)) then; devid=dev_id; else; devid=talshFlatDevId(DEV_HOST,0); endif
          if(present(ext_mem)) then; extmem=ext_mem; else; extmem=C_NULL_PTR; endif
          if(present(in_hab)) then; inhab=in_hab; else; inhab=-1; endif
          if(present(init_method)) then
           ierr=talsh_tensor_construct_num(tens_block,data_kind,tshape(1:trank),devid,extmem,inhab,init_method)
          else
           if(present(init_val)) then
            ierr=talsh_tensor_construct_num(tens_block,data_kind,tshape(1:trank),devid,extmem,inhab,init_val=init_val)
           else
            ierr=talsh_tensor_construct_num(tens_block,data_kind,tshape(1:trank),devid,extmem,inhab)
           endif
          endif
         else
          ierr=TALSH_INVALID_ARGS
         endif
         return
        end function talsh_tensor_construct_shp
!--------------------------------------------------------------
        function talsh_tensor_destruct(tens_block) result(ierr)
         implicit none
         integer(C_INT):: ierr
         type(talsh_tens_t), intent(inout):: tens_block !inout: tensor block (will become empty on exit)
         ierr=talshTensorDestruct(tens_block)
         return
        end function talsh_tensor_destruct
!----------------------------------------------------------
        function talsh_tensor_rank(tens_block) result(rank)
         implicit none
         integer(C_INT):: rank                       !out: tensor block rank (number of dimensions)
         type(talsh_tens_t), intent(in):: tens_block !in: tensor block
         rank=talshTensorRank(tens_block)
         return
        end function talsh_tensor_rank
!-----------------------------------------------------------
        function talsh_tensor_volume(tens_block) result(vol)
         implicit none
         integer(C_SIZE_T):: vol                     !out: number of elements in the tensor block (negative on error)
         type(talsh_tens_t), intent(in):: tens_block !in: tensor block
         vol=talshTensorVolume(tens_block)
         return
        end function talsh_tensor_volume
!------------------------------------------------------------------------------------
        function talsh_tensor_dimensions(tens_block,tens_rank,tens_dims) result(ierr)
         implicit none
         integer(C_INT):: ierr                         !out: error code (0:success)
         type(talsh_tens_t), intent(in):: tens_block   !in: tensor block
         integer(INTD), intent(out):: tens_rank        !out: tensor rank
         integer(INTD), intent(inout):: tens_dims(1:*) !out: tensor dimension extents
         type(talsh_tens_shape_t):: tens_shape
         integer(INTD), pointer:: tdims(:)
         ierr=talshTensorShape(tens_block,tens_shape)
         if(ierr.eq.TALSH_SUCCESS) then
          tens_rank=tens_shape%num_dim
          if(tens_rank.gt.0) then
           call c_f_pointer(tens_shape%dims,tdims,(/tens_rank/))
           tens_dims(1:tens_rank)=tdims(1:tens_rank)
          endif
         endif
         return
        end function talsh_tensor_dimensions
!----------------------------------------------------------------------
        function talsh_tensor_shape(tens_block,tens_shape) result(ierr)
         implicit none
         integer(C_INT):: ierr                                !out: error code (0:success)
         type(talsh_tens_t), intent(in):: tens_block          !in: tensor block
         type(talsh_tens_shape_t), intent(inout):: tens_shape !inout: tensor block shape (copy)
         ierr=talshTensorShape(tens_block,tens_shape)
         return
        end function talsh_tensor_shape
!-------------------------------------------------------------------------------------
        function talsh_tensor_data_kind(tens_block,num_images,data_kinds) result(ierr)
         implicit none
         integer(C_INT):: ierr                           !out: error code (0:success)
         type(talsh_tens_t), intent(in):: tens_block     !in: tensor block
         integer(C_INT), intent(out):: num_images        !out: number of tensor images
         integer(C_INT), intent(inout):: data_kinds(1:*) !out: data kind of each tensor image
         ierr=talshTensorDataKind(tens_block,num_images,data_kinds)
         return
        end function talsh_tensor_data_kind
!--------------------------------------------------------------------------------------------------------
        function talsh_tensor_presence(tens_block,ncopies,copies,data_kinds,dev_kind,dev_id) result(ierr)
         implicit none
         integer(C_INT):: ierr                           !out: error code (0:success)
         type(talsh_tens_t), intent(in):: tens_block     !in: tensor block
         integer(C_INT), intent(out):: ncopies           !out: number of found copies of the tensor block
         integer(C_INT), intent(inout):: copies(1:*)     !out: copies found (list of flat device id's)
         integer(C_INT), intent(inout):: data_kinds(1:*) !out: data kind of each copy
         integer(C_INT), intent(in), optional:: dev_kind !in: specific device kind of interest (defaults to All)
         integer(C_INT), intent(in), optional:: dev_id   !in: specific device of interest
         integer(C_INT):: devk,devnum

         if(present(dev_kind)) then; devk=dev_kind; else; devk=DEV_NULL; endif
         if(present(dev_id)) then; devnum=dev_id; else; devnum=-1; endif
         ierr=talshTensorPresence_(tens_block,ncopies,copies,data_kinds,devk,devnum)
         return
        end function talsh_tensor_presence
!------------------------------------------------------------------------------------------------------
        function talsh_tensor_get_body_access(tens_block,body_p,data_kind,dev_id,dev_kind) result(ierr)
         implicit none
         integer(C_INT):: ierr                           !out: error code (0:success)
         type(talsh_tens_t), intent(inout):: tens_block  !inout: tensor block
         type(C_PTR), intent(inout):: body_p             !out: pointer to the tensor body image
         integer(C_INT), intent(in):: data_kind          !in: requested data kind for the image
         integer(C_INT), intent(in):: dev_id             !in: requested device id for the image, either kind-specific or flat
         integer(C_INT), intent(in), optional:: dev_kind !in: requested device kind (if present <dev_id> is kind-specific, flat otherwise)
         integer(C_INT):: devk

         if(present(dev_kind)) then; devk=dev_kind; else; devk=DEV_NULL; endif
         ierr=talshTensorGetBodyAccess_(tens_block,body_p,data_kind,dev_id,devk)
         return
        end function talsh_tensor_get_body_access
!-------------------------------------------------------------------------------
        function talsh_tensor_get_scalar(tens_block,scalar_complex) result(ierr)
         implicit none
         integer(C_INT):: ierr                          !out: error code
         type(talsh_tens_t), intent(inout):: tens_block !in: tensor block (rank-0)
         complex(8), intent(out):: scalar_complex       !out: complex scalar value
         real(C_DOUBLE):: sreal,simag

         ierr=talshTensorGetScalar_(tens_block,sreal,simag)
         if(ierr.eq.TALSH_SUCCESS) scalar_complex=cmplx(sreal,simag,8)
         return
        end function talsh_tensor_get_scalar
!------------------------------------------------------------
        function talsh_task_destruct(talsh_task) result(ierr)
         implicit none
         integer(C_INT):: ierr                          !out: error code (0:success)
         type(talsh_task_t), intent(inout):: talsh_task !inout: TAL-SH task (clean on exit)
         ierr=talshTaskDestruct(talsh_task)
         return
        end function talsh_task_destruct
!---------------------------------------------------------------------
        function talsh_task_dev_id(talsh_task,dev_kind) result(dev_id)
         implicit none
         integer(C_INT):: dev_id                          !out: flat or kind-specific device id
         type(talsh_task_t), intent(inout):: talsh_task   !in: value-defined TAL-SH task
         integer(C_INT), intent(out), optional:: dev_kind !out: device kind (if present, <dev_id> is kind-specific, if absent <dev_id> is flat)
         integer(C_INT), target:: devk

         if(present(dev_kind)) then
          dev_id=talshTaskDevId_(talsh_task,c_loc(devk))
          dev_kind=devk
         else
          dev_id=talshTaskDevId_(talsh_task,C_NULL_PTR)
         endif
         return
        end function talsh_task_dev_id
!----------------------------------------------------------
        function talsh_task_status(talsh_task) result(stat)
         implicit none
         integer(C_INT):: stat                          !out: TAL-SH task status
         type(talsh_task_t), intent(inout):: talsh_task !inout: TAL-SH task
         stat=talshTaskStatus(talsh_task)
         return
        end function talsh_task_status
!-----------------------------------------------------------------------
        function talsh_task_complete(talsh_task,stats,ierr) result(done)
         implicit none
         integer(C_INT):: done                          !out: YEP if the task has completed, NOPE otherwise
         type(talsh_task_t), intent(inout):: talsh_task !inout: TAL-SH task
         integer(C_INT), intent(out):: stats            !out: TAL-SH task status
         integer(C_INT), intent(out):: ierr             !out: error code (0:success)
         done=talshTaskComplete(talsh_task,stats,ierr)
         return
        end function talsh_task_complete
!--------------------------------------------------------------
        function talsh_task_wait(talsh_task,stats) result(ierr)
         implicit none
         integer(C_INT):: ierr                          !out: error code (0:success)
         type(talsh_task_t), intent(inout):: talsh_task !inout: TAL-SH task
         integer(C_INT), intent(out):: stats            !out: TAL-SH task status
         ierr=talshTaskWait(talsh_task,stats)
         return
        end function talsh_task_wait
!-----------------------------------------------------------------------
        function talsh_tasks_wait(ntasks,talsh_tasks,stats) result(ierr)
         implicit none
         integer(C_INT):: ierr                                     !out: error code (0:success)
         integer(C_INT), intent(in):: ntasks                       !in: number of tasks to wait upon
         type(talsh_task_t), intent(inout):: talsh_tasks(1:ntasks) !inout: TAL-SH tasks
         integer(C_INT), intent(out):: stats(1:ntasks)             !out: TAL-SH task statuses
         ierr=talshTasksWait(ntasks,talsh_tasks,stats)
         return
        end function talsh_tasks_wait
!---------------------------------------------------------------------------------------
        function talsh_task_time(talsh_task,total,comput,input,output,mmul) result(ierr)
         implicit none
         integer(C_INT):: ierr                          !out: error code (0:success)
         type(talsh_task_t), intent(inout):: talsh_task !inout: TAL-SH task
         real(C_DOUBLE), intent(out):: total            !out: total execution time (sec)
         real(C_DOUBLE), intent(out), optional:: comput !out: time the computation took (sec)
         real(C_DOUBLE), intent(out), optional:: input  !out: time the ingoing data transfers took (sec)
         real(C_DOUBLE), intent(out), optional:: output !out: time the outgoing data transfers took (sec)
         real(C_DOUBLE), intent(out), optional:: mmul   !out: time the matrix multiplication took (sec)
         real(C_DOUBLE):: comp_tm,in_tm,out_tm,mmul_tm
         ierr=talshTaskTime_(talsh_task,total,comp_tm,in_tm,out_tm,mmul_tm)
         if(present(comput)) comput=comp_tm
         if(present(input)) input=in_tm
         if(present(output)) output=out_tm
         if(present(mmul)) mmul=mmul_tm
         return
        end function talsh_task_time
!--------------------------------------------------------------------------------------------------
        function talsh_tensor_place(tens,dev_id,dev_kind,dev_mem,copy_ctrl,talsh_task) result(ierr)
         implicit none
         integer(C_INT):: ierr                              !out: error code (0:success)
         type(talsh_tens_t), intent(inout):: tens           !inout: tensor block
         integer(C_INT), intent(in):: dev_id                !in: device id (flat or kind-specific)
         integer(C_INT), intent(in), optional:: dev_kind    !in: device kind (if present, <dev_id> is kind-specific)
         type(C_PTR), intent(in), optional:: dev_mem        !in: externally provided target device memory pointer
         integer(C_INT), intent(in), optional:: copy_ctrl   !in: copy control (COPY_X), defaults to COPY_M
         type(talsh_task_t), intent(inout), optional:: talsh_task !inout: TAL-SH task handle
         integer(C_INT):: dvk,coh,sts
         type(talsh_task_t):: tsk
         type(C_PTR):: dvm

         if(present(dev_kind)) then; dvk=dev_kind; else; dvk=DEV_NULL; endif
         if(present(dev_mem)) then; dvm=dev_mem; else; dvm=C_NULL_PTR; endif
         if(present(copy_ctrl)) then; coh=copy_ctrl; else; coh=COPY_M; endif
         if(present(talsh_task)) then
          ierr=talshTensorPlace_(tens,dev_id,dvk,dvm,coh,talsh_task)
         else
          ierr=talsh_task_clean(tsk)
          ierr=talshTensorPlace_(tens,dev_id,dvk,dvm,coh,tsk)
          if(ierr.eq.TALSH_SUCCESS) then
           ierr=talsh_task_wait(tsk,sts); if(sts.ne.TALSH_TASK_COMPLETED) ierr=TALSH_TASK_ERROR
          endif
          sts=talsh_task_destruct(tsk)
         endif
         return
        end function talsh_tensor_place
!-----------------------------------------------------------------------
        function talsh_tensor_discard(tens,dev_id,dev_kind) result(ierr)
         implicit none
         integer(C_INT):: ierr                           !out: error code (0:success)
         type(talsh_tens_t), intent(inout):: tens        !inout: tensor block
         integer(C_INT), intent(in):: dev_id             !in: device id (flat or kind-specific)
         integer(C_INT), intent(in), optional:: dev_kind !in: device kind (if present, <dev_id> is kind-specific)
         integer(C_INT):: dvk

         if(present(dev_kind)) then; dvk=dev_kind; else; dvk=DEV_NULL; endif
         ierr=talshTensorDiscard_(tens,dev_id,dvk)
         return
        end function talsh_tensor_discard
!-----------------------------------------------------------------------------
        function talsh_tensor_discard_other(tens,dev_id,dev_kind) result(ierr)
         implicit none
         integer(C_INT):: ierr                           !out: error code (0:success)
         type(talsh_tens_t), intent(inout):: tens        !inout: tensor block
         integer(C_INT), intent(in):: dev_id             !in: device id (flat or kind-specific)
         integer(C_INT), intent(in), optional:: dev_kind !in: device kind (if present, <dev_id> is kind-specific)
         integer(C_INT):: dvk

         if(present(dev_kind)) then; dvk=dev_kind; else; dvk=DEV_NULL; endif
         ierr=talshTensorDiscardOther_(tens,dev_id,dvk)
         return
        end function talsh_tensor_discard_other
!----------------------------------------------------------------------------------------------
        function talsh_tensor_init(dtens,val,dev_id,dev_kind,copy_ctrl,talsh_task) result(ierr)
         implicit none
         integer(C_INT):: ierr                            !out: error code (0:success)
         type(talsh_tens_t), intent(inout):: dtens        !inout: destination tensor block
         complex(8), intent(in), optional:: val           !in: initialization value, defaults to 0
         integer(C_INT), intent(in), optional:: dev_id    !in: device id (flat or kind-specific)
         integer(C_INT), intent(in), optional:: dev_kind  !in: device kind (if present, <dev_id> is kind-specific)
         integer(C_INT), intent(in), optional:: copy_ctrl !in: copy control (COPY_XXX), defaults to COPY_MT
         type(talsh_task_t), intent(inout), optional:: talsh_task !inout: TAL-SH task (must be clean)
         integer(C_INT):: coh_ctrl,devn,devk,sts
         real(C_DOUBLE):: val_real,val_imag
         type(talsh_task_t):: tsk

         ierr=TALSH_SUCCESS
         if(present(copy_ctrl)) then; coh_ctrl=copy_ctrl; else; coh_ctrl=COPY_M; endif
         if(present(val)) then; val_real=dble(val); val_imag=dimag(val); else; val_real=0d0; val_imag=0d0; endif
         if(present(dev_id)) then; devn=dev_id; else; devn=DEV_DEFAULT; endif
         if(present(dev_kind)) then; devk=dev_kind; else; devk=DEV_DEFAULT; endif
         if(present(talsh_task)) then
          ierr=talshTensorInit_(dtens,val_real,val_imag,devn,devk,coh_ctrl,talsh_task)
         else
          ierr=talsh_task_clean(tsk)
          ierr=talshTensorInit_(dtens,val_real,val_imag,devn,devk,coh_ctrl,tsk)
          if(ierr.eq.TALSH_SUCCESS) then
           ierr=talsh_task_wait(tsk,sts); if(sts.ne.TALSH_TASK_COMPLETED) ierr=TALSH_TASK_ERROR
          endif
          sts=talsh_task_destruct(tsk)
         endif
         return
        end function talsh_tensor_init
!---------------------------------------------------------------------------------------------------------
        function talsh_tensor_slice(dtens,ltens,offsets,dev_id,dev_kind,copy_ctrl,talsh_task) result(ierr)
         implicit none
         integer(C_INT):: ierr                            !out: error code (0:success)
         type(talsh_tens_t), intent(inout):: dtens        !inout: destination tensor block (tensor slice)
         type(talsh_tens_t), intent(inout):: ltens        !inout: left source tensor block
         integer(C_INT), intent(in):: offsets(1:*)        !in: base offsets of the slice (0-based)
         integer(C_INT), intent(in), optional:: dev_id    !in: device id (flat or kind-specific)
         integer(C_INT), intent(in), optional:: dev_kind  !in: device kind (if present, <dev_id> is kind-specific)
         integer(C_INT), intent(in), optional:: copy_ctrl !in: copy control (COPY_XXX), defaults to COPY_MT
         type(talsh_task_t), intent(inout), optional:: talsh_task !inout: TAL-SH task (must be clean)
         integer(C_INT):: coh_ctrl,devn,devk,sts
         type(talsh_task_t):: tsk

         ierr=TALSH_SUCCESS
         if(present(copy_ctrl)) then; coh_ctrl=copy_ctrl; else; coh_ctrl=COPY_MT; endif
         if(present(dev_id)) then; devn=dev_id; else; devn=DEV_DEFAULT; endif
         if(present(dev_kind)) then; devk=dev_kind; else; devk=DEV_DEFAULT; endif
         if(present(talsh_task)) then
          ierr=talshTensorSlice_(dtens,ltens,offsets,devn,devk,coh_ctrl,talsh_task)
         else
          ierr=talsh_task_clean(tsk)
          ierr=talshTensorSlice_(dtens,ltens,offsets,devn,devk,coh_ctrl,tsk)
          if(ierr.eq.TALSH_SUCCESS) then
           ierr=talsh_task_wait(tsk,sts); if(sts.ne.TALSH_TASK_COMPLETED) ierr=TALSH_TASK_ERROR
          endif
          sts=talsh_task_destruct(tsk)
         endif
         return
        end function talsh_tensor_slice
!----------------------------------------------------------------------------------------------------------
        function talsh_tensor_insert(dtens,ltens,offsets,dev_id,dev_kind,copy_ctrl,talsh_task) result(ierr)
         implicit none
         integer(C_INT):: ierr                            !out: error code (0:success)
         type(talsh_tens_t), intent(inout):: dtens        !inout: destination tensor block
         type(talsh_tens_t), intent(inout):: ltens        !inout: left source tensor block (tensor slice)
         integer(C_INT), intent(in):: offsets(1:*)        !in: base offsets of the slice (0-based)
         integer(C_INT), intent(in), optional:: dev_id    !in: device id (flat or kind-specific)
         integer(C_INT), intent(in), optional:: dev_kind  !in: device kind (if present, <dev_id> is kind-specific)
         integer(C_INT), intent(in), optional:: copy_ctrl !in: copy control (COPY_XXX), defaults to COPY_MT
         type(talsh_task_t), intent(inout), optional:: talsh_task !inout: TAL-SH task (must be clean)
         integer(C_INT):: coh_ctrl,devn,devk,sts
         type(talsh_task_t):: tsk

         ierr=TALSH_SUCCESS
         if(present(copy_ctrl)) then; coh_ctrl=copy_ctrl; else; coh_ctrl=COPY_MT; endif
         if(present(dev_id)) then; devn=dev_id; else; devn=DEV_DEFAULT; endif
         if(present(dev_kind)) then; devk=dev_kind; else; devk=DEV_DEFAULT; endif
         if(present(talsh_task)) then
          ierr=talshTensorInsert_(dtens,ltens,offsets,devn,devk,coh_ctrl,talsh_task)
         else
          ierr=talsh_task_clean(tsk)
          ierr=talshTensorInsert_(dtens,ltens,offsets,devn,devk,coh_ctrl,tsk)
          if(ierr.eq.TALSH_SUCCESS) then
           ierr=talsh_task_wait(tsk,sts); if(sts.ne.TALSH_TASK_COMPLETED) ierr=TALSH_TASK_ERROR
          endif
          sts=talsh_task_destruct(tsk)
         endif
         return
        end function talsh_tensor_insert
!-----------------------------------------------------------------------------------------------------------
        function talsh_tensor_add(cptrn,dtens,ltens,scale,dev_id,dev_kind,copy_ctrl,talsh_task) result(ierr)
         implicit none
         integer(C_INT):: ierr                            !out: error code (0:success)
         character(*), intent(in):: cptrn                 !in: symbolic addition pattern, e.g. "D(a,b,c,d)+=L(c,d,b,a)"
         type(talsh_tens_t), intent(inout):: dtens        !inout: destination tensor block
         type(talsh_tens_t), intent(inout):: ltens        !inout: left source tensor block
         complex(8), intent(in), optional:: scale         !in: scaling factor, defaults to 1
         integer(C_INT), intent(in), optional:: dev_id    !in: device id (flat or kind-specific)
         integer(C_INT), intent(in), optional:: dev_kind  !in: device kind (if present, <dev_id> is kind-specific)
         integer(C_INT), intent(in), optional:: copy_ctrl !in: copy control (COPY_XXX), defaults to COPY_MT
         type(talsh_task_t), intent(inout), optional:: talsh_task !inout: TAL-SH task (must be clean)
         character(C_CHAR):: contr_ptrn(1:1024) !contraction pattern as a C-string
         integer(C_INT):: coh_ctrl,devn,devk,sts
         integer:: l
         real(C_DOUBLE):: scale_real,scale_imag
         type(talsh_task_t):: tsk

         ierr=TALSH_SUCCESS; l=len_trim(cptrn)
         if(l.gt.0) then
          if(present(copy_ctrl)) then; coh_ctrl=copy_ctrl; else; coh_ctrl=COPY_MT; endif
          if(present(scale)) then; scale_real=dble(scale); scale_imag=dimag(scale); else; scale_real=1d0; scale_imag=0d0; endif
          if(present(dev_id)) then; devn=dev_id; else; devn=DEV_DEFAULT; endif
          if(present(dev_kind)) then; devk=dev_kind; else; devk=DEV_DEFAULT; endif
          call string2array(cptrn(1:l),contr_ptrn,l,ierr); l=l+1; contr_ptrn(l:l)=achar(0) !C-string
          if(ierr.eq.0) then
           if(present(talsh_task)) then
            ierr=talshTensorAdd_(contr_ptrn,dtens,ltens,scale_real,scale_imag,devn,devk,coh_ctrl,talsh_task)
           else
            ierr=talsh_task_clean(tsk)
            ierr=talshTensorAdd_(contr_ptrn,dtens,ltens,scale_real,scale_imag,devn,devk,coh_ctrl,tsk)
            if(ierr.eq.TALSH_SUCCESS) then
             ierr=talsh_task_wait(tsk,sts); if(sts.ne.TALSH_TASK_COMPLETED) ierr=TALSH_TASK_ERROR
            endif
            sts=talsh_task_destruct(tsk)
           endif
          else
           ierr=TALSH_INVALID_ARGS
          endif
         else
          ierr=TALSH_INVALID_ARGS
         endif
         return
        end function talsh_tensor_add
!------------------------------------------------------------------------------------
        function talsh_tensor_contract(cptrn,dtens,ltens,rtens,scale,dev_id,dev_kind,&
                                      &copy_ctrl,accumulative,talsh_task) result(ierr)
         implicit none
         integer(C_INT):: ierr                            !out: error code (0:success)
         character(*), intent(in):: cptrn                 !in: symbolic contraction pattern, e.g. "D(a,b,c,d)+=L(c,i,j,a)*R(b,j,d,i)"
         type(talsh_tens_t), intent(inout):: dtens        !inout: destination tensor block
         type(talsh_tens_t), intent(inout):: ltens        !inout: left source tensor block
         type(talsh_tens_t), intent(inout):: rtens        !inout: right source tensor block
         complex(8), intent(in), optional:: scale         !in: scaling factor, defaults to 1
         integer(C_INT), intent(in), optional:: dev_id    !in: device id (flat or kind-specific)
         integer(C_INT), intent(in), optional:: dev_kind  !in: device kind (if present, <dev_id> is kind-specific)
         integer(C_INT), intent(in), optional:: copy_ctrl !in: copy control (COPY_XXX), defaults to COPY_MTT
         logical, intent(in), optional:: accumulative     !in: accumulate (default) VS overwrite destination
         type(talsh_task_t), intent(inout), optional:: talsh_task !inout: TAL-SH task (must be clean)
         character(C_CHAR):: contr_ptrn(1:1024) !contraction pattern as a C-string
         integer(C_INT):: coh_ctrl,devn,devk,sts,accum
         integer:: l
         real(C_DOUBLE):: scale_real,scale_imag
         type(talsh_task_t):: tsk

         ierr=TALSH_SUCCESS; l=len_trim(cptrn)
         if(l.gt.0) then
          accum=YEP; if(present(accumulative)) then; if(.not.accumulative) accum=NOPE; endif
          if(present(copy_ctrl)) then; coh_ctrl=copy_ctrl; else; coh_ctrl=COPY_MTT; endif
          if(present(scale)) then; scale_real=dble(scale); scale_imag=dimag(scale); else; scale_real=1d0; scale_imag=0d0; endif
          if(present(dev_id)) then; devn=dev_id; else; devn=DEV_DEFAULT; endif
          if(present(dev_kind)) then; devk=dev_kind; else; devk=DEV_DEFAULT; endif
          call string2array(cptrn(1:l),contr_ptrn,l,ierr); l=l+1; contr_ptrn(l:l)=achar(0) !C-string
          if(ierr.eq.0) then
           if(present(talsh_task)) then
            ierr=talshTensorContract_(contr_ptrn,dtens,ltens,rtens,scale_real,scale_imag,devn,devk,coh_ctrl,accum,talsh_task)
           else
            ierr=talsh_task_clean(tsk)
            ierr=talshTensorContract_(contr_ptrn,dtens,ltens,rtens,scale_real,scale_imag,devn,devk,coh_ctrl,accum,tsk)
            if(ierr.eq.TALSH_SUCCESS) then
             ierr=talsh_task_wait(tsk,sts); if(sts.ne.TALSH_TASK_COMPLETED) ierr=TALSH_TASK_ERROR
            endif
            sts=talsh_task_destruct(tsk)
           endif
          else
           ierr=TALSH_INVALID_ARGS
          endif
         else
          ierr=TALSH_INVALID_ARGS
         endif
         return
        end function talsh_tensor_contract
![CP-TAL]=====================================================================================================================
        integer(C_INT) function cpu_tensor_block_init(dtens_p,val_real,val_imag,arg_conj) bind(c,name='cpu_tensor_block_init')
         implicit none
         type(C_PTR), value:: dtens_p               !inout: destination tensor argument
         real(C_DOUBLE), value:: val_real           !in: scaling prefactor (real part)
         real(C_DOUBLE), value:: val_imag           !in: scaling prefactor (imaginary part)
         integer(C_INT), value:: arg_conj           !in: argument complex conjugation bits (0:D)
         type(tensor_block_t), pointer:: dtp
         character(2):: dtk
         complex(8):: val
         integer:: ierr

         cpu_tensor_block_init=0
         if(c_associated(dtens_p)) then
          call c_f_pointer(dtens_p,dtp)
          if(associated(dtp)) then
           dtk=tensor_master_data_kind(dtp,ierr)
           if(ierr.eq.0) then
            val=cmplx(val_real,val_imag,8); if(mod(arg_conj,2_C_INT).ne.0) val=conjg(val)
            call tensor_block_init(dtk,dtp,ierr,val_c8=val)
           endif
           cpu_tensor_block_init=ierr
          else
           cpu_tensor_block_init=-2
          endif
         else
          cpu_tensor_block_init=-1
         endif
         return
        end function cpu_tensor_block_init
!--------------------------------------------------------------------------------------------------------------------
        integer(C_INT) function cpu_tensor_block_slice(ltens_p,dtens_p,offsets) bind(c,name='cpu_tensor_block_slice')
         implicit none
         type(C_PTR), value:: ltens_p            !in: left tensor argument (tensor)
         type(C_PTR), value:: dtens_p            !inout: destination tensor argument (tensor slice)
         integer(C_INT), intent(in):: offsets(*) !in: slice base offsets (each dimension numeration starts from 0)
         type(tensor_block_t), pointer:: dtp,ltp
         integer:: ierr

         cpu_tensor_block_slice=0
         if(c_associated(dtens_p).and.c_associated(ltens_p)) then
          call c_f_pointer(dtens_p,dtp); call c_f_pointer(ltens_p,ltp)
          if(associated(dtp).and.associated(ltp)) then
           call tensor_block_slice(ltp,dtp,offsets,ierr)
           cpu_tensor_block_slice=ierr
          else
           cpu_tensor_block_slice=-2
          endif
         else
          cpu_tensor_block_slice=-1
         endif
         return
        end function cpu_tensor_block_slice
!----------------------------------------------------------------------------------------------------------------------
        integer(C_INT) function cpu_tensor_block_insert(ltens_p,dtens_p,offsets) bind(c,name='cpu_tensor_block_insert')
         implicit none
         type(C_PTR), value:: ltens_p            !in: left tensor argument (tensor slice)
         type(C_PTR), value:: dtens_p            !inout: destination tensor argument (tensor)
         integer(C_INT), intent(in):: offsets(*) !in: slice base offsets (each dimension numeration starts from 0)
         type(tensor_block_t), pointer:: dtp,ltp
         integer:: ierr

         cpu_tensor_block_insert=0
         if(c_associated(dtens_p).and.c_associated(ltens_p)) then
          call c_f_pointer(dtens_p,dtp); call c_f_pointer(ltens_p,ltp)
          if(associated(dtp).and.associated(ltp)) then
           call tensor_block_insert(dtp,ltp,offsets,ierr)
           cpu_tensor_block_insert=ierr
          else
           cpu_tensor_block_insert=-2
          endif
         else
          cpu_tensor_block_insert=-1
         endif
         return
        end function cpu_tensor_block_insert
!-------------------------------------------------------------------------------------------------------------------------------
        integer(C_INT) function cpu_tensor_block_copy(permutation,ltens_p,dtens_p,arg_conj) bind(c,name='cpu_tensor_block_copy')
         implicit none
         integer(C_INT), intent(in):: permutation(1:*) !in: sign-free O2N tensor dimension permutation (1-based)
         type(C_PTR), value:: ltens_p                  !in: left tensor argument
         type(C_PTR), value:: dtens_p                  !inout: destination tensor argument
         integer(C_INT), value:: arg_conj              !in: argument complex conjugation bits (0:D,1:L)
         type(tensor_block_t), pointer:: dtp,ltp
         integer:: transp(0:MAX_TENSOR_RANK),conj_bits,n,ierr

         cpu_tensor_block_copy=0
         if(c_associated(dtens_p).and.c_associated(ltens_p)) then
          call c_f_pointer(dtens_p,dtp); call c_f_pointer(ltens_p,ltp)
          if(associated(dtp).and.associated(ltp)) then
           n=dtp%tensor_shape%num_dim
           if(n.ge.0) then
            transp(0)=+1; if(n.gt.0) transp(1:n)=permutation(1:n)
            conj_bits=arg_conj
            call tensor_block_copy(ltp,dtp,ierr,transp,conj_bits)
            cpu_tensor_block_copy=ierr
           else
            cpu_tensor_block_copy=-3
           endif
          else
           cpu_tensor_block_copy=-2
          endif
         else
          cpu_tensor_block_copy=-1
         endif
         return
        end function cpu_tensor_block_copy
!---------------------------------------------------------------------------------------------------------------
        integer(C_INT) function cpu_tensor_block_add(contr_ptrn,ltens_p,dtens_p,scale_real,scale_imag,arg_conj)&
                                                    &bind(c,name='cpu_tensor_block_add')
         implicit none
         integer(C_INT), intent(in):: contr_ptrn(*) !in: digital tensor addition pattern
         type(C_PTR), value:: ltens_p               !in: left tensor argument
         type(C_PTR), value:: dtens_p               !inout: destination tensor argument
         real(C_DOUBLE), value:: scale_real         !in: scaling prefactor (real part)
         real(C_DOUBLE), value:: scale_imag         !in: scaling prefactor (imaginary part)
         integer(C_INT), value:: arg_conj           !in: argument complex conjugation bits (0:D,1:L)
         type(tensor_block_t), pointer:: dtp,ltp
         complex(8):: scale_fac
         integer:: i,conj_bits,ierr
         logical:: permute

         cpu_tensor_block_add=0; conj_bits=arg_conj
         if(c_associated(dtens_p).and.c_associated(ltens_p)) then
          call c_f_pointer(dtens_p,dtp); call c_f_pointer(ltens_p,ltp)
          if(associated(dtp).and.associated(ltp)) then
           permute=.FALSE.
           do i=1,dtp%tensor_shape%num_dim
            if(contr_ptrn(i).ne.i) then; permute=.TRUE.; exit; endif
           enddo
           if(permute) then !`Add this feature
            write(CONS_OUT,'("#FATAL(talshf:cpu_tensor_block_add): addition with permutation is not implemented for CPU target!")')
            stop
           endif
           if(dabs(scale_real-1d0).gt.ZERO_THRESH.or.dabs(scale_imag-0d0).gt.ZERO_THRESH) then
            scale_fac=cmplx(scale_real,scale_imag,8)
            call tensor_block_add(dtp,ltp,ierr,scale_fac,conj_bits)
           else
            call tensor_block_add(dtp,ltp,ierr,arg_conj=conj_bits)
           endif
           cpu_tensor_block_add=ierr
          else
           cpu_tensor_block_add=-2
          endif
         else
          cpu_tensor_block_add=-1
         endif
         return
        end function cpu_tensor_block_add
!-----------------------------------------------------------------------------------------------------
        integer(C_INT) function cpu_tensor_block_contract(contr_ptrn,ltens_p,rtens_p,dtens_p,&
                                                         &scale_real,scale_imag,arg_conj,accumulative)&
                                                         &bind(c,name='cpu_tensor_block_contract')
         implicit none
         integer(C_INT), intent(in):: contr_ptrn(*) !in: digital tensor contraction pattern
         type(C_PTR), value:: ltens_p               !in: left tensor argument
         type(C_PTR), value:: rtens_p               !in: right tensor argument
         type(C_PTR), value:: dtens_p               !inout: destination tensor argument
         real(C_DOUBLE), value:: scale_real         !in: scaling prefactor (real part)
         real(C_DOUBLE), value:: scale_imag         !in: scaling prefactor (imaginary part)
         integer(C_INT), value:: arg_conj           !in: argument complex conjugation bits (0:D,1:L,2:R)
         integer(C_INT), value:: accumulative       !in: whether or not tensor contraction is accumulative [YEP|NOPE]
         type(tensor_block_t), pointer:: dtp,ltp,rtp
         integer:: conj_bits,ierr

         cpu_tensor_block_contract=0; conj_bits=arg_conj
         if(c_associated(dtens_p).and.c_associated(ltens_p).and.c_associated(rtens_p)) then
          call c_f_pointer(dtens_p,dtp); call c_f_pointer(ltens_p,ltp); call c_f_pointer(rtens_p,rtp)
          if(associated(dtp).and.associated(ltp).and.associated(rtp)) then
           call tensor_block_contract(contr_ptrn,ltp,rtp,dtp,ierr,alpha=cmplx(scale_real,scale_imag,8),&
                                     &arg_conj=conj_bits,accumulative=(accumulative.ne.NOPE))
           cpu_tensor_block_contract=ierr
          else
           cpu_tensor_block_contract=-2
          endif
         else
          cpu_tensor_block_contract=-1
         endif
         return
        end function cpu_tensor_block_contract

       end module talsh
