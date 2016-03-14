!ExaTensor::TAL-SH: Device-unified user-level API:
!REVISION: 2016/02/12

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
!--------------------------------------------------------------------------------
       module talsh
        use tensor_algebra_cpu_phi !device-specific tensor algebra API + basic
        implicit none
        private
!PARAMETERS:
 !Generic:
        integer(INTD), private:: CONS_OUT=6 !default output device for this module
        logical, private:: VERBOSE=.true.   !verbosity for errors
        logical, private:: DEBUG=.true.     !debugging mode for this module
 !Errors:
        integer(C_INT), parameter, public:: TALSH_SUCCESS=0             !success
        integer(C_INT), parameter, public:: TALSH_FAILURE=-666          !generic failure
 !Host argument buffer:
        integer(C_SIZE_T), parameter, private:: HAB_SIZE_DEFAULT=1024*1024 !default size of the Host argument buffer
!DERIVED TYPES:

!GLOBALS:

!INTERFACES:
        interface
 !TAL-SH device control C API:
  !Initialize TAL-SH:
         integer(C_INT) function talshInit(host_buf_size,host_arg_max,ngpus,gpu_list,nmics,mic_list,namds,amd_list) &
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

        end interface
!VISIBILITY:
 !TAL-SH device control API:
        public talsh_init
        public talsh_shutdown
        public talsh_flat_dev_id
        public talsh_kind_dev_id
!        public talsh_device_state
!        public talsh_device_busy_least
!        public talsh_stats
 !TAL-SH tensor block API:
!        public talsh_tensor_construct
!        public talsh_tensor_destroy
!        public talsh_tensor_volume
!        public talsh_tensor_datatype
!        public talsh_tensor_shape
!        public talsh_tensor_presence
 !TAL-SH task API:
!        public talsh_task_clean
!        public talsh_task_dev_id
!        public talsh_task_status
!        public talsh_task_completed
!        public talsh_task_wait
!        public talsh_tasks_wait
!        public talsh_task_time
 !TAL-SH tensor operations:
!        public talsh_tensor_place
!        public talsh_tensor_discard
!        public talsh_tensor_init
!        public talsh_tensor_scale
!        public talsh_tensor_norm1
!        public talsh_tensor_norm2
!        public talsh_tensor_copy
!        public talsh_tensor_add
!        public talsh_tensor_contract

       contains
!Fortran API definitions:
 !TAL-SH device control API:
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
         integer(C_INT):: res                  !out: Flat device Id [0..DEV_MAX-1]; Failure: DEV_MAX
         integer(C_INT), intent(in):: dev_kind !in: device kind
         integer(C_INT), intent(in):: dev_num  !in: device Id within its kind (0..MAX)
         res=talshFlatDevId(dev_kind,dev_num)
         return
        end function talsh_flat_dev_id
!--------------------------------------------------------------
        function talsh_kind_dev_id(dev_id,dev_kind) result(res)
         implicit none
         integer(C_INT):: res                   !out: kind-specific device Id [0..]; Failure: DEV_NULL
         integer(C_INT), intent(in):: dev_id    !in: flat device Id
         integer(C_INT), intent(out):: dev_kind !out: device kind
         res=talshKindDevId(dev_id,dev_kind)
         return
        end function talsh_kind_dev_id

       end module talsh
