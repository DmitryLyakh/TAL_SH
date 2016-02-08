!ExaTensor::TAL-SH: Device-unified user-level API:
!REVISION: 2016/02/08
!Copyright (C) 2015 Dmitry I. Lyakh (email: quant4me@gmail.com)
!Copyright (C) 2015 Oak Ridge National Laboratory (UT-Battelle)
!LICENSE: GPLv2

!This source file is free software; you can redistribute it and/or
!modify it under the terms of the GNU General Public License
!as published by the Free Software Foundation; either version 2
!of the License, or (at your option) any later version.

!This program is distributed in the hope that it will be useful,
!but WITHOUT ANY WARRANTY; without even the implied warranty of
!MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!GNU General Public License for more details.

!You should have received a copy of the GNU General Public License
!along with this program; if not, write to the Free Software
!Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
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

        end interface
!VISIBILITY:
 !TAL-SH device control API:
        public talsh_init
        public talsh_shutdown
!        public talsh_flat_dev_id
!        public talsh_kind_dev_id
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
         integer(C_INT):: ierr                                      !out: error code (0:success)
         integer(C_SIZE_T), intent(inout), optional:: host_buf_size !inout: desired size in bytes of the Host Argument Buffer.
                                                                    !       It will be replaced by the actual size.
         integer(C_INT), intent(out), optional:: host_arg_max       !out: max number of arguments the HAB can contain
         integer(C_INT), intent(in), optional:: gpu_list(1:)        !in: list of NVidia GPU's to use
         integer(C_INT), intent(in), optional:: mic_list(1:)        !in: list of Intel Xeon Phi's to use
         integer(C_INT), intent(in), optional:: amd_list(1:)        !in: list of AMD GPU's to use

         ierr=TALSH_SUCCESS; host_arg_max=0;
         return
        end function talsh_init
!---------------------------------------------
        function talsh_shutdown() result(ierr)
         integer(C_INT):: ierr !out: error code (0:success)

         ierr=TALSH_SUCCESS
         return
        end function talsh_shutdown

       end module talsh
