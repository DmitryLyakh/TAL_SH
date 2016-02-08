!Tensor Algebra Library for Intel MIC built upon
!the Tensor Algebra Library for multi-core CPU.
!AUTHOR: Dmitry I. Lyakh (Liakh): quant4me@gmail.com
!REVISION: 2015/09/23
!NOTES:
! # This library is not thread-safe. All functions of this library are
!   supposed to be called by the same single thread (due to the way
!   the task handle management is implemented).
!PREPROCESSOR:
! -D NO_OMP: Disable OpenMP;
! -D NO_PHI: Ignore Intel MIC;
       module tensor_algebra_cpu_phi
        use tensor_algebra_cpu !certain procedures and globals are imported from this module
        implicit none
        public
#ifndef NO_OMP
#ifndef USE_OMP_MOD
        integer, external, private:: omp_get_max_threads,omp_get_num_threads,omp_get_thread_num
#endif
#endif

#ifndef NO_PHI
!PARAMETERS:
        integer, parameter, public:: PHI_ALLOC_FREE=0
        integer, parameter, public:: PHI_ALLOC_KEEP=1
        integer, parameter, public:: PHI_REUSE_FREE=2
        integer, parameter, public:: PHI_REUSE_KEEP=3
!DIR$ ATTRIBUTES OFFLOAD:mic:: PHI_ALLOC_FREE,PHI_ALLOC_KEEP,PHI_REUSE_FREE,PHI_REUSE_KEEP
!DIR$ ATTRIBUTES ALIGN:128:: PHI_ALLOC_FREE,PHI_ALLOC_KEEP,PHI_REUSE_FREE,PHI_REUSE_KEEP
        integer, parameter, private:: phi_max_tasks=1024
!DERIVED TYPES:

!DATA:
        integer, private:: j_,phi_task_ffe=1
        integer, private:: phi_task_handles(1:phi_max_tasks)=(/(j_,j_=1,phi_max_tasks)/)

!FUNCTIONS:
        private phi_get_task_handle
        private phi_free_task_handle
        public phi_init_task_handles
        public phi_data_ctrl_clean
        public phi_data_ctrl_set
        public phi_tensor_block_copy_

       contains
!--------------------------------------------------------
        integer function phi_get_task_handle(task_handle) !SERIAL: Host
!Obtains a free task handle.
        implicit none
        integer, intent(out):: task_handle
        if(phi_task_ffe.gt.0) then
         if(phi_task_ffe.le.phi_max_tasks) then
          task_handle=phi_task_handles(phi_task_ffe); phi_task_ffe=phi_task_ffe+1
          phi_get_task_handle=0
         else !no free handles left
          phi_get_task_handle=1
         endif
        else !invalid phi_task_ffe
         phi_get_task_handle=-1
        endif
        return
        end function phi_get_task_handle
!---------------------------------------------------------
        integer function phi_free_task_handle(task_handle) !SERIAL: Host
!Returns a task handle back to the free storage.
        implicit none
        integer, intent(inout):: task_handle
        if(task_handle.gt.0.and.task_handle.le.phi_max_tasks) then
         if(phi_task_ffe.gt.1.and.phi_task_ffe.le.phi_max_tasks+1) then
          phi_task_ffe=phi_task_ffe-1; phi_task_handles(phi_task_ffe)=task_handle
          task_handle=0; phi_free_task_handle=0
         else !invalid phi_task_ffe
          phi_free_task_handle=-1
         endif
        else !invalid task handle passed
         phi_free_task_handle=-2
        endif
        return
        end function phi_free_task_handle
!------------------------------------------
        subroutine phi_init_task_handles() !SERIAL: Host
!Reinitializes task handle manager.
        implicit none
        integer i
        do i=1,phi_max_tasks; phi_task_handles(i)=i; enddo
        phi_task_ffe=1
        return
        end subroutine phi_init_task_handles
!------------------------------------------------
        subroutine phi_data_ctrl_clean(data_ctrl) !SERIAL: Host
!Cleans data residence control flags.
        implicit none
        integer, intent(out):: data_ctrl
        data_ctrl=0
        return
        end subroutine phi_data_ctrl_clean
!-----------------------------------------------------------------
        integer function phi_data_ctrl_set(data_ctrl,arg_num,flag) !SERIAL:Host
!Sets data residence control flag <flag> for argument #<arg_num>.
        implicit none
        integer, intent(inout):: data_ctrl !data residence control flags
        integer, intent(in):: arg_num      !argument number: [0:max_tensor_operands-1]: 0 is always the destination tensor
        integer, intent(in):: flag         !flag to set: {PHI_ALLOC_FREE, PHI_ALLOC_KEEP, PHI_REUSE_FREE, PHI_REUSE_KEEP}
        phi_data_ctrl_set=0
        if(arg_num.ge.0.and.arg_num.lt.max_tensor_operands) then
         select case(flag)
         case(PHI_ALLOC_FREE)
          data_ctrl=ibclr(data_ctrl,arg_num*2+1)
          data_ctrl=ibclr(data_ctrl,arg_num*2)
         case(PHI_ALLOC_KEEP)
          data_ctrl=ibclr(data_ctrl,arg_num*2+1)
          data_ctrl=ibset(data_ctrl,arg_num*2)
         case(PHI_REUSE_FREE)
          data_ctrl=ibset(data_ctrl,arg_num*2+1)
          data_ctrl=ibclr(data_ctrl,arg_num*2)
         case(PHI_REUSE_KEEP)
          data_ctrl=ibset(data_ctrl,arg_num*2+1)
          data_ctrl=ibset(data_ctrl,arg_num*2)
         case default
          phi_data_ctrl_set=-1
         end select
        else
         phi_data_ctrl_set=-2
        endif
        return
        end function phi_data_ctrl_set
!----------------------------------------------------------------------------------------------------
        integer function phi_tensor_block_copy_(mic_num,data_ctrl,tens_in,tens_out,phi_handle,transp) !Paralell: Host+MIC
!This function schedules an asynchronous tensor block copy/transpose on MIC device #<mic_num>.
        implicit none
        integer, intent(in):: mic_num                  !MIC device number: [0..max]
        integer, intent(in):: data_ctrl                !data reuse flags for all tensor arguments
        type(tensor_block_t), intent(in):: tens_in     !input tensor block (arg#1)
        type(tensor_block_t), intent(inout):: tens_out !output tensor block (arg#0)
        integer, intent(out):: phi_handle              !task handle (for further queries)
        integer, intent(in), optional:: transp(0:*)    !permutation (0:sign; 1..:permutation_O2N)
        integer i,j,k,l,m,n

        phi_tensor_block_copy_=0; n=tens_in%tensor_shape%num_dim
        
        i=phi_get_task_handle(phi_handle); if(i.ne.0) then; phi_tensor_block_copy_=1; return; endif
        
        return
        end function phi_tensor_block_copy_
#endif
       end module tensor_algebra_cpu_phi
