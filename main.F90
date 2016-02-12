!TALSH::Fortran API testing.
        program main
        use, intrinsic:: ISO_C_BINDING
        implicit none

        interface
         subroutine test_talsh_c(ierr) bind(c)
          import
          integer(C_INT), intent(out):: ierr
         end subroutine test_talsh_c
        end interface

        integer(C_INT):: ierr

!Test C API interface:
        write(*,'("Testing TALSH C/C++ API ...")')
        call test_talsh_c(ierr)
        write(*,'("Done: Status ",i5)') ierr
        if(ierr.ne.0) stop
!Test Fortran API interface:
        write(*,'("Testing TALSH Fortran API ...")')
        call test_talsh_f(ierr)
        write(*,'("Done: Status ",i5)') ierr
        if(ierr.ne.0) stop
        stop
        end program main
!------------------------------------
        subroutine test_talsh_f(ierr)
        use, intrinsic:: ISO_C_BINDING
        use talsh
        implicit none
        integer(C_INT):: ierr,host_arg_max
        integer(C_SIZE_T):: host_buf_size

        ierr=0
!Init TALSH:
        write(*,'(1x,"Initializing TALSH ... ")',ADVANCE='NO')
        host_buf_size=1024*1024*1024
        ierr=talsh_init(host_buf_size,host_arg_max,gpu_list=(/0/))
        write(*,'("Status ",i11,": Size (Bytes) = ",i13,": Max Args in HAB = ",i7)') ierr,host_buf_size,host_arg_max
        if(ierr.ne.TALSH_SUCCESS) then; ierr=1; return; endif
!Shutdown TALSH:
        write(*,'(1x,"Shutting down TALSH ... ")',ADVANCE='NO')
        ierr=talsh_shutdown()
        write(*,'("Status ",i11)') ierr
        if(ierr.ne.TALSH_SUCCESS) then; ierr=1; return; endif
        return
        end subroutine test_talsh_f
