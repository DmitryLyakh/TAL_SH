!Timing services (threadsafe).
!AUTHOR: Dmitry I. Lyakh (Liakh): quant4me@gmail.com
!REVISION: 2018/07/24

!Copyright (C) 2014-2022 Dmitry I. Lyakh (Liakh)
!Copyright (C) 2014-2022 Oak Ridge National Laboratory (UT-Battelle)

!LICENSE: BSD 3-Clause

!NOTES:
! # A timer handle (reference to an internal timer object) is thread-private:
!   Only the thread which acquired the timer is allowed to deal with it later.

!PREPROCESSOR:
! # -D NO_OMP: Disable OpenMP (switch to Fortran cpu_time);
! # -D USE_GNU: Switch to the GNU Fortran timing (secnds);
! # -D NO_PHI: Ignore Intel MIC;

       module timers
#ifndef NO_OMP
        use omp_lib
#endif
        implicit none
        private
!PARAMETERS:
        integer, parameter, private:: MAX_TIMERS=8192
        integer, parameter, public:: TIMERS_SUCCESS=0
        integer, parameter, public:: TIMERS_ERR_INVALID_ARG=1
        integer, parameter, public:: TIMERS_ERR_NO_TIMERS_LEFT=2
        integer, parameter, public:: TIMERS_ERR_TIMER_NULL=3
!TYPES:
        type, private:: timer_t
         real(8), private:: beg_time      !time the timer started (sec)
         real(8), private:: time_interval !time the timer is set for (sec)
        end type timer_t
!GLOBAL DATA:
        integer, private:: j_
        type(timer_t), private:: timer(0:MAX_TIMERS-1)=(/(timer_t(-1d0,-1d0),j_=0,MAX_TIMERS-1)/)
        integer, private:: handle_stack(0:MAX_TIMERS-1)=(/(j_,j_=0,MAX_TIMERS-1)/)
        integer, private:: handle_sp=0
        real(8), private:: timer_tick=-1d0 !uninitilized
!FUNCTION VISIBILITY:
        public timer_start
        public timer_expired
        public timer_reset
        public timer_destroy
        public timer_tick_sec
        public thread_wtime
        public accu_time
        public time_sys_sec
        public time_high_sec

!EXTERNAL INTERFACES:
        interface

         function accu_time() bind(C,name='accu_time')
          use, intrinsic:: ISO_C_BINDING, only: C_DOUBLE
          real(C_DOUBLE):: accu_time
         end function accu_time

         function time_sys_sec() bind(C,name='time_sys_sec')
          use, intrinsic:: ISO_C_BINDING, only: C_DOUBLE
          real(C_DOUBLE):: time_sys_sec
         end function time_sys_sec

         function time_high_sec() bind(C,name='time_high_sec')
          use, intrinsic:: ISO_C_BINDING, only: C_DOUBLE
          real(C_DOUBLE):: time_high_sec
         end function time_high_sec

        end interface

       contains
!---------------------------------------------------------
        integer function timer_start(time_handle,time_set)
!This function sets up a timer limited to <time_set> seconds
!and returns its handle in <time_handle>.
        implicit none
        integer, intent(out):: time_handle !out: timer handle
        real(8), intent(in):: time_set     !in: requested time in seconds
        real(8):: val

        timer_start=TIMERS_SUCCESS; time_handle=-1
        if(time_set.ge.0d0) then
!$OMP CRITICAL (TIMERS_REGION)
         if(handle_sp.ge.0.and.handle_sp.lt.MAX_TIMERS) then
          time_handle=handle_stack(handle_sp); handle_sp=handle_sp+1
         else
          timer_start=TIMERS_ERR_NO_TIMERS_LEFT
         endif
!$OMP END CRITICAL (TIMERS_REGION)
         if(timer_start.eq.TIMERS_SUCCESS) then
#ifndef NO_OMP
          val=omp_get_wtime()
#else
          call cpu_time(val)
#endif
          timer(time_handle)=timer_t(val,time_set)
         endif
        else
         timer_start=TIMERS_ERR_INVALID_ARG
        endif
        return
        end function timer_start
!-------------------------------------------------------------------------
        logical function timer_expired(time_handle,ierr,destroy,curr_time)
!This function tests whether a given timer has expired.
!If <destroy> is present and TRUE, timer handle will be destroyed if the timer has expired.
        implicit none
        integer, intent(inout):: time_handle       !inout: timer handle
        integer, intent(inout):: ierr              !out: error code (0:success)
        logical, intent(in), optional:: destroy    !in: request to destroy the timer if it has expired
        real(8), intent(out), optional:: curr_time !out: current timer value in seconds
        real(8):: tm,ct

        timer_expired=.FALSE.; ct=0d0
        if(time_handle.ge.0.and.time_handle.lt.MAX_TIMERS) then !valid range
         if(timer(time_handle)%time_interval.ge.0d0) then !valid handle
          ierr=TIMERS_SUCCESS
#ifndef NO_OMP
          tm=omp_get_wtime()
#else
          call cpu_time(tm)
#endif
          ct=tm-timer(time_handle)%beg_time
          if(ct.ge.timer(time_handle)%time_interval) timer_expired=.TRUE.
          if(timer_expired.and.present(destroy)) then
           if(destroy) then
!$OMP CRITICAL (TIMERS_REGION)
            timer(time_handle)=timer_t(-1d0,-1d0)
            handle_sp=handle_sp-1; handle_stack(handle_sp)=time_handle
!$OMP END CRITICAL (TIMERS_REGION)
           endif
          endif
         else
          ierr=TIMERS_ERR_TIMER_NULL
         endif
        else
         ierr=TIMERS_ERR_INVALID_ARG
        endif
        if(present(curr_time)) curr_time=ct
        return
        end function timer_expired
!---------------------------------------------------------
        integer function timer_reset(time_handle,time_set)
!Resets an existing timer, regardless of its expiration status, to a new setting.
        implicit none
        integer, intent(inout):: time_handle     !inout: timer handle
        real(8), intent(in), optional:: time_set !in: requested time in seconds
        real(8):: val

        timer_reset=TIMERS_SUCCESS
        if(time_handle.ge.0.and.time_handle.lt.MAX_TIMERS) then !valid range
         if(timer(time_handle)%time_interval.ge.0d0) then !valid handle
#ifndef NO_OMP
          val=omp_get_wtime()
#else
          call cpu_time(val)
#endif
          timer(time_handle)%beg_time=val
          if(present(time_set)) then
           if(time_set.ge.0d0) then
            timer(time_handle)%time_interval=time_set
           else
            timer_reset=TIMERS_ERR_INVALID_ARG
           endif
          endif
!$OMP FLUSH
         else
          timer_reset=TIMERS_ERR_TIMER_NULL
         endif
        else
         timer_reset=TIMERS_ERR_INVALID_ARG
        endif
        return
        end function timer_reset
!--------------------------------------------------
        integer function timer_destroy(time_handle)
!This function explicitly frees a timer handle.
        implicit none
        integer, intent(in):: time_handle

        timer_destroy=TIMERS_SUCCESS
        if(time_handle.ge.0.and.time_handle.lt.MAX_TIMERS) then !valid range
         if(timer(time_handle)%time_interval.ge.0d0) then !valid handle
!$OMP CRITICAL (TIMERS_REGION)
          timer(time_handle)=timer_t(-1d0,-1d0)
          handle_sp=handle_sp-1; handle_stack(handle_sp)=time_handle
!$OMP END CRITICAL (TIMERS_REGION)
         else
          timer_destroy=TIMERS_ERR_TIMER_NULL
         endif
        else
         timer_destroy=TIMERS_ERR_INVALID_ARG
        endif
        return
        end function timer_destroy
!----------------------------------------
        real(8) function timer_tick_sec()
!This function returns the wall clock tick length in seconds.
        implicit none

#ifndef NO_OMP
!$OMP CRITICAL (TIMERS_REGION)
        if(timer_tick.le.0d0) timer_tick=omp_get_wtick()
!$OMP END CRITICAL (TIMERS_REGION)
#endif
        timer_tick_sec=timer_tick
        return
        end function timer_tick_sec
!-------------------------------------------
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: thread_wtime
#endif
        real(8) function thread_wtime(tbase)
!This function returns the current wall clock time in seconds;
!if <tbase> is present, since that moment.
        implicit none
        real(8), intent(in), optional:: tbase
        real(8):: tm

#ifndef NO_OMP
        thread_wtime=omp_get_wtime()
#else
#ifdef USE_GNU
        thread_wtime=real(secnds(0.),8)
#else
        call cpu_time(tm); thread_wtime=tm
#endif
#endif
        if(present(tbase)) thread_wtime=thread_wtime-tbase
        return
        end function thread_wtime

       end module timers
