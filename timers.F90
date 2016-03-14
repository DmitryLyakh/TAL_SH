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

       module timers
!Timing services (threadsafe).
!AUTHOR: Dmitry I. Lyakh (Liakh): quant4me@gmail.com
!REVISION: 2015/07/30
!PUBLIC FUNCTIONS:
! # integer timer_start(real8:time_set, integer:time_handle);
! # logical time_is_off(integer:time_handle, integer:ierr[, logical:destroy]);
! # integer timer_destroy(integer:time_handle);
! # real8 timer_tick_sec();
! # real8 thread_wtime([real8:tbase]);
!PREPROCESSOR:
! # -D NO_OMP: Disable OpenMP (switch to Fortran cpu_time);
! # -D USE_OMP_MOD: Use OpenMP Fortran module;
! # -D USE_GNU: Switch to the GNU Fortran timing (secnds);
! # -D NO_PHI: Ignore Intel MIC;
#ifndef NO_OMP
#ifdef USE_OMP_MOD
        use omp_lib
        implicit none
        private
#else
        implicit none
        private
        real(8), external, private:: omp_get_wtime,omp_get_wtick
#endif
#endif
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
!DATA:
        integer, private:: j_
        type(timer_t), private:: timer(0:MAX_TIMERS-1)=(/(timer_t(-1d0,-1d0),j_=0,MAX_TIMERS-1)/)
        integer, private:: handle_stack(0:MAX_TIMERS-1)=(/(j_,j_=0,MAX_TIMERS-1)/)
        integer, private:: handle_sp=0
        real(8), private:: timer_tick=-1d0 !uninitilized
!FUNCTION VISIBILITY:
        public timer_start
        public time_is_off
        public timer_destroy
        public timer_tick_sec
        public thread_wtime
        public accu_time
!EXTERNAL INTERFACES:
        interface
         function accu_time() bind(C,name='accu_time')
          use, intrinsic:: ISO_C_BINDING, only: C_DOUBLE
          real(C_DOUBLE):: accu_time
         end function accu_time
        end interface

       contains
!---------------------------------------------------------
        integer function timer_start(time_set,time_handle)
!This function sets up a timer limited to <time_set> seconds and returns its handle in <time_handle>.
        implicit none
        real(8), intent(in):: time_set     !in: requested time in seconds
        integer, intent(out):: time_handle !out: timer handle
        real(8):: val
        timer_start=TIMERS_SUCCESS
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
!-------------------------------------------------------------
        logical function time_is_off(time_handle,ierr,destroy)
!This function tests whether a given timer had expired.
!If <destroy> is present and .true., timer handle will be destroyed if the timer is expired.
        implicit none
        integer, intent(inout):: time_handle    !inout: timer handle
        integer, intent(inout):: ierr           !out: error code (0:success)
        logical, intent(in), optional:: destroy !in: request to destroy the timer if expired
        real(8):: tm
        time_is_off=.false.
        if(time_handle.ge.0.and.time_handle.lt.MAX_TIMERS) then !valid range
         if(timer(time_handle)%time_interval.ge.0d0) then !valid handle
          ierr=0
#ifndef NO_OMP
          tm=omp_get_wtime()
#else
          call cpu_time(tm)
#endif
          if(tm.ge.timer(time_handle)%beg_time+timer(time_handle)%time_interval) time_is_off=.true.
          if(time_is_off.and.present(destroy)) then
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
        return
        end function time_is_off
!--------------------------------------------------
        integer function timer_destroy(time_handle)
!This function explicitly frees a time handle.
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
!This function returns the current wall clock time in seconds.
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
