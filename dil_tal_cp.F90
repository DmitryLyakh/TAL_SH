!TAL_CP: Tensor Algebra Library Routines for Multicore CPUs and Intel MIC.
!REVISION: 2015/01/15 (started 2013/05)
!Copyright (C) 2015 Dmitry I. Lyakh (email: quant4me@gmail.com)
!LICENSE: BSD 3-Clause
!-------------------------------------------------------------------------------
!NOTES:
! # Fortran 2003 at least + OpenMP.
!OPTIONS:
! # -DUSE_MIC: enables MIC offloading directives (disabled by default).
! # -DNO_OMP: disables OpenMP multithreading (enabled by default).

       module dil_tal_cp
        use, intrinsic:: ISO_C_BINDING
#ifndef NO_OMP
        use omp_lib
#endif
        implicit none
!PARAMETERS:
        integer, parameter, private:: INTD=4                   !default integer kind (size)
        integer, parameter, private:: INTL=C_SIZE_T            !long integer kind (size), used for addressing
        integer(INTD), parameter, private:: INT_BLAS=INTD      !default integer size for BLAS/LAPACK
        integer(INTD), parameter, private:: MAX_TENSOR_RANK=16 !max allowed tensor rank
        integer(INTD), parameter, private:: MAX_THREADS=1024   !max number of CPU threads on multi-core CPU (or MIC)
        integer(INTD), private:: CONS_OUT=6                    !console output device (defaults to screen)
        logical, private:: VERBOSE=.true.                      !verbosity (for errors)
        logical, private:: DIL_DEBUG=.false.                   !debugging
#ifdef USE_MIC
!DIR$ ATTRIBUTES OFFLOAD:mic:: INTD,INTL,INT_BLAS,MAX_TENSOR_RANK,MAX_THREADS,CONS_OUT,VERBOSE,DIL_DEBUG
!DIR$ ATTRIBUTES ALIGN:128:: INTD,INTL,INT_BLAS,MAX_TENSOR_RANK,MAX_THREADS,CONS_OUT,VERBOSE,DIL_DEBUG
#endif

!FUNCTION visibility:
        private divide_segment
        private thread_wtime
        private permutation_invert
        private permutation_trivial
        public dil_tensor_slice
        public dil_tensor_insert
        public dil_tensor_transpose

       contains
!------------------------------------------------------------------------
#ifdef USE_MIC
!DIR$ ATTRIBUTES OFFLOAD:mic:: divide_segment
#endif
        subroutine divide_segment(seg_range,subseg_num,subseg_sizes,ierr) !SERIAL
!A segment of extent <seg_range> will be divided into <subseg_num> subsegments maximally uniformly.
!The length of each subsegment will be returned in the array <subseg_sizes(1:subseg_num)>.
!Any two subsegments will not differ in length by more than 1, longer subsegments preceding the shorter ones.
        implicit none
        integer(INTL), intent(in):: seg_range                   !in: total segment extent
        integer(INTL), intent(in):: subseg_num                  !in: number of subsegments requested
        integer(INTL), intent(out):: subseg_sizes(1:subseg_num) !out: lengths of the subsegments
        integer(INTD), intent(inout):: ierr                     !error code (0:success)
        integer(INTL):: i,j,k,l,m,n
        ierr=0_INTD
        if(seg_range.gt.0_INTL.and.subseg_num.gt.0_INTL) then
         n=seg_range/subseg_num; m=mod(seg_range,subseg_num)
         do i=1_INTL,m; subseg_sizes(i)=n+1_INTL; enddo
         do i=m+1_INTL,subseg_num; subseg_sizes(i)=n; enddo
        else
         ierr=-1_INTD
        endif
        return
        end subroutine divide_segment
!----------------------------------------------------
#ifdef USE_MIC
!DIR$ ATTRIBUTES OFFLOAD:mic:: thread_wtime
#endif
        function thread_wtime(time_start) result(thw) !SERIAL
!Returns the time in seconds.
        implicit none
        real(8), intent(in), optional:: time_start !in: clock start time
        real(8):: thw
#ifndef NO_OMP
        thw=omp_get_wtime()
#else
        call cpu_time(thw)
#endif
        if(present(time_start)) thw=thw-time_start
        return
        end function thread_wtime
!--------------------------------------------------------
#ifdef USE_MIC
!DIR$ ATTRIBUTES OFFLOAD:mic:: permutation_invert
#endif
        subroutine permutation_invert(plen,pin,pout,ierr) !SERIAL
!This subroutine returns an inverse for a given permutation.
!Numeration starts from 1.
        implicit none
        integer(INTD), intent(in):: plen          !in: length of the permutation
        integer(INTD), intent(in):: pin(1:plen)   !in: input permutation (must be valid!)
        integer(INTD), intent(out):: pout(1:plen) !out: inverse of the input permutation
        integer(INTD), intent(inout):: ierr       !out: error code (0:success)
        integer(INTD):: i

        ierr=0
        if(plen.gt.0) then
         do i=1,plen; pout(pin(i))=i; enddo
        elseif(plen.lt.0) then
         ierr=1
        endif
        return
        end subroutine permutation_invert
!-------------------------------------------------------------------
#ifdef USE_MIC
!DIR$ ATTRIBUTES OFFLOAD:mic:: permutation_trivial
#endif
        function permutation_trivial(plen,prm,ierr) result(prm_triv) !SERIAL
!This function returns .true. if the given permutation is trivial, .false. otherwise.
!Numeration starts from 1.
        implicit none
        integer(INTD), intent(in):: plen        !in: permutation length
        integer(INTD), intent(in):: prm(1:plen) !in: permutation
        integer(INTD), intent(inout):: ierr     !out: error code (0:success)
        logical:: prm_triv                      !out: result
        integer(INTD):: i

        ierr=0; prm_triv=.true.
        if(plen.ge.0) then
         do i=1,plen; if(prm(i).ne.i) then; prm_triv=.false.; exit; endif; enddo
        else
         ierr=1
        endif
        return
        end function permutation_trivial
!--------------------------------------------------------------------------------------
#ifdef USE_MIC
!DIR$ ATTRIBUTES OFFLOAD:mic:: dil_tensor_slice
#endif
        subroutine dil_tensor_slice(dim_num,tens,tens_ext,slice,slice_ext,ext_beg,ierr) !PARALLEL (OMP)
!This subroutine extracts a slice from a tensor.
!Tensor dimensions are assumed beginning from 0: [0..(extent-1)].
        implicit none
!---------------------------------------------
        integer(INTD), parameter:: real_kind=8
!---------------------------------------------
        integer(INTD), intent(in):: dim_num              !in: number of tensor dimensions (tensor rank)
        real(real_kind), intent(in):: tens(0:*)          !in: tensor (Fortran-like linearized storage)
        integer(INTD), intent(in):: tens_ext(1:dim_num)  !in: dimension extents for <tens>
        real(real_kind), intent(inout):: slice(0:*)      !out: slice (Fortran-like linearized storage)
        integer(INTD), intent(in):: slice_ext(1:dim_num) !in: dimension extents for <slice>
        integer(INTD), intent(in):: ext_beg(1:dim_num)   !in: beginning dimension offsets for <tens> (numeration starts at 0)
        integer(INTD), intent(inout):: ierr              !out: error code (0:success)
        integer(INTD):: i,j,k,l,m,n,ks,kf,im(1:dim_num)
        integer(INTL):: lts,lss,l_in,l_out,lb,le,ll,bases_in(1:dim_num),bases_out(1:dim_num),segs(0:MAX_THREADS)
        real(8) time_beg
#ifdef USE_MIC
!DIR$ ATTRIBUTES OFFLOAD:mic:: real_kind
!DIR$ ATTRIBUTES ALIGN:128:: real_kind,im,bases_in,bases_out,segs
#endif
        ierr=0
!        time_beg=thread_wtime() !debug
        if(dim_num.gt.0) then
         lts=1_INTL; do i=1,dim_num; bases_in(i)=lts; lts=lts*tens_ext(i); enddo   !tensor indexing bases
         lss=1_INTL; do i=1,dim_num; bases_out(i)=lss; lss=lss*slice_ext(i); enddo !slice indexing bases
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,m,n,im,l_in,l_out,lb,le,ll)
#ifndef NO_OMP
         n=omp_get_thread_num(); m=omp_get_num_threads()
#else
         n=0; m=1
#endif
!$OMP MASTER
         segs(0)=0_INTL; call divide_segment(lss,int(m,INTL),segs(1:),ierr); do i=2,m; segs(i)=segs(i)+segs(i-1); enddo
!$OMP END MASTER
!$OMP BARRIER
!$OMP FLUSH(segs)
         l_out=segs(n); do i=dim_num,1,-1; im(i)=l_out/bases_out(i); l_out=l_out-im(i)*bases_out(i); enddo
         l_in=ext_beg(1); do i=2,dim_num; l_in=l_in+(ext_beg(i)+im(i))*bases_in(i); enddo
         lb=int(im(1),INTL); le=int(slice_ext(1)-1,INTL); l_out=segs(n)-lb
         sloop: do while(l_out+lb.lt.segs(n+1))
          le=min(le,segs(n+1)-1_INTL-l_out)
          do ll=lb,le; slice(l_out+ll)=tens(l_in+ll); enddo
          l_out=l_out+le+1_INTL; lb=0_INTL
          do i=2,dim_num
           l_in=l_in+bases_in(i); im(i)=im(i)+1
           if(im(i).ge.slice_ext(i)) then
            l_in=l_in-im(i)*bases_in(i); im(i)=0
           else
            exit
           endif
          enddo
         enddo sloop
!$OMP END PARALLEL
        else
         ierr=1 !zero-rank tensor
        endif
!        write(CONS_OUT,'("#DEBUG(dil_tensor_algebra::dil_tensor_slice): kernel time/error code: ",F10.4,1x,i3)')&
!        &thread_wtime(time_beg),ierr !debug
        return
        end subroutine dil_tensor_slice
!---------------------------------------------------------------------------------------
#ifdef USE_MIC
!DIR$ ATTRIBUTES OFFLOAD:mic:: dil_tensor_insert
#endif
        subroutine dil_tensor_insert(dim_num,tens,tens_ext,slice,slice_ext,ext_beg,ierr) !PARALLEL (OMP)
!This subroutine inserts a slice into a tensor.
!Tensor dimensions are assumed beginning from 0: [0..(extent-1)].
        implicit none
!---------------------------------------------
        integer(INTD), parameter:: real_kind=8
!---------------------------------------------
        integer(INTD), intent(in):: dim_num              !in: number of tensor dimensions
        real(real_kind), intent(inout):: tens(0:*)       !inout: tensor (Fortran-like linearized storage)
        integer(INTD), intent(in):: tens_ext(1:dim_num)  !in: dimension extents for <tens>
        real(real_kind), intent(in):: slice(0:*)         !in: slice (Fortran-like linearized storage)
        integer(INTD), intent(in):: slice_ext(1:dim_num) !in: dimension extents for <slice>
        integer(INTD), intent(in):: ext_beg(1:dim_num)   !in: beginning dimension offsets for <tens> (numeration starts at 0)
        integer(INTD), intent(inout):: ierr              !out: error code (0:success)
        integer(INTD):: i,j,k,l,m,n,ks,kf,im(1:dim_num)
        integer(INTL):: lts,lss,l_in,l_out,lb,le,ll,bases_in(1:dim_num),bases_out(1:dim_num),segs(0:MAX_THREADS)
        real(8) time_beg
#ifdef USE_MIC
!DIR$ ATTRIBUTES OFFLOAD:mic:: real_kind
!DIR$ ATTRIBUTES ALIGN:128:: real_kind,im,bases_in,bases_out,segs
#endif
        ierr=0
!        time_beg=thread_wtime() !debug
        if(dim_num.gt.0) then
         lts=1_INTL; do i=1,dim_num; bases_out(i)=lts; lts=lts*tens_ext(i); enddo !tensor indexing bases
         lss=1_INTL; do i=1,dim_num; bases_in(i)=lss; lss=lss*slice_ext(i); enddo !slice indexing bases
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,m,n,im,l_in,l_out,lb,le,ll)
#ifndef NO_OMP
         n=omp_get_thread_num(); m=omp_get_num_threads()
#else
         n=0; m=1
#endif
!$OMP MASTER
         segs(0)=0_INTL; call divide_segment(lss,int(m,INTL),segs(1:),ierr); do i=2,m; segs(i)=segs(i)+segs(i-1); enddo
!$OMP END MASTER
!$OMP BARRIER
!$OMP FLUSH(segs)
         l_in=segs(n); do i=dim_num,1,-1; im(i)=l_in/bases_in(i); l_in=l_in-im(i)*bases_in(i); enddo
         l_out=ext_beg(1); do i=2,dim_num; l_out=l_out+(ext_beg(i)+im(i))*bases_out(i); enddo
         lb=int(im(1),INTL); le=int(slice_ext(1)-1,INTL); l_in=segs(n)-lb
         sloop: do while(l_in+lb.lt.segs(n+1))
          le=min(le,segs(n+1)-1_INTL-l_in)
          do ll=lb,le; tens(l_out+ll)=slice(l_in+ll); enddo
          l_in=l_in+le+1_INTL; lb=0_INTL
          do i=2,dim_num
           l_out=l_out+bases_out(i); im(i)=im(i)+1
           if(im(i).ge.slice_ext(i)) then
            l_out=l_out-im(i)*bases_out(i); im(i)=0
           else
            exit
           endif
          enddo
         enddo sloop
!$OMP END PARALLEL
        else
         ierr=1 !zero-rank tensor
        endif
!        write(CONS_OUT,'("#DEBUG(dil_tensor_algebra::dil_tensor_insert): kernel time/error code: ",F10.4,1x,i3)')&
!        &thread_wtime(time_beg),ierr !debug
        return
        end subroutine dil_tensor_insert
!--------------------------------------------------------------------------------------------
#ifdef USE_MIC
!DIR$ ATTRIBUTES OFFLOAD:mic:: dil_tensor_transpose
#endif
        subroutine dil_tensor_transpose(dim_num,dim_extents,dim_transp,tens_in,tens_out,ierr) !PARALLEL (OMP)
!This subroutine reorders dimensions of an arbitrary, locally stored tensor (out-of-place).
!Algorithm: Lyakh, D.I. Computer Physics Communications, 2014. DOI:10.1016/j.cpc.2014.12.013.
        implicit none
!---------------------------------------------
        integer(INTD), parameter:: real_kind=8
!---------------------------------------------
        logical, parameter:: cache_efficiency=.true.               !turns on/off cache-efficiency
        integer(INTL), parameter:: cache_line_len=64/real_kind     !cache line length (words)
        integer(INTL), parameter:: cache_line_min=cache_line_len*2 !lower bound for the input/output minor volume: => L1_cache_line*2
        integer(INTL), parameter:: cache_line_lim=cache_line_len*4 !upper bound for the input/output minor volume: <= SQRT(L1_size)
        integer(INTL), parameter:: small_tens_size=2**10 !up to this size it is useless to apply cache efficiency (fully fits in L1)
        integer(INTL), parameter:: vec_size=2**8 !loop reorganization parameter for direct copy
#ifdef USE_MIC
!DIR$ ATTRIBUTES OFFLOAD:mic:: real_kind,cache_efficiency,cache_line_len,cache_line_min,cache_line_lim,small_tens_size,vec_size
!DIR$ ATTRIBUTES ALIGN:128:: real_kind,cache_efficiency,cache_line_len,cache_line_min,cache_line_lim,small_tens_size,vec_size
#endif
!------------------------------------------
        integer(INTD), intent(in):: dim_num                !in: tensor rank
        integer(INTD), intent(in):: dim_extents(1:dim_num) !in: tensor dimension extents
        integer(INTD), intent(in):: dim_transp(1:dim_num)  !in: dimension permutation (O2N)
        real(real_kind), intent(in):: tens_in(0:*)         !in: input tensor (Fortran-like linearized storage)
        real(real_kind), intent(out):: tens_out(0:*)       !out: output (permuted) tensor (Fortran-like linearized storage)
        integer(INTD), intent(inout):: ierr                !out: error code (0:success)
        integer(INTD):: i,j,k,l,m,n,k1,k2,ks,kf,split_in,split_out
        integer(INTD):: im(1:dim_num),n2o(0:dim_num+1),ipr(1:dim_num+1),dim_beg(1:dim_num),dim_end(1:dim_num)
        integer(INTL):: bases_in(1:dim_num+1),bases_out(1:dim_num+1),bases_pri(1:dim_num+1),segs(0:MAX_THREADS)
        integer(INTL):: bs,l0,l1,l2,l3,ll,lb,le,ls,l_in,l_out,seg_in,seg_out,vol_min,vol_ext
        logical:: trivial
        real(8):: time_beg,tm
#ifdef USE_MIC
!DIR$ ATTRIBUTES ALIGN:128:: im,n2o,ipr,dim_beg,dim_end,bases_in,bases_out,bases_pri,segs
#endif
        ierr=0
!        if(DIL_DEBUG) write(CONS_OUT,'(3x,"#DEBUG(DIL): Transposing ... ")',ADVANCE='NO')
!        time_beg=thread_wtime() !debug
        if(dim_num.lt.0) then; ierr=1; return; elseif(dim_num.eq.0) then; tens_out(0)=tens_in(0); return; endif
!Check the index permutation:
        trivial=.true.; do i=1,dim_num; if(dim_transp(i).ne.i) then; trivial=.false.; exit; endif; enddo
        if(trivial.and.cache_efficiency) then
!Trivial index permutation (no permutation):
 !Compute indexing bases:
         bs=1_INTL; do i=1,dim_num; bases_in(i)=bs; bs=bs*dim_extents(i); enddo
 !Copy input to output:
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(l0,l1)
!$OMP DO SCHEDULE(GUIDED)
         do l0=0_INTL,bs-1_INTL-mod(bs,vec_size),vec_size
          do l1=0_INTL,vec_size-1_INTL; tens_out(l0+l1)=tens_in(l0+l1); enddo
         enddo
!$OMP END DO NOWAIT
!$OMP SINGLE
         do l0=bs-mod(bs,vec_size),bs-1_INTL; tens_out(l0)=tens_in(l0); enddo
!$OMP END SINGLE
!$OMP END PARALLEL
        else
!Non-trivial index permutation:
 !Compute indexing bases:
         do i=1,dim_num; n2o(dim_transp(i))=i; enddo; n2o(dim_num+1)=dim_num+1 !get the N2O
         bs=1_INTL; do i=1,dim_num; bases_in(i)=bs; bs=bs*dim_extents(i); enddo; bases_in(dim_num+1)=bs
         bs=1_INTL; do i=1,dim_num; bases_out(n2o(i))=bs; bs=bs*dim_extents(n2o(i)); enddo; bases_out(dim_num+1)=bs
 !Configure cache-efficient algorithm:
         if(bs.le.small_tens_size.or.(.not.cache_efficiency)) then !tensor block is too small to think hard about it
          ipr(1:dim_num+1)=(/(j,j=1,dim_num+1)/); kf=dim_num !trivial priorities, all indices are minor
          split_in=kf; seg_in=dim_extents(split_in); split_out=kf; seg_out=dim_extents(split_out)
         else
          do k1=1,dim_num; if(bases_in(k1+1).ge.cache_line_lim) exit; enddo; k1=k1-1
          do k2=1,dim_num; if(bases_out(n2o(k2+1)).ge.cache_line_lim) exit; enddo; k2=k2-1
          do j=k1+1,dim_num; if(dim_transp(j).le.k2) then; k1=k1+1; else; exit; endif; enddo
          do j=k2+1,dim_num; if(n2o(j).le.k1) then; k2=k2+1; else; exit; endif; enddo
          if(bases_in(k1+1).lt.cache_line_min.and.bases_out(n2o(k2+1)).ge.cache_line_min) then !split the last minor input dim
           k1=k1+1; split_in=k1; seg_in=(cache_line_lim-1_INTL)/bases_in(split_in)+1_INTL
           split_out=n2o(k2); seg_out=dim_extents(split_out)
          elseif(bases_in(k1+1).ge.cache_line_min.and.bases_out(n2o(k2+1)).lt.cache_line_min) then !split the last minor output dim
           k2=k2+1; split_in=n2o(k2); seg_in=(cache_line_lim-1_INTL)/bases_out(split_in)+1_INTL
           split_out=k1; seg_out=dim_extents(split_out)
          elseif(bases_in(k1+1).lt.cache_line_min.and.bases_out(n2o(k2+1)).lt.cache_line_min) then !split both
           k1=k1+1; k2=k2+1
           if(k1.eq.n2o(k2)) then
            split_in=k1; seg_in=(cache_line_lim-1_INTL)/min(bases_in(split_in),bases_out(split_in))+1_INTL
            split_out=k1; seg_out=dim_extents(split_out)
           else
            split_in=k1; seg_in=(cache_line_lim-1_INTL)/bases_in(split_in)+1_INTL
            split_out=n2o(k2); seg_out=(cache_line_lim-1_INTL)/bases_out(split_out)+1_INTL
           endif
          else !split none
           split_in=k1; seg_in=dim_extents(split_in)
           split_out=n2o(k2); seg_out=dim_extents(split_out)
          endif
          vol_min=1_INTL
          if(seg_in.lt.dim_extents(split_in)) vol_min=vol_min*seg_in
          if(seg_out.lt.dim_extents(split_out)) vol_min=vol_min*seg_out
          if(vol_min.gt.1_INTL) then
           do j=1,k1
            if(j.ne.split_in.and.j.ne.split_out) vol_min=vol_min*dim_extents(j)
           enddo
           do j=1,k2
            l=n2o(j)
            if(l.gt.k1.and.l.ne.split_in.and.l.ne.split_out) vol_min=vol_min*dim_extents(l)
           enddo
           l=int((cache_line_lim*cache_line_lim)/vol_min,4)
           if(l.ge.2) then
            if(split_in.eq.split_out) then
             seg_in=seg_in*l
            else
             if(l.gt.4) then
              l=int(sqrt(float(l)),4)
              seg_in=min(seg_in*l,int(dim_extents(split_in),INTL))
              seg_out=min(seg_out*l,int(dim_extents(split_out),INTL))
             else
              seg_in=min(seg_in*l,int(dim_extents(split_in),INTL))
             endif
            endif
           endif
          endif
          l=0
          do while(l.lt.k1)
           l=l+1; ipr(l)=l; if(bases_in(l+1).ge.cache_line_min) exit
          enddo
          m=l+1
          j=0
          do while(j.lt.k2)
           j=j+1; n=n2o(j)
           if(n.ge.m) then; l=l+1; ipr(l)=n; endif
           if(bases_out(n2o(j+1)).ge.cache_line_min) exit
          enddo
          n=j+1
          do j=m,k1; if(dim_transp(j).ge.n) then; l=l+1; ipr(l)=j; endif; enddo
          do j=n,k2; if(n2o(j).gt.k1) then; l=l+1; ipr(l)=n2o(j); endif; enddo
          kf=l
          do j=k2+1,dim_num; if(n2o(j).gt.k1) then; l=l+1; ipr(l)=n2o(j); endif; enddo !kf is the length of the combined minor set
          ipr(dim_num+1)=dim_num+1 !special setting
         endif
         vol_ext=1_INTL; do j=kf+1,dim_num; vol_ext=vol_ext*dim_extents(ipr(j)); enddo !external volume
!         write(CONS_OUT,'("#DEBUG(dil_tensor_algebra::dil_tensor_transpose): extents:",99(1x,i5))') dim_extents(1:dim_num) !debug
!         write(CONS_OUT,'("#DEBUG(dil_tensor_algebra::dil_tensor_transpose): permutation:",99(1x,i2))') dim_transp(1:dim_num) !debug
!         write(CONS_OUT,'("#DEBUG(dil_tensor_algebra::dil_tensor_transpose): minor ",i3,": priority:",99(1x,i2))')&
!         &kf,ipr(1:dim_num) !debug
!         write(CONS_OUT,'("#DEBUG(dil_tensor_algebra::dil_tensor_transpose): vol_ext ",i11,": segs:",4(1x,i5))')&
!         &vol_ext,split_in,split_out,seg_in,seg_out !debug
 !Transpose:
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,j,m,n,ks,l0,l1,l2,l3,ll,lb,le,ls,l_in,l_out,vol_min,im,dim_beg,dim_end)
#ifndef NO_OMP
         n=omp_get_thread_num(); m=omp_get_num_threads() !multi-threaded execution
#else
         n=0; m=1 !serial execution
#endif
!         if(n.eq.0) write(CONS_OUT,'("#DEBUG(dil_tensor_algebra::dil_tensor_transpose): number of threads = ",i5)') m !debug
         if(kf.lt.dim_num) then !external indices present
!$OMP MASTER
          segs(0)=0_INTL; call divide_segment(vol_ext,int(m,INTL),segs(1:),i); do j=2,m; segs(j)=segs(j)+segs(j-1); enddo
          l0=1_INTL; do i=kf+1,dim_num; bases_pri(ipr(i))=l0; l0=l0*dim_extents(ipr(i)); enddo !priority bases
!$OMP END MASTER
!$OMP BARRIER
!$OMP FLUSH(segs,bases_pri)
          dim_beg(1:dim_num)=0; dim_end(1:dim_num)=dim_extents(1:dim_num)-1
          l2=dim_end(split_in); l3=dim_end(split_out); ls=bases_out(1)
          loop0: do l1=0_INTL,l3,seg_out !output dimension
           dim_beg(split_out)=l1; dim_end(split_out)=min(l1+seg_out-1_INTL,l3)
           do l0=0_INTL,l2,seg_in !input dimension
            dim_beg(split_in)=l0; dim_end(split_in)=min(l0+seg_in-1_INTL,l2)
            ll=segs(n); do i=dim_num,kf+1,-1; j=ipr(i); im(j)=ll/bases_pri(j); ll=ll-im(j)*bases_pri(j); enddo
            vol_min=1_INTL; do i=1,kf; j=ipr(i); vol_min=vol_min*(dim_end(j)-dim_beg(j)+1); im(j)=dim_beg(j); enddo
            l_in=0_INTL; do j=1,dim_num; l_in=l_in+im(j)*bases_in(j); enddo
            l_out=0_INTL; do j=1,dim_num; l_out=l_out+im(j)*bases_out(j); enddo
            le=dim_end(1)-dim_beg(1); lb=(segs(n+1)-segs(n))*vol_min; ks=0
            loop1: do while(lb.gt.0_INTL)
             do ll=0_INTL,le
              tens_out(l_out+ll*ls)=tens_in(l_in+ll)
             enddo
             lb=lb-(le+1_INTL)
             do i=2,dim_num
              j=ipr(i) !old index number
              if(im(j).lt.dim_end(j)) then
               im(j)=im(j)+1; l_in=l_in+bases_in(j); l_out=l_out+bases_out(j)
               ks=ks+1; exit
              else
               l_in=l_in-(im(j)-dim_beg(j))*bases_in(j); l_out=l_out-(im(j)-dim_beg(j))*bases_out(j); im(j)=dim_beg(j)
              endif
             enddo !i
             ks=ks-1; if(ks.lt.0) exit loop1
            enddo loop1
            if(lb.ne.0_INTL) then
             if(VERBOSE) write(CONS_OUT,'("ERROR(dil_tensor_algebra::dil_tensor_transpose): invalid remainder: ",i11,1x,i4)') lb,n
!$OMP ATOMIC WRITE
             ierr=2
             exit loop0
            endif
           enddo !l0
          enddo loop0 !l1
         else !external indices absent
!$OMP MASTER
          l0=1_INTL; do i=kf+1,dim_num; bases_pri(ipr(i))=l0; l0=l0*dim_extents(ipr(i)); enddo !priority bases
!$OMP END MASTER
!$OMP BARRIER
!$OMP FLUSH(bases_pri)
          dim_beg(1:dim_num)=0; dim_end(1:dim_num)=dim_extents(1:dim_num)-1
          l2=dim_end(split_in); l3=dim_end(split_out); ls=bases_out(1)
!$OMP DO SCHEDULE(DYNAMIC) COLLAPSE(2)
          do l1=0_INTL,l3,seg_out !output dimension
           do l0=0_INTL,l2,seg_in !input dimension
            dim_beg(split_out)=l1; dim_end(split_out)=min(l1+seg_out-1_INTL,l3)
            dim_beg(split_in)=l0; dim_end(split_in)=min(l0+seg_in-1_INTL,l2)
            vol_min=1_INTL; do i=1,kf; j=ipr(i); vol_min=vol_min*(dim_end(j)-dim_beg(j)+1); im(j)=dim_beg(j); enddo
            l_in=0_INTL; do j=1,dim_num; l_in=l_in+im(j)*bases_in(j); enddo
            l_out=0_INTL; do j=1,dim_num; l_out=l_out+im(j)*bases_out(j); enddo
            le=dim_end(1)-dim_beg(1); lb=vol_min; ks=0
            loop2: do while(lb.gt.0_INTL)
             do ll=0_INTL,le
              tens_out(l_out+ll*ls)=tens_in(l_in+ll)
             enddo
             lb=lb-(le+1_INTL)
             do i=2,dim_num
              j=ipr(i) !old index number
              if(im(j).lt.dim_end(j)) then
               im(j)=im(j)+1; l_in=l_in+bases_in(j); l_out=l_out+bases_out(j)
               ks=ks+1; exit
              else
               l_in=l_in-(im(j)-dim_beg(j))*bases_in(j); l_out=l_out-(im(j)-dim_beg(j))*bases_out(j); im(j)=dim_beg(j)
              endif
             enddo !i
             ks=ks-1; if(ks.lt.0) exit loop2
            enddo loop2
           enddo !l0
          enddo !l1
!$OMP END DO
         endif
!$OMP END PARALLEL
        endif !trivial or not
!        tm=thread_wtime(time_beg) !debug
!        if(DIL_DEBUG) write(CONS_OUT,'("Done: ",F10.4," s, ",F10.4," GB/s: stat ",i9)')&
!        &tm,dble(2_INTL*bs*real_kind)/(tm*1048576d0*1024d0),ierr !debug
!        write(CONS_OUT,'("#DEBUG(dil_tensor_algebra::dil_tensor_transpose): Done: ",F10.4," sec, ",F10.4," GB/s, error ",i3)')&
!        &tm,dble(2_INTL*bs*real_kind)/(tm*1024d0*1024d0*1024d0),ierr !debug
        return
        end subroutine dil_tensor_transpose

       end module dil_tal_cp
