       module tensor_dil_omp
!Multithreaded tensor algebra kernels tailored for ACESIII/ACESIV.
!AUTHOR: Dmitry I. Lyakh: quant4me@gmail.com
!Revision: 2015/04/27
!-------------------------------------------
!COMPILE:
! # GFORTRAN flags: -O3 --free-line-length-none -fopenmp -x f95-cpp-input
!------------------------------------------------------------------------
!LINK:
! # GCC flags: -lgfortran -lgomp
!-------------------------------
!PREPROCESSOR DIRECTIVES:
! # NO_OMP - a serialized version will be compiled (no OpenMP);
! # NO_BLAS - BLAS calls will be replaced by my own routines (DIL);
! # USE_MKL - use MKL BLAS;
!----------------------------------------------------------------------------------------------------
!PUBLIC SUBROUTINES CONVENTIONS:
! # Serial subroutines/functions have one underscore at the end; parallel subroutines have two underscores at the end.
! # The last argument is always the error code (0:success).
! # For parallel subroutines, the first argument is always the number of threads requested (max).
! # Scalar variables can be passed from C/C++ either by pointers (C) or by reference (C++) (use appropriate prototypes).
! # C/C++ integer type <int> must be at least 32 bit!
!PROTOTYPES by POINTER:
! # long long int tensor_size_by_shape_(int*, int*, int*);
! # void get_contraction_ptrn_(int*, int*, int*, int*, int*, int*);
! # void tensor_block_init__(int*, double*, int*, int*, double*, int*);
! # void tensor_block_scale__(int*, double*, int*, int*, double*, int*);
! # double tensor_block_norm2__(int*, double*, int*, int*, int*);
! # void tensor_block_slice__(int*, int*, double*, int*, double*, int*, int*, int*);
! # void tensor_block_insert__(int*, int*, double*, int*, double*, int*, int*, int*);
! # void tensor_block_add__(int*, int*, int*, double*, double*, double*, int*);
! # void tensor_block_copy__(int*, int*, int*, int*, double*, double*, int*);
! # void tensor_block_contract__(int*, int*, double*, int*, int*, double*, int*, int*, double*, int*, int*, int*);
!PROTOTYPES by REFERENCE:
! # long long int tensor_size_by_shape_(int&, int*, int&);
! # void get_contraction_ptrn_(int&, int&, int&, int*, int*, int&);
! # void tensor_block_init__(int&, double*, int&, int*, double&, int&);
! # void tensor_block_scale__(int&, double*, int&, int*, double&, int&);
! # double tensor_block_norm2__(int&, double*, int&, int*, int&);
! # void tensor_block_slice__(int&, int&, double*, int*, double*, int*, int*, int&);
! # void tensor_block_insert__(int&, int&, double*, int*, double*, int*, int*, int&);
! # void tensor_block_add__(int&, int&, int*, double*, double*, double&, int&);
! # void tensor_block_copy__(int&, int&, int*, int*, double*, double*, int&);
! # void tensor_block_contract__(int&, int*, double*, int&, int*, double*, int&, int*, double*, int&, int*, int&);
!----------------------------------------------------------------------------------------------------------------
	use, intrinsic:: ISO_C_BINDING
	use timers
#ifdef USE_MKL
        use mkl95_blas
        use mkl95_lapack
        use mkl95_precision
#endif
        implicit none
!MODULE PARAMETERS:
        integer, parameter:: max_tensor_rank=32 !max allowed tensor rank
        integer, parameter:: max_threads=1024   !max allowed number of OMP threads
!C TYPES:
        integer, parameter:: int_kind=C_INT
        integer, parameter:: int8_kind=C_LONG_LONG
        integer, parameter:: real4_kind=C_FLOAT
        integer, parameter:: real8_kind=C_DOUBLE
!OpenMP functions (or include omp_lib.h instead):
#ifndef NO_OMP
	integer, external, private:: omp_get_thread_num,omp_get_num_threads,omp_get_max_threads
#endif
	contains
!-----------------------------------------
!PUBLIC SUBROUTINES (callable from C/C++):
!--------------------------------------------------------------------------
	integer(int8_kind) function tensor_size_by_shape(num_dim,dims,ierr) bind(c,name='tensor_size_by_shape_') !SERIAL
!This function returns the size of a tensor block specified by its shape.
!INPUT:
! - dims(1:num_dim) - tensor dimension extents;
!OUTPUT:
! - tensor_size_by_shape - tensor size;
! - ierr - error code (0:success).
	implicit none
	integer(int_kind), intent(in):: num_dim,dims(1:num_dim)
	integer(int_kind), intent(inout):: ierr
	integer(int_kind):: i

	ierr=0; tensor_size_by_shape=1_8
	if(num_dim.gt.0) then !true tensor
	 do i=1,num_dim; tensor_size_by_shape=tensor_size_by_shape*dims(i); enddo
	 if(tensor_size_by_shape.le.0_8) ierr=2
	elseif(num_dim.lt.0) then !invalid number of dimensions
	 ierr=1; tensor_size_by_shape=0_8
	endif
	return
	end function tensor_size_by_shape
!--------------------------------------------------------------------------------
	subroutine get_contraction_ptrn(drank,lrank,rrank,aces_ptrn,my_ptrn,ierr) bind(c,name='get_contraction_ptrn_') !SERIAL
!This subroutine converts an ACESIII contraction pattern specification into my internal contraction pattern specification.
!INPUT:
! - drank - rank of the destination tensor block;
! - lrank - rank of the left tensor operand;
! - rrank - rank of the right tensor operand;
! - aces_ptrn(1:drank+lrank+rrank) - a union of three integer ACESIII arrays which uniquely label the indices (by integers);
!OUTPUT:
! - my_ptrn(1:lrank+rrank) - my format of the contraction pattern specification (supplied to tensor_block_contract__);
! - ierr - error code (0:success).
	implicit none
	integer(int_kind), intent(in):: drank,lrank,rrank,aces_ptrn(1:*)
	integer(int_kind), intent(out):: my_ptrn(1:*)
	integer(int_kind), intent(inout):: ierr
	integer(int_kind):: i,j,k,l,m,n,ks,kf,trn(0:drank+lrank+rrank)

	ierr=0
	if(drank.ge.0.and.lrank.ge.0.and.rrank.ge.0) then
	 m=lrank+rrank; n=drank+m
	 if(mod(n,2).eq.0) then
	  if(m.gt.0) then
 !Sort:
	   trn(0)=+1; do i=1,n; trn(i)=i; enddo
	   call merge_sort_key_int_(n,aces_ptrn,trn)
 !Check validity of <aces_ptrn>:
	   do i=1,n-1,2
	    if(aces_ptrn(trn(i)).eq.aces_ptrn(trn(i+1))) then
	     if(i.gt.1) then
	      if(aces_ptrn(trn(i)).eq.aces_ptrn(trn(i-1))) then; ierr=5; return; endif
	     endif
	    else
	     ierr=4; return
	    endif
	   enddo
  !Create <my_ptrn>:
	   do i=1,n-1,2
	    j=trn(i); k=trn(i+1)
	    if(j.le.drank.and.k.gt.drank) then !uncontracted index
	     my_ptrn(k-drank)=j
	    elseif(j.gt.drank.and.j.le.drank+lrank.and.k.gt.drank+lrank) then !contracted index
	     my_ptrn(j-drank)=-(k-(drank+lrank)); my_ptrn(k-drank)=-(j-drank)
	    else
	     ierr=6; return
	    endif
	   enddo
	  else
	   if(drank.gt.0) ierr=3
	  endif
	 else
	  ierr=2
	 endif
	else
	 ierr=1
	endif
	return
	end subroutine get_contraction_ptrn
!----------------------------------------------------------------------------------------
	subroutine tensor_block_init_(nthreads,tens,tens_rank,tens_dim_extents,sval,ierr) &
                                      bind(c,name='tensor_block_init__') !PARALLEL
!This subroutine initializes a tensor block <tens> with some scalar value <sval>.
!INPUT:
! - nthreads - number of threads requested;
! - tens - tensor block;
! - tens_rank - tensor rank;
! - tens_dim_extents(1:tens_rank) - tensor dimension extents;
! - sval - initialization scalar value <val>;
!OUTPUT:
! - tens - filled tensor block;
! - ierr - error code (0: success):
	implicit none
!----------------------------------------------------
	integer(int8_kind), parameter:: vec_size=2**8 !loop reorganization parameter
!----------------------------------------------------
	integer(int_kind), intent(in):: nthreads
	real(real8_kind), intent(inout):: tens(0:*)
	integer(int_kind), intent(in):: tens_rank,tens_dim_extents(1:tens_rank)
	real(real8_kind), intent(in):: sval
	integer(int_kind), intent(inout):: ierr
	integer(int_kind):: i,j,k,l,m,n,ks,kf
	integer(int8_kind):: tens_size,l0,l1,l2
	real(real8_kind):: vec(0:vec_size-1),val
	real(real8_kind):: time_beg

	ierr=0
!	time_beg=thread_wtime() !debug
	if(tens_rank.eq.0) then !scalar
	 tens(0)=sval
	elseif(tens_rank.gt.0) then !tensor
	 tens_size=1_8; do i=1,tens_rank; tens_size=tens_size*tens_dim_extents(i); enddo
	 vec(0_8:vec_size-1_8)=sval
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(l0,l1) NUM_THREADS(nthreads)
!$OMP DO SCHEDULE(GUIDED)
	 do l0=0_8,tens_size-1_8-mod(tens_size,vec_size),vec_size
	  do l1=0_8,vec_size-1_8; tens(l0+l1)=vec(l1); enddo
	 enddo
!$OMP END DO NOWAIT
!$OMP MASTER
	 tens(tens_size-mod(tens_size,vec_size):tens_size-1_8)=sval
!$OMP END MASTER
!$OMP END PARALLEL
	else
	 ierr=-1
	endif
!	write(*,'("DEBUG(tensor_algebra_dil::tensor_block_init_): exit code / kernel time = ",i5,1x,F10.4)') &
!        ierr,thread_wtime(time_beg) !debug
	return
	end subroutine tensor_block_init_
!----------------------------------------------------------------------------------------------
	subroutine tensor_block_scale_(nthreads,tens,tens_rank,tens_dim_extents,scale_fac,ierr) &
                                       bind(c,name='tensor_block_scale__') !PARALLEL
!This subroutine multiplies a tensor block <tens> by <scale_fac>.
!INPUT:
! - nthreads - number of threads requested;
! - tens - tensor block;
! - tens_rank - tensor rank;
! - tens_dim_extents(1:tens_rank) - tensor dimension extents;
! - scale_fac - scaling factor;
!OUTPUT:
! - tens - scaled tensor block;
! - ierr - error code (0:success).
	implicit none
	integer(int_kind), intent(in):: nthreads
	real(real8_kind), intent(inout):: tens(0:*)
	integer(int_kind), intent(in):: tens_rank,tens_dim_extents(1:tens_rank)
	real(real8_kind), intent(in):: scale_fac
	integer(int_kind), intent(inout):: ierr
	integer(int_kind):: i,j,k,l,m,n
	integer(int8_kind):: l0,tens_size
	real(real8_kind):: time_beg

	ierr=0
!	time_beg=thread_wtime() !debug
	if(tens_rank.eq.0) then !scalar
	 tens(0)=tens(0)*scale_fac
	elseif(tens_rank.gt.0) then !tensor
	 tens_size=1_8; do i=1,tens_rank; tens_size=tens_size*tens_dim_extents(i); enddo
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED) NUM_THREADS(nthreads)
	 do l0=0_8,tens_size-1_8; tens(l0)=tens(l0)*scale_fac; enddo
!$OMP END PARALLEL DO
	else
	 ierr=-1
	endif
!	write(*,'("DEBUG(tensor_algebra_dil::tensor_block_scale_): exit code / kernel time = ",i5,1x,F10.4)') &
!        ierr,thread_wtime(time_beg) !debug
	return
	end subroutine tensor_block_scale_
!---------------------------------------------------------------------------------------------------
	real(real8_kind) function tensor_block_norm2_(nthreads,tens,tens_rank,tens_dim_extents,ierr) &
                                                      bind(c,name='tensor_block_norm2__') !PARALLEL
!This function computes the squared Euclidean (Frobenius) norm of a tensor block.
!INPUT:
! - nthreads - number of threads requested;
! - tens - tensor block;
! - tens_rank - tensor rank;
! - tens_dim_extents(1:tens_rank) - dimension extents;
!OUTPUT:
! - tensor_block_norm2_ - squared Euclidean norm;
! - ierr - error code (0:success).
	implicit none
	integer(int_kind), intent(in):: nthreads
	real(real8_kind), intent(in):: tens(0:*)
	integer(int_kind), intent(in):: tens_rank,tens_dim_extents(1:tens_rank)
	integer(int_kind), intent(inout):: ierr
	integer(int_kind):: i,j,k,l,m,n
	integer(int8_kind):: l0,tens_size
	real(real8_kind):: val
	real(real8_kind):: time_beg

	ierr=0
!	time_beg=thread_wtime() !debug
	tensor_block_norm2_=0d0
	if(tens_rank.eq.0) then !scalar
	 tensor_block_norm2_=tens(0)**2
	elseif(tens_rank.gt.0) then !tensor
	 tens_size=1_8; do i=1,tens_rank; tens_size=tens_size*tens_dim_extents(i); enddo
	 val=0d0
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(l0) NUM_THREADS(nthreads)
!$OMP DO SCHEDULE(GUIDED) REDUCTION(+:val)
	 do l0=0_8,tens_size-1_8; val=val+tens(l0)**2; enddo
!$OMP END DO
!$OMP END PARALLEL
	 tensor_block_norm2_=val
	else
	 ierr=-1
	endif
!	write(*,'("DEBUG(tensor_algebra_dil::tensor_block_norm2_): exit code / kernel time = ",i5,1x,F10.4)') &
!        ierr,thread_wtime(time_beg) !debug
	return
	end function tensor_block_norm2_
!--------------------------------------------------------------------------------------------------
	subroutine tensor_block_slice_(nthreads,dim_num,tens,tens_ext,slice,slice_ext,ext_beg,ierr) &
                                       bind(c,name='tensor_block_slice__') !PARALLEL
!This subroutine extracts a slice from a tensor block.
!INPUT:
! - nthreads - number of threads requested;
! - dim_num - number of tensor dimensions;
! - tens(0:) - tensor block (array);
! - tens_ext(1:dim_num) - dimension extents for <tens>;
! - slice_ext(1:dim_num) - dimension extents for <slice>;
! - ext_beg(1:dim_num) - beginning dimension offsets for <tens> (numeration starts at 0);
!OUTPUT:
! - slice(0:) - slice (array);
! - ierr - error code (0:success).
	implicit none
	integer(int_kind), intent(in):: nthreads,dim_num,tens_ext(1:dim_num),slice_ext(1:dim_num),ext_beg(1:dim_num)
	real(real8_kind), intent(in):: tens(0:*)
	real(real8_kind), intent(out):: slice(0:*)
	integer(int_kind), intent(inout):: ierr
	integer(int_kind):: i,j,k,l,m,n,ks,kf,im(1:dim_num)
	integer(int8_kind):: bases_in(1:dim_num),bases_out(1:dim_num),lts,lss,l_in,l_out,segs(0:max_threads)
	real(real8_kind):: time_beg

	ierr=0
!	time_beg=thread_wtime() !debug
	if(dim_num.gt.0) then !true tensor
	 lts=1_8; do i=1,dim_num; bases_in(i)=lts; lts=lts*tens_ext(i); enddo   !tensor block indexing bases
	 lss=1_8; do i=1,dim_num; bases_out(i)=lss; lss=lss*slice_ext(i); enddo !slice indexing bases
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,m,n,kf,im,l_in,l_out)
#ifndef NO_OMP
	 n=omp_get_thread_num(); m=omp_get_num_threads()
#else
	 n=0; m=1
#endif
!$OMP MASTER
	 segs(0)=0_8; call divide_segment_(lss,int(m,8),segs(1:),ierr); do i=2,m; segs(i)=segs(i)+segs(i-1); enddo
!$OMP END MASTER
!$OMP BARRIER
!$OMP FLUSH(segs)
	 l_out=segs(n); do i=dim_num,1,-1; im(i)=l_out/bases_out(i); l_out=mod(l_out,bases_out(i)); enddo
	 l_in=ext_beg(1)+im(1); do i=2,dim_num; l_in=l_in+(ext_beg(i)+im(i))*bases_in(i); enddo; kf=0
	 sloop: do l_out=segs(n),segs(n+1)-1_8
	  slice(l_out)=tens(l_in)
	  do i=1,dim_num
	   if(im(i)+1.lt.slice_ext(i)) then
	    im(i)=im(i)+1; l_in=l_in+bases_in(i)
	    kf=kf+1; exit
	   else
	    l_in=l_in-im(i)*bases_in(i); im(i)=0
	   endif
	  enddo
	  kf=kf-1; if(kf.lt.0) exit sloop
	 enddo sloop
!$OMP END PARALLEL
	elseif(dim_num.eq.0) then !scalar
	 slice(0)=tens(0)
	else
	 ierr=1 !invalid number of dimensions
	endif
!	write(cons_out,'("DEBUG(tensor_algebra_dil::tensor_block_slice_): exit code / kernel time: ",i5,1x,F10.4)') &
!        ierr,thread_wtime(time_beg) !debug
	return
	end subroutine tensor_block_slice_
!---------------------------------------------------------------------------------------------------
	subroutine tensor_block_insert_(nthreads,dim_num,tens,tens_ext,slice,slice_ext,ext_beg,ierr) &
                                        bind(c,name='tensor_block_insert__') !PARALLEL
!This subroutine inserts a slice into a tensor block.
!INPUT:
! - nthreads - number of threads requested;
! - dim_num - number of tensor dimensions;
! - tens_ext(1:dim_num) - dimension extents for <tens>;
! - slice(0:) - slice (array);
! - slice_ext(1:dim_num) - dimension extents for <slice>;
! - ext_beg(1:dim_num) - beginning dimension offsets for <tens> (numeration starts at 0);
!OUTPUT:
! - tens(0:) - tensor block (array);
! - ierr - error code (0:success).
	implicit none
	integer(int_kind), intent(in):: nthreads,dim_num,tens_ext(1:dim_num),slice_ext(1:dim_num),ext_beg(1:dim_num)
	real(real8_kind), intent(in):: slice(0:*)
	real(real8_kind), intent(inout):: tens(0:*)
	integer(int_kind), intent(inout):: ierr
	integer(int_kind):: i,j,k,l,m,n,ks,kf,im(1:dim_num)
	integer(int8_kind):: bases_in(1:dim_num),bases_out(1:dim_num),lts,lss,l_in,l_out,segs(0:max_threads)
	real(real8_kind):: time_beg

	ierr=0
!	time_beg=thread_wtime() !debug
	if(dim_num.gt.0) then !true tensor
	 lts=1_8; do i=1,dim_num; bases_out(i)=lts; lts=lts*tens_ext(i); enddo !tensor block indexing bases
	 lss=1_8; do i=1,dim_num; bases_in(i)=lss; lss=lss*slice_ext(i); enddo !slice indexing bases
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,m,n,kf,im,l_in,l_out)
#ifndef NO_OMP
	 n=omp_get_thread_num(); m=omp_get_num_threads()
#else
	 n=0; m=1
#endif
!$OMP MASTER
	 segs(0)=0_8; call divide_segment_(lss,int(m,8),segs(1:),ierr); do i=2,m; segs(i)=segs(i)+segs(i-1); enddo
!$OMP END MASTER
!$OMP BARRIER
!$OMP FLUSH(segs)
	 l_in=segs(n); do i=dim_num,1,-1; im(i)=l_in/bases_in(i); l_in=mod(l_in,bases_in(i)); enddo
	 l_out=ext_beg(1)+im(1); do i=2,dim_num; l_out=l_out+(ext_beg(i)+im(i))*bases_out(i); enddo; kf=0
	 sloop: do l_in=segs(n),segs(n+1)-1_8
	  tens(l_out)=slice(l_in)
	  do i=1,dim_num
	   if(im(i)+1.lt.slice_ext(i)) then
	    im(i)=im(i)+1; l_out=l_out+bases_out(i)
	    kf=kf+1; exit
	   else
	    l_out=l_out-im(i)*bases_out(i); im(i)=0
	   endif
	  enddo
	  kf=kf-1; if(kf.lt.0) exit sloop
	 enddo sloop
!$OMP END PARALLEL
	elseif(dim_num.eq.0) then !scalar
	 tens(0)=slice(0)
	else
	 ierr=1 !invalid number of dimensions
	endif
!	write(cons_out,'("DEBUG(tensor_algebra_dil::tensor_block_insert_): exit code / kernel time: ",i5,1x,F10.4)') &
!        ierr,thread_wtime(time_beg) !debug
	return
	end subroutine tensor_block_insert_
!--------------------------------------------------------------------------------------------
	subroutine tensor_block_add_(nthreads,dim_num,dim_extents,tens0,tens1,scale_fac,ierr) &
                                     bind(c,name='tensor_block_add__') !PARALLEL
!This subroutine adds tensor block <tens1> to tensor block <tens0>:
!tens0(:)+=tens1(:)*scale_fac
!INPUT:
! - nthreads - number of threads requested;
! - dim_num - number of dimensions;
! - dim_extents(1:dim_num) - dimension extents;
! - tens0, tens1 - tensor blocks;
! - scale_fac - scaling factor;
!OUTPUT:
! - tens0 - modified tensor block;
! - ierr - error code (0:success);
	implicit none
	integer(int_kind), intent(in):: nthreads,dim_num,dim_extents(1:dim_num)
	real(real8_kind), intent(inout):: tens0(0:*)
	real(real8_kind), intent(in):: tens1(0:*)
	real(real8_kind), intent(in):: scale_fac
	integer(int_kind), intent(inout):: ierr
	integer(int_kind):: i,j,k,l,m,n,ks,kf
	integer(int8_kind):: l0,l1,tens_size
	real(real8_kind):: time_beg

	ierr=0
!	time_beg=thread_wtime() !debug
	if(dim_num.eq.0) then !scalar
	 tens0(0)=tens0(0)+tens1(0)*scale_fac
	elseif(dim_num.gt.0) then !tensor
	 tens_size=1_8; do i=1,dim_num; tens_size=tens_size*dim_extents(i); enddo
	 if(abs(scale_fac-1d0).gt.1d-13) then !scaling present
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED) NUM_THREADS(nthreads)
	  do l0=0_8,tens_size-1_8; tens0(l0)=tens0(l0)+tens1(l0)*scale_fac; enddo
!$OMP END PARALLEL DO
	 else !no scaling
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED) NUM_THREADS(nthreads)
	  do l0=0_8,tens_size-1_8; tens0(l0)=tens0(l0)+tens1(l0); enddo
!$OMP END PARALLEL DO
	 endif
	else
	 ierr=-1
	endif
!	write(*,'("DEBUG(tensor_algebra_dil::tensor_block_add_): exit code / kernel time = ",i5,1x,F10.4)') &
!        ierr,thread_wtime(time_beg) !debug
	return
	end subroutine tensor_block_add_
!---------------------------------------------------------------------------------------------------
	subroutine tensor_block_copy_(nthreads,dim_num,dim_extents,dim_transp,tens_in,tens_out,ierr) &
                                      bind(c,name='tensor_block_copy__') !PARALLEL
!Given a dense tensor block, this subroutine makes a copy of it, permuting the indices according to the <dim_transp>.
!The algorithm is expected to be cache-efficient (Author: Dmitry I. Lyakh: quant4me@gmail.com).
!In practice, regardless of the index permutation, it should be only two times slower than the direct copy.
!INPUT:
! - nthreads - number of threads requested;
! - dim_num - number of dimensions (>0);
! - dim_extents(1:dim_num) - dimension extents;
! - dim_transp(0:dim_num) - index permutation (old-to-new, sign containing);
! - tens_in(0:) - input tensor block;
!OUTPUT:
! - tens_out(0:) - output (possibly transposed) tensor block;
! - ierr - error code (0:success).
!NOTES:
! - The output tensor block is expected to have the same size as the input tensor block.
	implicit none
!---------------------------------------------------
	logical, parameter:: cache_efficiency=.true.          !turns on/off cache efficiency
	integer(int8_kind), parameter:: cache_line_lim=2**5   !approx. number of simultaneously open cache lines per thread
	integer(int8_kind), parameter:: small_tens_size=2**10 !up to this size of a tensor block it is useless to apply cache efficiency
	integer(int8_kind), parameter:: ave_thread_num=32     !average number of executing threads (approx.)
	integer(int8_kind), parameter:: max_dim_ext=cache_line_lim*ave_thread_num !max acceptable extent of a dimension that will not be split
	integer(int8_kind), parameter:: vec_size=2**4         !loop reorganization parameter
!-----------------------------------------------------
	integer(int_kind), intent(in):: nthreads,dim_num,dim_extents(1:dim_num),dim_transp(0:dim_num)
	real(real8_kind), intent(in):: tens_in(0:*)
	real(real8_kind), intent(out):: tens_out(0:*)
	integer(int_kind), intent(inout):: ierr
	integer(int_kind):: i,j,k,l,m,n,k0,k1,k2,k3,ks,kf
	integer(int_kind):: im(1:max_tensor_rank),n2o(0:max_tensor_rank+1),ipr(1:max_tensor_rank+1)
	integer(int8_kind):: bases_in(1:max_tensor_rank+1),bases_out(1:max_tensor_rank+1),bases_pri(1:max_tensor_rank+1)
	integer(int8_kind):: bs,l0,l1,l2,l3,l_in,l_out,segs(0:max_threads)
	integer(int_kind):: dim_beg(1:dim_num),dim_end(1:dim_num),ac1(1:max_tensor_rank+1),split_in,split_out
	logical:: trivial,in_out_dif
	real(real8_kind):: time_beg

	ierr=0
!	time_beg=thread_wtime() !debug
	if(dim_num.lt.0) then; ierr=dim_num; return; elseif(dim_num.eq.0) then; tens_out(0)=tens_in(0); return; endif
!Check the index permutation:
	trivial=.true.; do i=1,dim_num; if(dim_transp(i).ne.i) then; trivial=.false.; exit; endif; enddo
	if(trivial) then !trivial index permutation
 !Compute indexing bases:
	 n2o(0:dim_num+1)=(/+1,dim_transp(1:dim_num),dim_num+1/)
	 bs=1_8; do i=1,dim_num; bases_in(i)=bs; bs=bs*dim_extents(i); enddo
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(l0,l1) NUM_THREADS(nthreads)
!$OMP DO SCHEDULE(GUIDED)
	 do l0=0_8,bs-1_8-mod(bs,vec_size),vec_size
	  do l1=0_8,vec_size-1_8; tens_out(l0+l1)=tens_in(l0+l1); enddo
	 enddo
!$OMP END DO NOWAIT
!$OMP MASTER
	 do l0=bs-mod(bs,vec_size),bs-1_8; tens_out(l0)=tens_in(l0); enddo
!$OMP END MASTER
!$OMP END PARALLEL
	else !non-trivial index permutation
 !Compute indexing bases:
	 do i=1,dim_num; n2o(dim_transp(i))=i; enddo; n2o(dim_num+1)=dim_num+1 !get N2O
	 bs=1_8; do i=1,dim_num; bases_in(i)=bs; bs=bs*dim_extents(i); enddo
	 bs=1_8; do i=1,dim_num; bases_out(n2o(i))=bs; bs=bs*dim_extents(n2o(i)); enddo
	 bases_in(dim_num+1)=bs; bases_out(dim_num+1)=bs
 !Determine index looping priorities:
	 in_out_dif=.false.; split_in=0; split_out=0
	 if(bs.le.small_tens_size.or.(.not.cache_efficiency)) then !tensor block is too small
	  ipr(1:dim_num+1)=(/(j,j=1,dim_num+1)/); kf=dim_num !trivial priorities
	 else
	  do k1=2,dim_num; if(bases_in(k1).ge.cache_line_lim) exit; enddo; k1=k1-1 !first k1 input dimensions form the input minor set
	  ipr(1:k1)=(/(j,j=1,k1)/); n=k1 !first k1 input dimensions form the input minor set
	  do j=1,k1; if(dim_transp(j).gt.k1) then; in_out_dif=.true.; exit; endif; enddo !if .true., the output minor set differs from the input one
	  if(in_out_dif) then !check whether I need to split long ranges
	   if(dim_extents(k1).gt.max_dim_ext) split_in=k1 !input dimension which will be split
	   do k2=2,dim_num; if(bases_out(n2o(k2)).ge.cache_line_lim) exit; enddo; k2=k2-1 !first k2 output dimensions form the output minor set
	   if(dim_extents(n2o(k2)).gt.max_dim_ext) split_out=n2o(k2) !output dimension which will be split
	   if(split_out.eq.split_in) split_out=0 !input and output split dimensions coincide
	  else
	   k2=k1
	  endif
	  kf=k1; do j=1,k2; if(n2o(j).gt.k1) then; n=n+1; ipr(n)=n2o(j); kf=kf+1; endif; enddo !ipr(priority) = old_num: dimension looping priorities
	  do j=k2+1,dim_num; if(n2o(j).gt.k1) then; n=n+1; ipr(n)=n2o(j); endif; enddo !kf is the length of the combined minor set
	  ipr(dim_num+1)=dim_num+1 !special setting
	 endif
	 do i=1,dim_num; ac1(i)=n2o(dim_transp(i)+1); enddo !accelerator array
!	 write(*,'("DEBUG(tensor_algebra_dil::tensor_block_copy_): block size, split dims = ",i10,3x,i2,1x,i2)') &
!         bs,split_in,split_out !debug
!	 write(*,'("DEBUG(tensor_algebra_dil::tensor_block_copy_): index extents =",128(1x,i2))') &
!         dim_extents(1:dim_num) !debug
!	 write(*,'("DEBUG(tensor_algebra_dil::tensor_block_copy_): index permutation =",128(1x,i2))') &
!         dim_transp(1:dim_num) !debug
!	 write(*,'("DEBUG(tensor_algebra_dil::tensor_block_copy_): index priorities = ",i3,1x,l1,128(1x,i2))') &
!         kf,in_out_dif,ipr(1:dim_num) !debug
 !Transpose loop:
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,j,m,n,ks,im,l_in,l_out,l0,l1,l2,l3,dim_beg,dim_end) NUM_THREADS(nthreads)
#ifndef NO_OMP
	 n=omp_get_thread_num(); m=omp_get_num_threads() !multi-threaded execution: m is the real number of threads running in parallel
#else
	 n=0; m=1 !serial execution
#endif
!	 if(n.eq.0) write(*,'("DEBUG(tensor_algebra_dil::tensor_block_copy_): total number of threads = ",i4)') m !debug
	 if(.not.in_out_dif) then !input minor set coincides with the output minor set: no splitting
!$OMP MASTER
	  segs(0)=0_8; call divide_segment_(bs,int(m,8),segs(1:),ierr); do j=2,m; segs(j)=segs(j)+segs(j-1); enddo
	  l0=1_8; do i=1,dim_num; bases_pri(ipr(i))=l0; l0=l0*dim_extents(ipr(i)); enddo !priority bases
!$OMP END MASTER
!$OMP BARRIER
!$OMP FLUSH(segs,bases_pri)
	  l0=segs(n); do i=dim_num,1,-1; j=ipr(i); im(j)=l0/bases_pri(j); l0=mod(l0,bases_pri(j)); enddo !initial multiindex for each thread
	  l_in=0_8; do j=1,dim_num; l_in=l_in+im(j)*bases_in(j); enddo !initital input address
	  l_out=0_8; do j=1,dim_num; l_out=l_out+im(j)*bases_out(j); enddo !initial output address
	  do l0=segs(n),segs(n+1)-1_8,cache_line_lim
	   loop0: do l1=l0,min(l0+cache_line_lim-1_8,segs(n+1)-1_8)
	    tens_out(l_out)=tens_in(l_in)
  !Increament of the multi-index (scheme 1):
	    do i=1,dim_num
	     j=ipr(i)
	     if(im(j)+1.eq.dim_extents(j)) then
	      l_in=l_in+bases_in(j)-bases_in(j+1); l_out=l_out+bases_out(j)-bases_out(ac1(j)); im(j)=0
	     else
	      im(j)=im(j)+1; l_in=l_in+bases_in(j); l_out=l_out+bases_out(j)
	      exit
	     endif
	    enddo !i
	   enddo loop0 !l1
	  enddo !l0
	 else !input and output minor sets differ: range splitting possible
	  if(split_in.gt.0.and.split_out.eq.0) then !split the last dimension from the input minor set
	   dim_beg(1:dim_num)=0; dim_end(1:dim_num)=dim_extents(1:dim_num)-1; im(1:dim_num)=dim_beg(1:dim_num)
	   l1=dim_extents(split_in)-1_8
!$OMP DO SCHEDULE(GUIDED)
	   do l0=0_8,l1,cache_line_lim
	    dim_beg(split_in)=int(l0,4); dim_end(split_in)=int(min(l0+cache_line_lim-1_8,l1),4); ks=0
	    im(split_in)=dim_beg(split_in); l_in=im(split_in)*bases_in(split_in); l_out=im(split_in)*bases_out(split_in)
	    loop2: do
	     tens_out(l_out)=tens_in(l_in)
	     do i=1,dim_num
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
!$OMP END DO
	  elseif(split_in.eq.0.and.split_out.gt.0) then !split the last dimension from the output minor set
	   dim_beg(1:dim_num)=0; dim_end(1:dim_num)=dim_extents(1:dim_num)-1; im(1:dim_num)=dim_beg(1:dim_num)
           l1=dim_extents(split_out)-1_8
!$OMP DO SCHEDULE(GUIDED)
	   do l0=0_8,l1,cache_line_lim
	    dim_beg(split_out)=int(l0,4); dim_end(split_out)=int(min(l0+cache_line_lim-1_8,l1),4); ks=0
	    im(split_out)=dim_beg(split_out); l_in=im(split_out)*bases_in(split_out); l_out=im(split_out)*bases_out(split_out)
	    loop3: do
	     tens_out(l_out)=tens_in(l_in)
	     do i=1,dim_num
	      j=ipr(i) !old index number
	      if(im(j).lt.dim_end(j)) then
	       im(j)=im(j)+1; l_in=l_in+bases_in(j); l_out=l_out+bases_out(j)
	       ks=ks+1; exit
	      else
	       l_in=l_in-(im(j)-dim_beg(j))*bases_in(j); l_out=l_out-(im(j)-dim_beg(j))*bases_out(j); im(j)=dim_beg(j)
	      endif
	     enddo !i
	     ks=ks-1; if(ks.lt.0) exit loop3
	    enddo loop3
	   enddo !l0
!$OMP END DO
	  elseif(split_in.gt.0.and.split_out.gt.0) then !split the last dimensions from both the input and output minor sets
	   dim_beg(1:dim_num)=0; dim_end(1:dim_num)=dim_extents(1:dim_num)-1; im(1:dim_num)=dim_beg(1:dim_num)
           l2=dim_end(split_in); l3=dim_end(split_out)
!$OMP DO SCHEDULE(GUIDED) COLLAPSE(2)
	   do l0=0_8,l2,cache_line_lim !input dimension
	    do l1=0_8,l3,cache_line_lim !output dimension
	     dim_beg(split_in)=int(l0,4); dim_end(split_in)=int(min(l0+cache_line_lim-1_8,l2),4)
	     dim_beg(split_out)=int(l1,4); dim_end(split_out)=int(min(l1+cache_line_lim-1_8,l3),4)
	     im(split_in)=dim_beg(split_in); im(split_out)=dim_beg(split_out)
	     l_in=im(split_in)*bases_in(split_in)+im(split_out)*bases_in(split_out)
	     l_out=im(split_in)*bases_out(split_in)+im(split_out)*bases_out(split_out); ks=0
	     loop4: do
	      tens_out(l_out)=tens_in(l_in)
	      do i=1,dim_num
	       j=ipr(i) !old index number
	       if(im(j).lt.dim_end(j)) then
	        im(j)=im(j)+1; l_in=l_in+bases_in(j); l_out=l_out+bases_out(j)
	        ks=ks+1; exit
	       else
	        l_in=l_in-(im(j)-dim_beg(j))*bases_in(j); l_out=l_out-(im(j)-dim_beg(j))*bases_out(j); im(j)=dim_beg(j)
	       endif
	      enddo !i
	      ks=ks-1; if(ks.lt.0) exit loop4
	     enddo loop4
	    enddo !l1
	   enddo !l0
!$OMP END DO
	  else !no range splitting
!$OMP MASTER
	   segs(0)=0_8; call divide_segment_(bs,int(m,8),segs(1:),ierr); do j=2,m; segs(j)=segs(j)+segs(j-1); enddo
	   l0=1_8; do i=1,dim_num; bases_pri(ipr(i))=l0; l0=l0*dim_extents(ipr(i)); enddo !priority bases
!$OMP END MASTER
!$OMP BARRIER
!$OMP FLUSH(segs,bases_pri)
	   l0=segs(n); do i=dim_num,1,-1; j=ipr(i); im(j)=l0/bases_pri(j); l0=mod(l0,bases_pri(j)); enddo
	   l_in=0_8; do j=1,dim_num; l_in=l_in+im(j)*bases_in(j); enddo
	   l_out=0_8; do j=1,dim_num; l_out=l_out+im(j)*bases_out(j); enddo
	   do l0=segs(n),segs(n+1)-1_8,cache_line_lim
	    loop1: do l1=l0,min(l0+cache_line_lim-1_8,segs(n+1)-1_8)
	     tens_out(l_out)=tens_in(l_in)
  !Increament of the multi-index (scheme 2):
	     do i=1,dim_num
	      j=ipr(i) !old index number
	      if(im(j)+1.lt.dim_extents(j)) then
	       im(j)=im(j)+1; l_in=l_in+bases_in(j); l_out=l_out+bases_out(j)
	       exit
	      else
	       l_in=l_in-im(j)*bases_in(j); l_out=l_out-im(j)*bases_out(j); im(j)=0
	      endif
	     enddo !i
	    enddo loop1 !l1
	   enddo !l0
	  endif
	 endif
!$OMP END PARALLEL
	endif !trivial or not
!	write(*,'("DEBUG(tensor_algebra_dil::tensor_block_copy_): exit code / kernel time = ",i5,1x,F10.4)') &
!        ierr,thread_wtime(time_beg) !debug
	return
	end subroutine tensor_block_copy_
!-------------------------------------------------------------------------------------------------------
	subroutine tensor_block_contract_(nthreads,contr_ptrn,ltens,ltens_num_dim,ltens_dim_extent, &   !PARALLEL
	                                                      rtens,rtens_num_dim,rtens_dim_extent, &
	                                                      dtens,dtens_num_dim,dtens_dim_extent,ierr) &
                                          bind(c,name='tensor_block_contract__')
!This subroutine contracts two arbitrary dense tensor blocks and puts the result into another tensor block:
!dtens(:)=ltens(:)*rtens(:)
!The algorithm is based on cache-efficient tensor transposes (Author: Dmitry I. Lyakh: quant4me@gmail.com)
!Possible cases:
! A) tensor+=tensor*tensor (no traces!): all tensor operands can be transposed;
! B) scalar+=tensor*tensor (no traces!): only the left tensor operand can be transposed;
! C) tensor+=tensor*scalar OR tensor+=scalar*tensor (no traces!): no transpose;
! D) scalar+=scalar*scalar: no transpose.
!INPUT:
! - nthreads - number of threads requested;
! - contr_ptrn(1:left_rank+right_rank) - contraction pattern;
! - ltens/ltens_num_dim/ltens_dim_extent - left tensor block / its rank / and dimension extents;
! - rtens/rtens_num_dim/rtens_dim_extent - right tensor block / its rank / and dimension extents;
! - dtens/dtens_num_dim/dtens_dim_extent - destination tensor block (initialized!) / its rank / and dimension extents;
!OUTPUT:
! - dtens - modified destination tensor block;
! - ierr - error code (0: success);
	implicit none
!-----------------------------------------------
	logical, parameter:: disable_blas=.false. !if .true. && BLAS is accessible, BLAS calls will be replaced by my own routines
!-----------------------------------------------
	integer(int_kind), intent(in):: nthreads,contr_ptrn(1:*)
	real(real8_kind), intent(in), target:: ltens(0:*),rtens(0:*)
	integer(int_kind), intent(in):: ltens_num_dim,ltens_dim_extent(1:ltens_num_dim)
	integer(int_kind), intent(in):: rtens_num_dim,rtens_dim_extent(1:rtens_num_dim)
	integer(int_kind), intent(in):: dtens_num_dim,dtens_dim_extent(1:dtens_num_dim)
	real(real8_kind), intent(inout), target:: dtens(0:*)
	integer(int_kind), intent(inout):: ierr
!-----------------------------------------------
	integer(int_kind):: i,j,k,l,m,n,k0,k1,k2,k3,ks,kf
	integer(int8_kind):: l0,l1,l2,l3,lld,lrd,lcd,lsize,rsize,dsize
	integer(int_kind):: lo2n(0:max_tensor_rank),ro2n(0:max_tensor_rank),do2n(0:max_tensor_rank),dn2o(0:max_tensor_rank)
        integer(int_kind):: nlu,nru,ncd
	real(real8_kind), pointer:: ltp(:),rtp(:),dtp(:)
	real(real8_kind):: d_scalar
	logical:: contr_ok,ltransp,rtransp,dtransp
	real(real8_kind):: time_beg,time_p,time_perm,time_mult,start_dgemm

	ierr=0
!	time_beg=thread_wtime(); time_perm=0d0; time_mult=0d0 !debug
	if(ltens_num_dim.ge.0.and.ltens_num_dim.le.max_tensor_rank.and. &
	   rtens_num_dim.ge.0.and.rtens_num_dim.le.max_tensor_rank.and. &
	   dtens_num_dim.ge.0.and.dtens_num_dim.le.max_tensor_rank) then
	 ltransp=.false.; rtransp=.false.; dtransp=.false.
!Check the requested contraction pattern:
	 contr_ok=contr_ptrn_ok(contr_ptrn,ltens_num_dim,rtens_num_dim,dtens_num_dim)
	 if(.not.contr_ok) then; ierr=1; goto 999; endif
!	 write(*,'("DEBUG(tensor_algebra_dil::tensor_block_contract_): contraction pattern accepted:",128(1x,i2))') &
!         contr_ptrn(1:ltens_num_dim+rtens_num_dim) !debug
!	 write(*,'("DEBUG(tensor_algebra_dil::tensor_block_contract_): left index extents  :",128(1x,i4))') &
!         ltens_dim_extent(1:ltens_num_dim) !debug
!	 write(*,'("DEBUG(tensor_algebra_dil::tensor_block_contract_): right index extents :",128(1x,i4))') &
!         rtens_dim_extent(1:rtens_num_dim) !debug
!	 write(*,'("DEBUG(tensor_algebra_dil::tensor_block_contract_): result index extents:",128(1x,i4))') &
!         dtens_dim_extent(1:dtens_num_dim) !debug
!Determine index permutations for all tensor operands together with the numbers of contracted/uncontraced indices (ncd/{nlu,nru}):
	 call determine_index_permutations !sets {dtransp,ltransp,rtransp},{do2n,lo2n,ro2n},{lsize,rsize,dsize},{ncd,nlu,nru},{lld,lrd,lcd}
!	 write(*,'("DEBUG(tensor_algebra_dil::tensor_block_contract_): left uncontr, right uncontr, contr dims: ",i2,1x,i2,1x,i2)')&
!         nlu,nru,ncd !debug
!	 write(*,'("DEBUG(tensor_algebra_dil::tensor_block_contract_): left index permutation (O2N)           :",128(1x,i2))') &
!         lo2n(1:ltens_num_dim) !debug
!	 write(*,'("DEBUG(tensor_algebra_dil::tensor_block_contract_): right index permutation (O2N)          :",128(1x,i2))') &
!         ro2n(1:rtens_num_dim) !debug
!	 write(*,'("DEBUG(tensor_algebra_dil::tensor_block_contract_): destination index permutation (O2N)    :",128(1x,i2))') &
!         do2n(1:dtens_num_dim) !debug
!	 write(*,'("DEBUG(tensor_algebra_dil::tensor_block_contract_): sizes of the matrices(left,right,dest) : "&
!         &,i10,1x,i10,1x,i10)') lsize,rsize,dsize !debug
!	 write(*,'("DEBUG(tensor_algebra_dil::tensor_block_contract_): matrix dimensions (left,right,contr)   : "&
!         &,i10,1x,i10,1x,i10)') lld,lrd,lcd !debug
!Transpose tensor arguments, if needed:
!	 time_p=thread_wtime() !debug
 !Left tensor argument:
	 if(ltransp) then
	  allocate(ltp(1:lsize),STAT=ierr); if(ierr.ne.0) goto 999
	  call tensor_block_copy_(nthreads,ltens_num_dim,ltens_dim_extent,lo2n,ltens,ltp,ierr); if(ierr.ne.0) goto 999
	 else
	  ltp=>ltens(0:lsize-1) !ltp(1:lsize)
	 endif
 !Right tensor argument:
	 if(rtransp) then
	  allocate(rtp(1:rsize),STAT=ierr); if(ierr.ne.0) goto 999
	  call tensor_block_copy_(nthreads,rtens_num_dim,rtens_dim_extent,ro2n,rtens,rtp,ierr); if(ierr.ne.0) goto 999
	 else
	  rtp=>rtens(0:rsize-1) !rtp(1:rsize)
	 endif
 !Destination tensor association:
	 if(dtransp) then
	  allocate(dtp(1:dsize),STAT=ierr); if(ierr.ne.0) goto 999
	  dn2o(0)=+1; do k=1,dtens_num_dim; dn2o(do2n(k))=k; enddo
!	  call tensor_block_copy_(nthreads,dtens_num_dim,dtens_dim_extent,dn2o,dtens,dtp,ierr); if(ierr.ne.0) goto 999
	 else
	  dtp=>dtens(0:dsize-1) !dtp(1:dsize)
	 endif
	 do k=1,dsize; dtp(k)=0d0; enddo !remove this if accumulation is desired, uncomment call tensor_block_copy_ above!
!	 time_perm=time_perm+thread_wtime(time_p) !debug
!Multiply two matrices (dtp+=ltp*rtp):
!	 time_p=thread_wtime() !debug
	 if(dtens_num_dim.gt.0.and.ltens_num_dim.gt.0.and.rtens_num_dim.gt.0) then !partial contraction
#ifdef NO_BLAS
	  call tensor_block_pcontract(nthreads,lld,lrd,lcd,ltp,rtp,dtp,ierr); if(ierr.ne.0) goto 999
#else
	  if(.not.disable_blas) then
!	   start_dgemm=thread_wtime() !debug
	   call dgemm('T','N',int(lld,4),int(lrd,4),int(lcd,4),1d0,ltp,int(lcd,4),rtp,int(lcd,4),1d0,dtp,int(lld,4))
!	   write(*,'("DEBUG(tensor_algebra_dil::tensor_block_contract_): DGEMM time: ",F10.6)') thread_wtime(start_dgemm) !debug
	  else
	   call tensor_block_pcontract(nthreads,lld,lrd,lcd,ltp,rtp,dtp,ierr); if(ierr.ne.0) goto 999
	  endif
#endif
	 elseif(dtens_num_dim.eq.0.and.ltens_num_dim.gt.0.and.rtens_num_dim.gt.0) then !full contraction
          dtp(1)=0d0 !remove this if accumulation is desired!
	  call tensor_block_fcontract(nthreads,lcd,ltp,rtp,dtp(1),ierr); if(ierr.ne.0) goto 999
	 elseif(dtens_num_dim.gt.0.and.ltens_num_dim.gt.0.and.rtens_num_dim.eq.0) then !right scalar multiplication
	  call tensor_block_add_(nthreads,dtens_num_dim,dtens_dim_extent,dtp,ltp,rtp(1),ierr); if(ierr.ne.0) goto 999
	 elseif(dtens_num_dim.gt.0.and.ltens_num_dim.eq.0.and.rtens_num_dim.gt.0) then !left scalar multiplication
	  call tensor_block_add_(nthreads,dtens_num_dim,dtens_dim_extent,dtp,rtp,ltp(1),ierr); if(ierr.ne.0) goto 999
	 elseif(dtens_num_dim.eq.0.and.ltens_num_dim.eq.0.and.rtens_num_dim.eq.0) then !scalar multiplication
!	  dtp(1)=dtp(1)+ltp(1)*rtp(1)
          dtp(1)=ltp(1)*rtp(1) !remove this if accumulation is desired, uncomment above!
	 endif
!	 time_mult=time_mult+thread_wtime(time_p) !debug
!Transpose the matrix-result into the output tensor:
!	 time_p=thread_wtime() !debug
	 if(dtransp) then
	  call tensor_block_copy_(nthreads,dtens_num_dim,dtens_dim_extent(do2n(1:dtens_num_dim)),do2n,dtp,dtens,ierr)
	  if(ierr.ne.0) goto 999
	 endif
!	 time_perm=time_perm+thread_wtime(time_p) !debug
!Destroy temporary arrays and pointers:
999	 if(ltransp) then; deallocate(ltp); else; if(associated(ltp)) nullify(ltp); endif
	 if(rtransp) then; deallocate(rtp); else; if(associated(rtp)) nullify(rtp); endif
	 if(dtransp) then; deallocate(dtp); else; if(associated(dtp)) nullify(dtp); endif
	else
	 ierr=-1 !invalid tensor ranks
	endif
!	write(*,'("DEBUG(tensor_algebra_dil::tensor_block_contract_): exit error code / timings: ",i5,3(1x,F10.6))') &
!        ierr,thread_wtime(time_beg),time_perm,time_mult !debug
	return
	contains

	 subroutine determine_index_permutations !sets {dtransp,ltransp,rtransp},{do2n,lo2n,ro2n},{lsize,rsize,dsize},{ncd,nlu,nru},{lld,lrd,lcd}
	 integer:: jkey(1:max_tensor_rank),jtrn0(0:max_tensor_rank),jtrn1(0:max_tensor_rank),jj,j0,j1
	 lld=1_8; lrd=1_8; lcd=1_8 !{left, right, contracted} dimension extents
 !Destination operand:
	 dsize=1_8 !destination matrix size
	 if(dtens_num_dim.gt.0) then
	  do j0=1,dtens_num_dim; dsize=dsize*dtens_dim_extent(j0); enddo
	  do2n(0)=+1; j1=0
	  do j0=1,ltens_num_dim+rtens_num_dim
	   if(contr_ptrn(j0).gt.0) then; j1=j1+1; do2n(j1)=contr_ptrn(j0); endif
	  enddo
	  if(perm_trivial(j1,do2n)) then; dtransp=.false.; else; dtransp=.true.; endif
	 else
	  dtransp=.false.
	 endif
 !Right tensor operand:
	 rsize=1_8; nru=0; ncd=0 !right matrix size, numbers of the right_uncontracted and contracted dimensions
	 if(rtens_num_dim.gt.0) then
	  do j0=1,rtens_num_dim; rsize=rsize*rtens_dim_extent(j0); enddo
	  ro2n(0)=+1; j1=0
	  do j0=1,rtens_num_dim
	   if(contr_ptrn(ltens_num_dim+j0).lt.0) then
	    j1=j1+1; ro2n(j0)=j1; lcd=lcd*rtens_dim_extent(j0)
	   endif
	  enddo
	  ncd=j1 !number of contracted dimensions
	  do j0=1,rtens_num_dim
	   if(contr_ptrn(ltens_num_dim+j0).gt.0) then
	    j1=j1+1; ro2n(j0)=j1; lrd=lrd*rtens_dim_extent(j0)
	   endif
	  enddo
	  nru=j1-ncd !number of uncontracted dimensions
	  if(perm_trivial(j1,ro2n)) then; rtransp=.false.; else; rtransp=.true.; endif
	 else
	  rtransp=.false.
	 endif
 !Left tensor operand:
	 lsize=1_8; nlu=0 !left matrix size, number of the left_uncontracted dimensions
	 if(ltens_num_dim.gt.0) then
	  do j0=1,ltens_num_dim; lsize=lsize*ltens_dim_extent(j0); enddo
	  lo2n(0)=+1; j1=0
	  do j0=1,ltens_num_dim
	   if(contr_ptrn(j0).lt.0) then
	    j1=j1+1; jtrn1(j1)=j0; jkey(j1)=abs(contr_ptrn(j0))
	   endif
	  enddo
	  ncd=j1 !contracted dimensions
	  jtrn0(0:j1)=(/+1,(jj,jj=1,j1)/); call merge_sort_key_int_(j1,jkey,jtrn0) !align contracted dimensions according to the right tensor operand
	  do j0=1,j1; jj=jtrn0(j0); lo2n(jtrn1(jj))=j0; enddo !contracted dimensions of the left operand are aligned to the corresponding dimensions of the right operand
	  do j0=1,ltens_num_dim
	   if(contr_ptrn(j0).gt.0) then
	    j1=j1+1; lo2n(j0)=j1; lld=lld*ltens_dim_extent(j0)
	   endif
	  enddo
	  nlu=j1-ncd !uncontracted dimensions
	  if(perm_trivial(j1,lo2n)) then; ltransp=.false.; else; ltransp=.true.; endif
	 else
	  ltransp=.false.
	 endif
	 return
	 end subroutine determine_index_permutations

	 logical function contr_ptrn_ok(ptrn,lr,rr,dr)
	 integer, intent(in):: ptrn(1:*),lr,rr,dr
	 integer:: j0,j1,jl,jbus(dr+lr+rr)
	 contr_ptrn_ok=.true.; jl=dr+lr+rr
	 if(jl.gt.0) then
	  jbus(1:jl)=0
	  do j0=1,lr !left tensor-argument
	   j1=ptrn(j0)
	   if(j1.gt.0.and.j1.le.dr) then !uncontracted index
	    jbus(j1)=jbus(j1)+1; jbus(dr+j0)=jbus(dr+j0)+1
	    if(ltens_dim_extent(j0).ne.dtens_dim_extent(j1)) then; contr_ptrn_ok=.false.; return; endif
	   elseif(j1.lt.0.and.abs(j1).le.rr) then !contracted index
	    jbus(dr+lr+abs(j1))=jbus(dr+lr+abs(j1))+1
	    if(ltens_dim_extent(j0).ne.rtens_dim_extent(abs(j1))) then; contr_ptrn_ok=.false.; return; endif
	   else
	    contr_ptrn_ok=.false.; return
	   endif
	  enddo
	  do j0=lr+1,lr+rr !right tensor-argument
	   j1=ptrn(j0)
	   if(j1.gt.0.and.j1.le.dr) then !uncontracted index
	    jbus(j1)=jbus(j1)+1; jbus(dr+j0)=jbus(dr+j0)+1
	    if(rtens_dim_extent(j0-lr).ne.dtens_dim_extent(j1)) then; contr_ptrn_ok=.false.; return; endif
	   elseif(j1.lt.0.and.abs(j1).le.lr) then !contracted index
	    jbus(dr+abs(j1))=jbus(dr+abs(j1))+1
	    if(rtens_dim_extent(j0-lr).ne.ltens_dim_extent(abs(j1))) then; contr_ptrn_ok=.false.; return; endif
	   else
	    contr_ptrn_ok=.false.; return
	   endif
	  enddo
	  do j0=1,jl
	   if(jbus(j0).ne.1) then; contr_ptrn_ok=.false.; return; endif
	  enddo
	 endif
	 return
	 end function contr_ptrn_ok

         logical function perm_trivial(ni,trn)
         integer, intent(in):: ni,trn(0:*)
         integer:: j0
         perm_trivial=.true.
         do j0=1,ni; if(trn(j0).ne.j0) then; perm_trivial=.false.; exit; endif; enddo
         return
         end function perm_trivial

	end subroutine tensor_block_contract_
!----------------------------------------------------------------------------
!PRIVATE SUBROUTINES (no C interfaces):
!----------------------------------------------------------------------------
	subroutine tensor_block_fcontract(nthreads,dc,ltens,rtens,dtens,ierr) !PARALLEL
!This subroutine fully reduces two vectors derived from the corresponding tensors by index permutations:
!dtens+=ltens(0:dc-1)*rtens(0:dc-1), where dtens is a scalar.
	implicit none
	integer(int_kind), intent(in):: nthreads
	integer(int8_kind), intent(in):: dc
	real(real8_kind), intent(in):: ltens(0:*),rtens(0:*) !true tensors
	real(real8_kind), intent(inout):: dtens !scalar
	integer(int_kind), intent(inout):: ierr
	integer(int_kind):: i,j,k,l,m,n
	integer(int8_kind):: l0
	real(real8_kind):: val
	real(real8_kind):: time_beg

	ierr=0
!	time_beg=thread_wtime() !debug
!	write(*,'("DEBUG(tensor_algebra_dil::tensor_block_fcontract): dc: ",i9)') dc !debug
	if(dc.gt.0_8) then
	 val=0d0
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED) REDUCTION(+:val) NUM_THREADS(nthreads)
	 do l0=0_8,dc-1_8; val=val+ltens(l0)*rtens(l0); enddo
!$OMP END PARALLEL DO
	 dtens=dtens+val
	else
	 ierr=1
	endif
!	write(*,'("DEBUG(tensor_algebra_dil::tensor_block_fcontract): exit code / kernel time: ",i5,1x,F10.4)') &
!        ierr,thread_wtime(time_beg) !debug
	return
	end subroutine tensor_block_fcontract
!----------------------------------------------------------------------------------
	subroutine tensor_block_pcontract(nthreads,dl,dr,dc,ltens,rtens,dtens,ierr) !PARALLEL
!This subroutine multiplies two matrices derived from the corresponding tensors by index permutations:
!dtens(0:dl-1,0:dr-1)+=ltens(0:dc-1,0:dl-1)*rtens(0:dc-1,0:dr-1)
!The result is a matrix as well (cannot be a scalar, see tensor_block_fcontract).
	implicit none
!-------------------------------------------------------
	integer(int8_kind), parameter:: red_mat_size=32       !the dimension of the local reduction matrix
	integer(int8_kind), parameter:: arg_cache_size=2**16  !cache-size dependent parameter (increase it as there are more cores per node)
	integer(int_kind), parameter:: min_distr_seg_size=128 !min segment size of an omp distributed dimension
	integer(int_kind), parameter:: cdim_stretch=4         !makes the segmentation of the contracted dimension coarser
	integer(int_kind), parameter:: core_slope=16          !regulates the slope of the segment size of the distributed dimension w.r.t. the number of cores
	integer(int_kind), parameter:: ker1=0                 !kernel 1 scheme #
	integer(int_kind), parameter:: ker2=0                 !kernel 2 scheme #
	integer(int_kind), parameter:: ker3=0                 !kernel 3 scheme #
!-------------------------------------------------------
	logical, parameter:: no_case1=.false.
	logical, parameter:: no_case2=.false.
	logical, parameter:: no_case3=.false.
	logical, parameter:: no_case4=.false.
!-------------------------------------------------------
	integer(int_kind), intent(in):: nthreads
	integer(int8_kind), intent(in):: dl,dr,dc !matrix dimensions
	real(real8_kind), intent(in):: ltens(0:*),rtens(0:*)
	real(real8_kind), intent(inout):: dtens(0:*)
	integer(int_kind), intent(inout):: ierr
	integer(int_kind):: i,j,k,l,m,n,nthr
	integer(int8_kind):: l0,l1,l2,ll,lr,ld,ls,lf,b0,b1,b2,e0,e1,e2,cl,cr,cc,chunk
	real(real8_kind):: val,redm(0:red_mat_size-1,0:red_mat_size-1)
	real(real8_kind):: time_beg

	ierr=0
!	time_beg=thread_wtime() !debug
	if(dl.gt.0_8.and.dr.gt.0_8.and.dc.gt.0_8) then
#ifndef NO_OMP
	 nthr=min(omp_get_max_threads(),nthreads)
#else
	 nthr=1
#endif
	 if(dr.ge.core_slope*nthr.and.(.not.no_case1)) then !the right dimension is large enough to be distributed
!	  write(*,'("DEBUG(tensor_algebra_dil::tensor_block_pcontract): kernel case/scheme: 1/",i1)') ker1 !debug
	  select case(ker1)
	  case(0)
!SCHEME 0:
	   cr=min(dr,max(core_slope*nthr,min_distr_seg_size))
	   cc=min(dc,max(arg_cache_size*cdim_stretch/cr,1_8))
	   cl=min(dl,min(max(arg_cache_size/cc,1_8),max(arg_cache_size/cr,1_8)))
!	   write(*,'("DEBUG(tensor_algebra_dil::tensor_block_pcontract): cl,cr,cc,dl,dr,dc:",6(1x,i9))') cl,cr,cc,dl,dr,dc !debug
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(b0,b1,b2,e0,e1,e2,l0,l1,l2,ll,lr,ld,val) NUM_THREADS(nthreads)
	   do b0=0_8,dc-1_8,cc
	    e0=min(b0+cc-1_8,dc-1_8)
	    do b1=0_8,dl-1_8,cl
	     e1=min(b1+cl-1_8,dl-1_8)
	     do b2=0_8,dr-1_8,cr
	      e2=min(b2+cr-1_8,dr-1_8)
!$OMP DO SCHEDULE(GUIDED)
	      do l2=b2,e2
	       lr=l2*dc; ld=l2*dl
	       do l1=b1,e1
	        ll=l1*dc
	        val=dtens(ld+l1)
	        do l0=b0,e0; val=val+ltens(ll+l0)*rtens(lr+l0); enddo
	        dtens(ld+l1)=val
	       enddo
	      enddo
!$OMP END DO NOWAIT
	     enddo
	    enddo
!$OMP BARRIER
	   enddo
!$OMP END PARALLEL
	  case(1)
!SCHEME 1:
	   chunk=max(arg_cache_size/dc,1_8)
!           write(*,'("DEBUG(tensor_algebra_dil::tensor_block_pcontract): chunk,dl,dr,dc:",4(1x,i9))') chunk,dl,dr,dc !debug
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(l0,l1,l2,ll,lr,ld,ls,lf,val) NUM_THREADS(nthreads)
	   do ls=0_8,dl-1_8,chunk
	    lf=min(ls+chunk-1_8,dl-1_8)
!!$OMP DO SCHEDULE(DYNAMIC,chunk)
!$OMP DO SCHEDULE(GUIDED)
	    do l2=0_8,dr-1_8
	     lr=l2*dc; ld=l2*dl
	     do l1=ls,lf
	      ll=l1*dc
	      val=dtens(ld+l1)
	      do l0=0_8,dc-1_8; val=val+ltens(ll+l0)*rtens(lr+l0); enddo
	      dtens(ld+l1)=val
	     enddo
	    enddo
!$OMP END DO NOWAIT
	   enddo
!$OMP END PARALLEL
	  case(2)
!SCHEME 2:
	   chunk=max(arg_cache_size/dc,1_8)
	   if(mod(dl,chunk).ne.0) then; ls=dl/chunk+1_8; else; ls=dl/chunk; endif
	   if(mod(dr,chunk).ne.0) then; lf=dr/chunk+1_8; else; lf=dr/chunk; endif
!	   write(*,'("DEBUG(tensor_algebra_dil::tensor_block_pcontract): chunk,ls,lf,dl,dr,dc:",6(1x,i9))') chunk,ls,lf,dl,dr,dc !debug
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(b0,b1,b2,l0,l1,l2,ll,lr,ld,val) SCHEDULE(GUIDED) NUM_THREADS(nthreads)
	   do b0=0_8,lf*ls-1_8
	    b2=b0/ls*chunk; b1=mod(b0,ls)*chunk
	    do l2=b2,min(b2+chunk-1_8,dr-1_8)
	     lr=l2*dc; ld=l2*dl
	     do l1=b1,min(b1+chunk-1_8,dl-1_8)
	      ll=l1*dc
	      val=dtens(ld+l1)
	      do l0=0_8,dc-1_8; val=val+ltens(ll+l0)*rtens(lr+l0); enddo
	      dtens(ld+l1)=val
	     enddo
	    enddo
	   enddo
!$OMP END PARALLEL DO
	  case(3)
!SCHEME 3:
!	   write(*,'("DEBUG(tensor_algebra_dil::tensor_block_pcontract): dl,dr,dc:",3(1x,i9))') dl,dr,dc !debug
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0,l1,l2,ll,lr,ld,val) SCHEDULE(DYNAMIC) NUM_THREADS(nthreads)
	   do l2=0_8,dr-1_8
	    lr=l2*dc; ld=l2*dl
	    do l1=0_8,dl-1_8
	     ll=l1*dc
	     val=dtens(ld+l1)
	     do l0=0_8,dc-1_8; val=val+ltens(ll+l0)*rtens(lr+l0); enddo
	     dtens(ld+l1)=val
	    enddo
	   enddo
!$OMP END PARALLEL DO
	  case default
	   ierr=-1
	  end select
	 else !dr is small
	  if(dl.ge.core_slope*nthr.and.(.not.no_case2)) then !the left dimension is large enough to be distributed
!	   write(*,'("DEBUG(tensor_algebra_dil::tensor_block_pcontract): kernel case/scheme: 2/",i1)') ker2 !debug
	   select case(ker2)
	   case(0)
!SCHEME 0:
!            write(*,'("DEBUG(tensor_algebra_dil::tensor_block_pcontract): dl,dr,dc:",3(1x,i9))') dl,dr,dc !debug
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(l0,l1,l2,ll,lr,ld,val) NUM_THREADS(nthreads)
	    do l2=0_8,dr-1_8
	     lr=l2*dc; ld=l2*dl
!$OMP DO SCHEDULE(GUIDED)
	     do l1=0_8,dl-1_8
	      ll=l1*dc
	      val=dtens(ld+l1)
	      do l0=0_8,dc-1_8; val=val+ltens(ll+l0)*rtens(lr+l0); enddo
	      dtens(ld+l1)=val
	     enddo
!$OMP END DO NOWAIT
	    enddo
!$OMP END PARALLEL
	   case(1)
!SCHEME 1:
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0,l1,l2,ll,lr,val) SCHEDULE(GUIDED) COLLAPSE(2) NUM_THREADS(nthreads)
	    do l2=0_8,dr-1_8
	     do l1=0_8,dl-1_8
	      ll=l1*dc; lr=l2*dc
	      val=dtens(l2*dl+l1)
	      do l0=0_8,dc-1_8; val=val+ltens(ll+l0)*rtens(lr+l0); enddo
	      dtens(l2*dl+l1)=val
	     enddo
	    enddo
!$OMP END PARALLEL DO
	   case default
	    ierr=-2
	   end select
	  else !dr & dl are both small
	   if(dc.ge.core_slope*nthr.and.(.not.no_case3)) then !the contraction dimension is large enough to be distributed
!	    write(*,'("DEBUG(tensor_algebra_dil::tensor_block_pcontract): kernel case/scheme: 3/",i1)') ker3 !debug
	    select case(ker3)
	    case(0)
!SCHEME 0:
!            write(*,'("DEBUG(tensor_algebra_dil::tensor_block_pcontract): dl,dr,dc:",3(1x,i9))') dl,dr,dc !debug
             redm(:,:)=0d0
             do b2=0_8,dr-1_8,red_mat_size
              e2=min(red_mat_size-1_8,dr-1_8-b2)
              do b1=0_8,dl-1_8,red_mat_size
               e1=min(red_mat_size-1_8,dl-1_8-b1)
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(l0,l1,l2,ll,lr) NUM_THREADS(nthreads)
	       do l2=0_8,e2
	        lr=(b2+l2)*dc
	        do l1=0_8,e1
	         ll=(b1+l1)*dc
!$OMP MASTER
	         val=0d0
!$OMP END MASTER
!$OMP BARRIER
!$OMP DO SCHEDULE(GUIDED) REDUCTION(+:val)
	         do l0=0_8,dc-1_8; val=val+ltens(ll+l0)*rtens(lr+l0); enddo
!$OMP END DO
!$OMP MASTER
		 redm(l1,l2)=val
!$OMP END MASTER
	        enddo
	       enddo
!$OMP END PARALLEL
	       do l2=0_8,e2
	        ld=(b2+l2)*dl
	        do l1=0_8,e1
	         dtens(ld+b1+l1)=dtens(ld+b1+l1)+redm(l1,l2)
	        enddo
	       enddo
	      enddo
	     enddo
	    case default
	     ierr=-3
	    end select
	   else !dr & dl & dc are all small
	    if(dr*dl.ge.core_slope*nthr.and.(.not.no_case4)) then !the destination matrix is large enough to be distributed
!	     write(*,'("DEBUG(tensor_algebra_dil::tensor_block_pcontract): kernel case: 4")') !debug
!	     write(*,'("DEBUG(tensor_algebra_dil::tensor_block_pcontract): dl,dr,dc:",3(1x,i9))') dl,dr,dc !debug
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0,l1,l2,ll,lr,val) SCHEDULE(GUIDED) COLLAPSE(2) NUM_THREADS(nthreads)
	     do l2=0_8,dr-1_8
	      do l1=0_8,dl-1_8
	       ll=l1*dc; lr=l2*dc
	       val=dtens(l2*dl+l1)
	       do l0=0_8,dc-1_8; val=val+ltens(ll+l0)*rtens(lr+l0); enddo
	       dtens(l2*dl+l1)=val
	      enddo
	     enddo
!$OMP END PARALLEL DO
	    else !all matrices are very small (serial execution)
!	     write(*,'("DEBUG(tensor_algebra_dil::tensor_block_pcontract): kernel case: 5 (serial)")') !debug
!	     write(*,'("DEBUG(tensor_algebra_dil::tensor_block_pcontract): dl,dr,dc:",3(1x,i9))') dl,dr,dc !debug
	     do l2=0_8,dr-1_8
	      lr=l2*dc; ld=l2*dl
	      do l1=0_8,dl-1_8
	       ll=l1*dc
	       val=dtens(ld+l1)
	       do l0=0_8,dc-1_8; val=val+ltens(ll+l0)*rtens(lr+l0); enddo
	       dtens(ld+l1)=val
	      enddo
	     enddo
	    endif
	   endif
	  endif
	 endif
	else
	 ierr=1
	endif
!	write(*,'("DEBUG(tensor_algebra_dil::tensor_block_pcontract): exit code / kernel time: ",i5,1x,F10.4)') &
!        ierr,thread_wtime(time_beg) !debug
	return
	end subroutine tensor_block_pcontract
!--------------------------------------------
!AUXILIARY SUBROUTINES (also PRIVATE):
!-------------------------------------------------
	subroutine merge_sort_key_int_(ni,key,trn) !SERIAL
!This subroutine sorts an array of NI items in a non-descending order according to their keys.
!The algorithm is due to Johann von Neumann.
!INPUT:
! - ni - number of items;
! - key(1:ni) - item keys (retrieved by old item numbers!): arbitrary integers;
! - trn(0:ni) - initial permutation of ni items (a sequence of old numbers), trn(0) is the initial sign;
!OUTPUT:
! - trn(0:ni) - sorted permutation (new sequence of old numbers) of ni items (according to their keys), the sign is in trn(0).
	implicit none
	integer(int_kind), intent(in):: ni,key(1:ni)
	integer(int_kind), intent(inout):: trn(0:ni)
!-----------------------------------------------------
	integer(int_kind), parameter:: max_in_mem=1024
!-----------------------------------------------------
	integer(int_kind):: i,j,k,l,m,n,k0,k1,k2,k3,k4,ks,kf,ierr
	integer(int_kind), target:: prms(1:max_in_mem)
	integer(int_kind), pointer:: prm(:)

	if(ni.gt.1) then
	 if(ni.le.max_in_mem) then; prm=>prms; else; allocate(prm(1:ni)); endif
	 n=1
	 do while(n.lt.ni)
	  m=n*2
	  do i=1,ni,m
	   k1=i; k2=i+n
	   if(k2.gt.ni) then
	    k2=ni+1; k3=0; k4=0 !no right block, only left block
	   else
	    k3=i+n; k4=min(ni+1,i+m) !right block present
	   endif
	   kf=min(ni+1,i+m)-i; l=0
	   do while(l.lt.kf)
	    if(k3.ge.k4) then !right block is over
	     prm(i+l:i+kf-1)=trn(k1:k2-1); l=kf
	    elseif(k1.ge.k2) then !left block is over
	     prm(i+l:i+kf-1)=trn(k3:k4-1); l=kf
	    else
	     if(key(trn(k1))-key(trn(k3)).gt.0) then
	      prm(i+l)=trn(k3); k3=k3+1; trn(0)=(1-2*mod(k2-k1,2))*trn(0)
	     else
	      prm(i+l)=trn(k1); k1=k1+1
	     endif
	     l=l+1
	    endif
	   enddo
	  enddo
	  trn(1:ni)=prm(1:ni)
	  n=m
	 enddo
	 if(ni.le.max_in_mem) then; nullify(prm); else; deallocate(prm); if(associated(prm)) nullify(prm); endif
	endif
	return
	end subroutine merge_sort_key_int_
!-------------------------------------------------------------------------
        subroutine divide_segment_(seg_range,subseg_num,subseg_sizes,ierr) !SERIAL
!A segment of range <seg_range> will be divided into <subseg_num> subsegments maximally uniformly.
!The length of each subsegment will be returned in the array <subseg_sizes(1:subseg_num)>.
!Any two subsegments will not differ in length by more than 1, longer subsegments preceding the shorter ones.
        implicit none
        integer(int8_kind), intent(in):: seg_range,subseg_num
        integer(int8_kind), intent(out):: subseg_sizes(1:subseg_num)
        integer(int_kind), intent(inout):: ierr
        integer(int8_kind):: i,m,n
        ierr=0
        if(seg_range.gt.0.and.subseg_num.gt.0) then
         n=seg_range/subseg_num; m=mod(seg_range,subseg_num)
         do i=1,m; subseg_sizes(i)=n+1; enddo
         do i=m+1,subseg_num; subseg_sizes(i)=n; enddo
        else
         ierr=-1
        endif
        return
        end subroutine divide_segment_

       end module tensor_dil_omp
