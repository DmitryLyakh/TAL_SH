!Tensor Algebra for Multi- and Many-core CPUs (OpenMP based).
!AUTHOR: Dmitry I. Lyakh (Liakh): quant4me@gmail.com
!REVISION: 2019/03/11

!Copyright (C) 2013-2019 Dmitry I. Lyakh (Liakh)
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

!GNU linking: -lblas -gfortran -lgomp
!ACRONYMS:
! - mlndx - multiindex;
! - Lm - Level-min segment size (the lowest level segment size for bricked storage);
! - dlf - dimension-led storage of tensor blocks where the 1st dimension is the most minor (Fortran like) (DEFAULT):
!         Numeration within each dimension starts from 0: [0..extent-1].
! - dlc - dimension-led storage of tensor blocks where the 1st dimension is the most senior (C like);
!         Numeration within each dimension starts from 0: [0..extent-1].
! - brf - bricked storage of tensor blocks where the 1st dimension is the most minor (Fortran like) (DEFAULT);
! - brc - bricked storage of tensor blocks where the 1st dimension is the most senior (C like);
! - r4 - real(4);
! - r8 - real(8);
! - c4 - complex(4);
! - c8 - complex(8);
!PREPROCESSOR:
! -D NO_OMP: Do not use OpenMP (serial);
! -D NO_BLAS: Replace BLAS calls with in-house routines (slower);
! -D USE_MKL: Use Intel MKL library interface for BLAS;
! -D NO_PHI: Ignore Intel MIC (Xeon Phi);
       module tensor_algebra_cpu
!       use, intrinsic:: ISO_C_BINDING
!       use dil_basic
        use tensor_algebra !includes ISO_C_BINDING, dil_basic
        use stsubs
        use combinatoric
        use timers
        use symm_index
#ifdef USE_MKL
        !use blas95
        !use lapack95
        !use f95_precision
        !use mkl_service
#endif
#ifndef NO_OMP
        use omp_lib
        implicit none
        public
#else
        implicit none
        public
        integer, external:: omp_get_max_threads,omp_set_num_threads
#endif
!PARAMETERS:
 !Default output for the module procedures:
        integer, private:: CONS_OUT=6     !default output device for this module (also used for INTEL MIC TAL)
        logical, private:: VERBOSE=.TRUE. !verbosity (also used for INTEL MIC TAL)
        integer, private:: DEBUG=0        !debugging mode
        integer, private:: LOGGING=0      !logging mode
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: CONS_OUT,VERBOSE
!DIR$ ATTRIBUTES ALIGN:128:: CONS_OUT,VERBOSE
#endif
 !Global:
        integer, parameter, public:: MAX_SHAPE_STR_LEN=1024   !max allowed length for a tensor shape specification string (TSSS)
        integer, parameter, public:: LONGINT=INTL             !long integer size in bytes
        integer, parameter, private:: MAX_THREADS=1024        !max allowed number of threads in this module
        integer, private:: MEM_ALLOC_POLICY=MEM_ALLOC_TMP_BUF !memory allocation policy
        logical, private:: MEM_ALLOC_FALLBACK=.TRUE.          !memory allocation fallback to the regular allocator
        logical, private:: DATA_KIND_SYNC=.TRUE. !if .TRUE., each tensor operation will syncronize all existing data kinds
        logical, private:: TRANS_SHMEM=.TRUE.    !cache-efficient (true) VS scatter (false) tensor transpose algorithm
#ifndef NO_BLAS
        logical, private:: DISABLE_BLAS=.FALSE.  !if .TRUE. and BLAS is accessible, BLAS calls will be replaced by my own routines
#else
        logical, private:: DISABLE_BLAS=.TRUE.   !if .TRUE. and BLAS is accessible, BLAS calls will be replaced by my own routines
#endif
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: MAX_SHAPE_STR_LEN,LONGINT,MAX_THREADS,MEM_ALLOC_POLICY,MEM_ALLOC_FALLBACK
!DIR$ ATTRIBUTES OFFLOAD:mic:: DATA_KIND_SYNC,TRANS_SHMEM,DISABLE_BLAS
!DIR$ ATTRIBUTES ALIGN:128:: MAX_SHAPE_STR_LEN,LONGINT,MAX_THREADS,MEM_ALLOC_POLICY,MEM_ALLOC_FALLBACK
!DIR$ ATTRIBUTES ALIGN:128:: DATA_KIND_SYNC,TRANS_SHMEM,DISABLE_BLAS
#endif
 !Numerical:
        real(8), parameter, private:: ABS_CMP_THRESH=1d-13 !default absolute error threshold for numerical comparisons
        real(8), parameter, private:: REL_CMP_THRESH=1d-2  !default relative error threshold for numerical comparisons
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: ABS_CMP_THRESH,REL_CMP_THRESH
!DIR$ ATTRIBUTES ALIGN:128:: ABS_CMP_THRESH,REL_CMP_THRESH
#endif
!DERIVED DATA TYPES:
 !Tensor shape (storage layout specification for a tensor block):
        type, public:: tensor_shape_t
         integer:: num_dim=-1                      !total number of dimensions (num_dim=0 defines a scalar tensor).
         integer, pointer:: dim_extent(:)=>NULL()  !extent of each dimension (if num_dim>0): range=[0..extent-1].
         integer, pointer:: dim_divider(:)=>NULL() !divider for each dimension, i.e. the <Lm_segment_size> (ordered dimensions must have the same divider!): %dim_divider(1)=0 means that an alternative (neither dimension-led nor bricked) storage layout is used.
         integer, pointer:: dim_group(:)=>NULL()   !dimension grouping (default group 0 means no symmetry restrictions): if %dim_divider(1)=0, then %dim_group(1) regulates the alternative storage layout kind.
        end type tensor_shape_t
 !Tensor block:
        type, public:: tensor_block_t
         integer(LONGINT):: tensor_block_size=0_LONGINT           !total number of elements in the tensor block (informal, set after creation)
         integer:: ptr_alloc=0                                    !pointer allocation status (bits set when pointers are getting allocated, not just associated)
         type(tensor_shape_t):: tensor_shape                      !shape of the tensor block (see above)
         complex(8):: scalar_value=(0d0,0d0)                      !scalar value if the rank is zero, otherwise can be used for storing the norm of the tensor block
         real(4), pointer, contiguous:: data_real4(:)=>NULL()     !tensor block data (float)
         real(8), pointer, contiguous:: data_real8(:)=>NULL()     !tensor block data (double)
         complex(4), pointer, contiguous:: data_cmplx4(:)=>NULL() !tensor block data (float complex)
         complex(8), pointer, contiguous:: data_cmplx8(:)=>NULL() !tensor block data (double complex)
        end type tensor_block_t

!GENERIC INTERFACES:
        interface tensor_block_shape_create
         module procedure tensor_block_shape_create_sym
         module procedure tensor_block_shape_create_num
        end interface tensor_block_shape_create

        interface array_alloc
         module procedure array_alloc_r4
         module procedure array_alloc_r8
         module procedure array_alloc_c4
         module procedure array_alloc_c8
        end interface array_alloc

        interface array_free
         module procedure array_free_r4
         module procedure array_free_r8
         module procedure array_free_c4
         module procedure array_free_c8
        end interface array_free

        interface tensor_block_slice_dlf
         module procedure tensor_block_slice_dlf_r4
         module procedure tensor_block_slice_dlf_r8
         module procedure tensor_block_slice_dlf_c4
         module procedure tensor_block_slice_dlf_c8
        end interface tensor_block_slice_dlf

        interface tensor_block_insert_dlf
         module procedure tensor_block_insert_dlf_r4
         module procedure tensor_block_insert_dlf_r8
         module procedure tensor_block_insert_dlf_c4
         module procedure tensor_block_insert_dlf_c8
        end interface tensor_block_insert_dlf

        interface tensor_block_copy_dlf
         module procedure tensor_block_copy_dlf_r4
         module procedure tensor_block_copy_dlf_r8
         module procedure tensor_block_copy_dlf_c4
         module procedure tensor_block_copy_dlf_c8
        end interface tensor_block_copy_dlf

        interface tensor_block_copy_scatter_dlf
         module procedure tensor_block_copy_scatter_dlf_r4
         module procedure tensor_block_copy_scatter_dlf_r8
         module procedure tensor_block_copy_scatter_dlf_c4
         module procedure tensor_block_copy_scatter_dlf_c8
        end interface tensor_block_copy_scatter_dlf

        interface tensor_block_fcontract_dlf
         module procedure tensor_block_fcontract_dlf_r4
         module procedure tensor_block_fcontract_dlf_r8
         module procedure tensor_block_fcontract_dlf_c4
         module procedure tensor_block_fcontract_dlf_c8
        end interface tensor_block_fcontract_dlf

        interface tensor_block_pcontract_dlf
         module procedure tensor_block_pcontract_dlf_r4
         module procedure tensor_block_pcontract_dlf_r8
         module procedure tensor_block_pcontract_dlf_c4
         module procedure tensor_block_pcontract_dlf_c8
        end interface tensor_block_pcontract_dlf

        interface tensor_block_ftrace_dlf
         module procedure tensor_block_ftrace_dlf_r4
         module procedure tensor_block_ftrace_dlf_r8
         module procedure tensor_block_ftrace_dlf_c4
         module procedure tensor_block_ftrace_dlf_c8
        end interface tensor_block_ftrace_dlf

        interface tensor_block_ptrace_dlf
         module procedure tensor_block_ptrace_dlf_r4
         module procedure tensor_block_ptrace_dlf_r8
         module procedure tensor_block_ptrace_dlf_c4
         module procedure tensor_block_ptrace_dlf_c8
        end interface tensor_block_ptrace_dlf

!FUNCTION VISIBILITY:
        public get_mem_alloc_policy        !gets the current memory allocation policy for sizeable arrays
        public set_mem_alloc_policy        !sets the memory allocation policy for sizeable arrays
        public set_data_kind_sync          !turns on/off data kind synchronization (0/1)
        public set_transpose_algorithm     !switches between scatter (0) and shared-memory (1) tensor transpose algorithms
        public set_matmult_algorithm       !switches between BLAS GEMM (0) and my OpenMP matmult kernels (1)
        public cmplx4_to_real4             !returns the real approximate of a complex number (algorithm by D.I.L.)
        public cmplx8_to_real8             !returns the real approximate of a complex number (algorithm by D.I.L.)
        public tensor_shape_assoc          !constructs a tensor shape object by pointer associating with external data
        public tensor_block_layout         !returns the type of the storage layout for a given tensor block
        public tensor_block_shape_size     !determines the tensor block size induced by its shape
        public tensor_master_data_kind     !determines the master data kind present in a tensor block
        public tensor_common_data_kind     !determines the common data kind present in two compatible tensor blocks
        public tensor_block_compatible     !determines whether two tensor blocks are compatible (under an optional index permutation)
        public tensor_block_mimic          !mimics the internal structure of a tensor block without copying the actual data
        public tensor_block_create         !creates a tensor block based on the shape specification string (SSS)
        public tensor_block_init           !initializes a tensor block with either a predefined value or random numbers
        public tensor_block_is_empty       !returns TRUE if the tensor block is empty, FALSE otherwise
        public tensor_block_assoc          !associates an empty <tensor_block_t> object with externally provided data
        public tensor_block_destroy        !destroys a tensor block
        public tensor_block_sync           !allocates and/or synchronizes different data kinds in a tensor block
        public tensor_block_scale          !multiplies all elements of a tensor block by some factor
        public tensor_block_conjg          !complex conjugates all elements of a tensor block
        public tensor_block_norm1          !determines the 1-norm of a tensor block (the sum of moduli of all elements)
        public tensor_block_norm2          !determines the squared Euclidean (Frobenius) 2-norm of a tensor block
        public tensor_block_max            !determines the maximal (by modulus) tensor block element
        public tensor_block_min            !determines the minimal (by modulus) tensor block element
        public tensor_block_slice          !extracts a slice from a tensor block
        public tensor_block_insert         !inserts a slice into a tensor block
        public tensor_block_print          !prints a tensor block
        public tensor_block_trace          !intra-tensor index contraction (accumulative trace)
        public tensor_block_cmp            !compares two tensor blocks
        public tensor_block_copy           !makes a copy of a tensor block (with an optional index permutation)
        public tensor_block_add            !adds one tensor block to another
        public tensor_block_contract       !inter-tensor index contraction (accumulative contraction)
        public tensor_block_scalar_value   !returns the scalar value component of <tensor_block_t>
        public get_mlndx_addr              !generates an array of addressing increments for the linearization map for symmetric multi-indices
        public mlndx_value                 !returns the address associated with a (symmetric) multi-index, based on the array generated by <get_mlndx_addr>
        public tensor_shape_rnd            !returns a random tensor-shape-specification-string (TSSS)
        public tensor_shape_rank           !returns the number of dimensions in a tensor-shape-specification-string (TSSS)
        public tensor_shape_str_create     !creates a tensor shape specification string
        public get_contr_pattern           !converts a mnemonic contraction pattern into the digital form used by tensor_block_contract
        public get_contr_pattern_sym       !converts a digital contraction pattern into a symbolic form
        public get_contr_permutations      !given a digital contraction pattern, returns all tensor permutations necessary for the subsequent matrix multiplication
        public contr_pattern_rnd           !returns a random tensor contraction pattern
        public coherence_control_var       !returns a coherence control variable based on a mnemonic input
        public tensor_block_shape_create   !generates the tensor shape based on either the tensor shape specification string (TSSS) or numeric arguments
        public tensor_block_shape_create_sym !generates the tensor shape based on the tensor shape specification string (TSSS)
        public tensor_block_shape_create_num !generates the tensor shape based on the numeric arguments
        public tensor_block_shape_ok       !checks the correctness of a tensor shape generated from a tensor shape specification string (TSSS)
        public tensor_block_alloc          !sets/queries the allocation status of data pointers in a tensor block
        private array_alloc                !allocates an array pointer {R4,R8,C4,C8}
        private array_alloc_r4             !allocates an array pointer R4
        private array_alloc_r8             !allocates an array pointer R8
        private array_alloc_c4             !allocates an array pointer C4
        private array_alloc_c8             !allocates an array pointer C8
        private array_free                 !frees an array pointer {R4,R8,C4,C8}
        private array_free_r4              !frees an array pointer R4
        private array_free_r8              !frees an array pointer R8
        private array_free_c4              !frees an array pointer C4
        private array_free_c8              !frees an array pointer C8
        private tensor_block_slice_dlf     !extracts a slice from a tensor block (Fortran-like dimension-led storage layout)
        private tensor_block_insert_dlf    !inserts a slice into a tensor block (Fortran-like dimension-led storage layout)
        private tensor_block_copy_dlf      !tensor transpose for dimension-led (Fortran-like-stored) dense tensor blocks
        private tensor_block_copy_scatter_dlf !tensor transpose for dimension-led (Fortran-like-stored) dense tensor blocks (scattering variant)
        private tensor_block_fcontract_dlf !multiplies two matrices derived from tensors to produce a scalar (left is transposed, right is normal)
        private tensor_block_pcontract_dlf !multiplies two matrices derived from tensors to produce a third matrix (left is transposed, right is normal)
        private tensor_block_ftrace_dlf    !takes a full trace of a tensor block
        private tensor_block_ptrace_dlf    !takes a partial trace of a tensor block

       contains
!-----------------
!PUBLIC FUNCTIONS:
!----------------------------------------------------------------------
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: get_mem_alloc_policy
#endif
        function get_mem_alloc_policy(ierr,fallback) result(mem_policy) !SERIAL
!Gets the current memory allocation policy.
         implicit none
         integer:: mem_policy                      !out: memory allocation policy
         integer, intent(out), optional:: ierr     !out: error code
         logical, intent(out), optional:: fallback !out: fallback status

         mem_policy=MEM_ALLOC_POLICY
         if(present(fallback)) fallback=MEM_ALLOC_FALLBACK
         if(present(ierr)) ierr=0
         return
        end function get_mem_alloc_policy
!----------------------------------------------------------------
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: set_mem_alloc_policy
#endif
        subroutine set_mem_alloc_policy(mem_policy,ierr,fallback) !SERIAL
!Sets the memory allocation policy.
         implicit none
         integer, intent(in):: mem_policy         !in: memory allocation policy
         integer, intent(out), optional:: ierr    !out: error code
         logical, intent(in), optional:: fallback !in: memory allocation fallback to regular

         select case(mem_policy)
         case(MEM_ALLOC_REGULAR,MEM_ALLOC_TMP_BUF,MEM_ALLOC_ALL_BUF)
!!!$OMP ATOMIC WRITE SEQ_CST
!$OMP ATOMIC WRITE
          MEM_ALLOC_POLICY=mem_policy
         case default
          if(present(ierr)) ierr=1 !invalid policy
          return
         end select
         if(present(fallback)) then
!!!$OMP ATOMIC WRITE SEQ_CST
!$OMP ATOMIC WRITE
          MEM_ALLOC_FALLBACK=fallback
         endif
         if(present(ierr)) ierr=0
         return
        end subroutine set_mem_alloc_policy
!-----------------------------------------
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: set_data_kind_sync
#endif
	subroutine set_data_kind_sync(alg) !SERIAL
	implicit none
	integer, intent(in):: alg
	if(alg.eq.0) then
!!!$OMP ATOMIC WRITE SEQ_CST
!$OMP ATOMIC WRITE
	 DATA_KIND_SYNC=.FALSE.
	else
!!!$OMP ATOMIC WRITE SEQ_CST
!$OMP ATOMIC WRITE
	 DATA_KIND_SYNC=.TRUE.
	endif
	return
	end subroutine set_data_kind_sync
!----------------------------------------------
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: set_transpose_algorithm
#endif
	subroutine set_transpose_algorithm(alg) !SERIAL
	implicit none
	integer, intent(in):: alg
	if(alg.eq.0) then
!!!$OMP ATOMIC WRITE SEQ_CST
!$OMP ATOMIC WRITE
	 TRANS_SHMEM=.FALSE.
	else
!!!$OMP ATOMIC WRITE SEQ_CST
!$OMP ATOMIC WRITE
	 TRANS_SHMEM=.TRUE.
	endif
	return
	end subroutine set_transpose_algorithm
!--------------------------------------------
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: set_matmult_algorithm
#endif
	subroutine set_matmult_algorithm(alg) !SERIAL
	implicit none
	integer, intent(in):: alg
#ifndef NO_BLAS
	if(alg.eq.0) then
!!!$OMP ATOMIC WRITE SEQ_CST
!$OMP ATOMIC WRITE
	 DISABLE_BLAS=.FALSE.
	else
!!!$OMP ATOMIC WRITE SEQ_CST
!$OMP ATOMIC WRITE
	 DISABLE_BLAS=.TRUE.
	endif
#endif
	return
	end subroutine set_matmult_algorithm
!---------------------------------------------
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: cmplx4_to_real4
#endif
        real(4) function cmplx4_to_real4(cmplx_num) !SERIAL
!This function returns a real approximant for a complex number with the following properties:
! 1) The Euclidean (Frobenius) norm (modulus) is preserved;
! 2) The sign inversion symmetry is preserved.
         implicit none
         complex(4), intent(in):: cmplx_num
         real(4):: real_part

         real_part=real(cmplx_num)
         if(real_part.ne.0.0) then
          cmplx4_to_real4=abs(cmplx_num)*sign(1.0,real_part)
         else
          cmplx4_to_real4=aimag(cmplx_num)
         endif
         return
        end function cmplx4_to_real4
!---------------------------------------------
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: cmplx8_to_real8
#endif
        real(8) function cmplx8_to_real8(cmplx_num) !SERIAL
!This function returns a real approximant for a complex number with the following properties:
! 1) The Euclidean (Frobenius) norm (modulus) is preserved;
! 2) The sign inversion symmetry is preserved.
         implicit none
         complex(8), intent(in):: cmplx_num
         real(8):: real_part

         real_part=real(cmplx_num,8)
         if(real_part.ne.0d0) then
          cmplx8_to_real8=abs(cmplx_num)*sign(1d0,real_part)
         else
          cmplx8_to_real8=aimag(cmplx_num)
         endif
         return
        end function cmplx8_to_real8
!--------------------------------------------------------------------
        subroutine tensor_shape_assoc(tens_shape,ierr,dims,divs,grps)
!Constructs a tensor shape object <tensor_shape_t) by pointer associating it
!with external data. Incoming <tens_shape> must be empty on entrance.
         implicit none
         type(tensor_shape_t), intent(inout):: tens_shape !inout: tensor shape
         integer, intent(out):: ierr                      !out: error code (0:success)
         integer, intent(in), pointer, contiguous, optional:: dims(:) !in: dimension extents (length = tensor rank)
         integer, intent(in), pointer, contiguous, optional:: divs(:) !in: dimension dividers
         integer, intent(in), pointer, contiguous, optional:: grps(:) !in: dimension groups
         integer:: n

         ierr=0
         if(tens_shape%num_dim.lt.0.and.&
           &(.not.(associated(tens_shape%dim_extent).or.&
                  &associated(tens_shape%dim_divider).or.&
                  &associated(tens_shape%dim_group)))) then
          if(present(dims)) then
           if(associated(dims)) then
            n=size(dims)
            if(n.ge.0.and.n.le.MAX_TENSOR_RANK) then
             tens_shape%num_dim=n
             if(n.gt.0) then
              tens_shape%dim_extent(1:n)=>dims(1:n)
              if(present(divs)) then
               if(associated(divs)) then
                if(size(divs).eq.n) then
                 tens_shape%dim_divider(1:n)=>divs(1:n)
                else
                 ierr=1; return
                endif
               endif
              endif
              if(present(grps)) then
               if(associated(grps)) then
                if(size(grps).eq.n) then
                 tens_shape%dim_group(1:n)=>grps(1:n)
                else
                 ierr=2; return
                endif
               endif
              endif
             endif
            else
             ierr=3
            endif
           else
            tens_shape%num_dim=0
           endif
          else
           tens_shape%num_dim=0
          endif
         else
          ierr=4 !tensor shape is not empty
         endif
         return
        end subroutine tensor_shape_assoc
!------------------------------------------------------------------
	integer function tensor_block_layout(tens,ierr,check_shape) !SERIAL
!Returns the type of the storage layout for a given tensor block <tens>.
!INPUT:
! - tens - tensor block;
! - check_shape - (optional) if .TRUE., the tensor shape will be checked;
!OUTPUT:
! - tensor_block_layout - tensor block storage layout;
! - ierr - error code (0:sucess).
!NOTES:
! - %dim_divider(1)=0 means that an alternative (neither dimension-led nor bricked) storage layout is used,
!                     whose kind is regulated by the %dim_group(1) then.
! - For dimension_led, bricked_dense and bricked_ordered tensor blocks,
!   index group 0 does not imply any index ordering restrictions.
!   The presence of group 1,2,etc...(up to the tensor rank) assumes
!   that the bricked_ordered storage layout is in use. It is illegal
!   to assign non-zero group numbers to indices for dimension_led
!   and bricked_dense tensor blocks.
	implicit none
	type(tensor_block_t), intent(inout):: tens !(out) because of <tensor_block_shape_ok>
	logical, intent(in), optional:: check_shape
	integer, intent(inout):: ierr
	integer i,j,k,l,m,n,ibus(0:max_tensor_rank)

	ierr=0; tensor_block_layout=not_allocated
	if(present(check_shape)) then
	 if(check_shape) then; ierr=tensor_block_shape_ok(tens); if(ierr.ne.0) return; endif
	endif
	if(tens%tensor_shape%num_dim.gt.0.and.associated(tens%tensor_shape%dim_extent).and.&
	   &associated(tens%tensor_shape%dim_divider).and.associated(tens%tensor_shape%dim_group)) then !true tensor
	 if(tens%tensor_shape%dim_divider(1).gt.0) then !dimension-led or bricked
	  tensor_block_layout=dimension_led
	  do i=1,tens%tensor_shape%num_dim
	   if(tens%tensor_shape%dim_extent(i).ne.tens%tensor_shape%dim_divider(i)) then
	    tensor_block_layout=bricked_dense; exit
	   endif
	  enddo
	  if(tensor_block_layout.eq.bricked_dense) then
	   ibus(0:tens%tensor_shape%num_dim)=0
	   do i=1,tens%tensor_shape%num_dim
	    j=tens%tensor_shape%dim_group(i); if(j.lt.0.or.j.gt.tens%tensor_shape%num_dim) then; ierr=1000; return; endif
	    if(j.gt.0.and.ibus(j).gt.0) then; tensor_block_layout=bricked_ordered; exit; endif
	    ibus(j)=ibus(j)+1
	   enddo
	  endif
	 else !alternative storage layout
	  !`Future
	 endif
	elseif(tens%tensor_shape%num_dim.eq.0) then !scalar tensor
	 tensor_block_layout=scalar_tensor
	endif
	return
	end function tensor_block_layout
!-------------------------------------------------------------------------
	integer(LONGINT) function tensor_block_shape_size(tens_block,ierr) !SERIAL
!This function determines the size of a tensor block (number of elements) by its shape.
!Note that a scalar (0-dimensional tensor) and a 1-dimensional tensor with extent 1 are not the same!
	implicit none
	type(tensor_block_t), intent(inout):: tens_block !(out) because of <tensor_block_layout> because of <tensor_block_shape_ok>
	integer, intent(inout):: ierr
	integer i,j,k,l,m,n,k0,k1,k2,k3,ks,kf,tst

	ierr=0; tensor_block_shape_size=0_LONGINT
	tst=tensor_block_layout(tens_block,ierr); if(ierr.ne.0) return
	select case(tst)
	case(not_allocated)
	 tensor_block_shape_size=0_LONGINT; ierr=-1
	case(scalar_tensor)
	 tensor_block_shape_size=1_LONGINT
	case(dimension_led,bricked_dense)
	 tensor_block_shape_size=1_LONGINT
	 do i=1,tens_block%tensor_shape%num_dim
	  if(tens_block%tensor_shape%dim_extent(i).gt.0.and.&
	     &tens_block%tensor_shape%dim_divider(i).gt.0.and.&
	     &tens_block%tensor_shape%dim_divider(i).le.tens_block%tensor_shape%dim_extent(i)) then
	   tensor_block_shape_size=tensor_block_shape_size*int(tens_block%tensor_shape%dim_extent(i),LONGINT)
	  else
	   ierr=100+i; return !invalid dimension specificator in tens_block%tensor_shape%
	  endif
	 enddo
	case(bricked_ordered)
	 !`Future: Compute volumes of all ordered multi-indices and multiply them.
	case(sparse_list)
	 !`Future
	case(compressed)
	 !`Future
	case default
	 ierr=-2
	end select
	return
	end function tensor_block_shape_size
!---------------------------------------------------------------
	character(2) function tensor_master_data_kind(tens,ierr) !SERIAL
!This function determines the master data kind present in a tensor block.
!INPUT:
! - tens - tensor block;
!OUTPUT:
! - tensor_master_data_kind - one of {'r4','r8','c4','c8','--'}, where the latter means "not allocated";
! - ierr - error code (0:success).
	implicit none
	type(tensor_block_t), intent(in):: tens
	integer, intent(inout):: ierr

	ierr=0; tensor_master_data_kind='--'
	if(tens%tensor_shape%num_dim.eq.0) then
	 tensor_master_data_kind='c8'
	elseif(tens%tensor_shape%num_dim.gt.0) then
	 if(associated(tens%data_real4)) tensor_master_data_kind='r4'
	 if(associated(tens%data_real8)) tensor_master_data_kind='r8'
	 if(associated(tens%data_cmplx4)) tensor_master_data_kind='c4'
	 if(associated(tens%data_cmplx8)) tensor_master_data_kind='c8'
	endif
	return
	end function tensor_master_data_kind
!----------------------------------------------------------------------
	character(2) function tensor_common_data_kind(tens1,tens2,ierr) !SERIAL
!This function determines the common data kind present in two commpatible tensor blocks.
!INPUT:
! - tens1, tens2 - compatible tensor blocks;
!OUTPUT:
! - tensor_common_data_kind - one of {'r4','r8','c4','c8','--'}, where the latter means "not applicable";
! - ierr - error code (0:success).
	implicit none
	type(tensor_block_t), intent(in):: tens1,tens2
	integer, intent(inout):: ierr

	ierr=0; tensor_common_data_kind='--'
	if(tens1%tensor_shape%num_dim.eq.0.and.tens2%tensor_shape%num_dim.eq.0) then
	 tensor_common_data_kind='c8'
	elseif(tens1%tensor_shape%num_dim.gt.0.and.tens2%tensor_shape%num_dim.gt.0) then
	 if(associated(tens1%data_real4).and.associated(tens2%data_real4)) tensor_common_data_kind='r4'
	 if(associated(tens1%data_real8).and.associated(tens2%data_real8)) tensor_common_data_kind='r8'
	 if(associated(tens1%data_cmplx4).and.associated(tens2%data_cmplx4)) tensor_common_data_kind='c4'
	 if(associated(tens1%data_cmplx8).and.associated(tens2%data_cmplx8)) tensor_common_data_kind='c8'
	endif
	return
	end function tensor_common_data_kind
!-------------------------------------------------------------------------------------------------
	logical function tensor_block_compatible(tens_in,tens_out,ierr,transp,no_check_data_kinds) !SERIAL
!This function decides whether two tensor blocks are compatible
!under some index permutation (the latter is optional).
!INPUT:
! - tens_in - input tensor;
! - tens_out - output tensor;
! - transp(0:*) - (optional) O2N index permutation;
! - no_check_data_kinds - (optional) if .TRUE., the two tensor blocks do not have to have the same data kinds allocated;
!OUTPUT:
! - tensor_block_compatible - .TRUE./.FALSE.;
! - ierr - error code (0:success).
!NOTES:
! - Non-allocated tensor blocks are all compatible.
! - Tensor block storage layouts are ignored.
	implicit none
	type(tensor_block_t), intent(in):: tens_in,tens_out
	integer, intent(in), optional:: transp(0:*)
	logical, intent(in), optional:: no_check_data_kinds
	integer, intent(inout):: ierr
	integer i,j,k,l,m,n,k0,k1,k2,k3,ks,kf
	integer trn(0:max_tensor_rank)
	integer(LONGINT) ls
	logical chdtk

	ierr=0; tensor_block_compatible=.TRUE.
	if(tens_in%tensor_shape%num_dim.eq.tens_out%tensor_shape%num_dim) then
	 n=tens_in%tensor_shape%num_dim
	 if(n.gt.0) then
!Check tensor shapes:
	  if(associated(tens_in%tensor_shape%dim_extent).and.associated(tens_in%tensor_shape%dim_divider).and.&
	    &associated(tens_in%tensor_shape%dim_group).and.associated(tens_out%tensor_shape%dim_extent).and.&
	    &associated(tens_out%tensor_shape%dim_divider).and.associated(tens_out%tensor_shape%dim_group)) then
	   if(present(transp)) then; trn(0:n)=transp(0:n); else; trn(0:n)=(/+1,(j,j=1,n)/); endif
	   do i=1,n
	    if(tens_out%tensor_shape%dim_extent(trn(i)).ne.tens_in%tensor_shape%dim_extent(i).or.&
	       &tens_out%tensor_shape%dim_divider(trn(i)).ne.tens_in%tensor_shape%dim_divider(i).or.&
	       &tens_out%tensor_shape%dim_group(trn(i)).ne.tens_in%tensor_shape%dim_group(i)) then
	     tensor_block_compatible=.FALSE.
	     exit
	    endif
	   enddo
!Check data kinds:
	   if(tensor_block_compatible) then
	    if(tens_in%tensor_block_size.ne.tens_out%tensor_block_size) then
	     tensor_block_compatible=.FALSE.; ierr=1 !the same shape tensor blocks have different total sizes
	    else
	     if(present(no_check_data_kinds)) then; chdtk=no_check_data_kinds; else; chdtk=.FALSE.; endif
	     if(.not.chdtk) then
	      if((associated(tens_in%data_real4).and.(.not.associated(tens_out%data_real4))).or.&
	        &((.not.associated(tens_in%data_real4)).and.associated(tens_out%data_real4))) then
	       tensor_block_compatible=.FALSE.; return
	      else
	       if(associated(tens_in%data_real4)) then
	        ls=size(tens_in%data_real4)
	        if(size(tens_out%data_real4).ne.ls.or.tens_out%tensor_block_size.ne.ls) then
	         tensor_block_compatible=.FALSE.; ierr=2; return
	        endif
	       endif
	      endif
	      if((associated(tens_in%data_real8).and.(.not.associated(tens_out%data_real8))).or.&
	        &((.not.associated(tens_in%data_real8)).and.associated(tens_out%data_real8))) then
	       tensor_block_compatible=.FALSE.; return
	      else
	       if(associated(tens_in%data_real8)) then
	        ls=size(tens_in%data_real8)
	        if(size(tens_out%data_real8).ne.ls.or.tens_out%tensor_block_size.ne.ls) then
	         tensor_block_compatible=.FALSE.; ierr=3; return
	        endif
	       endif
	      endif
	      if((associated(tens_in%data_cmplx4).and.(.not.associated(tens_out%data_cmplx4))).or.&
	        &((.not.associated(tens_in%data_cmplx4)).and.associated(tens_out%data_cmplx4))) then
	       tensor_block_compatible=.FALSE.; return
	      else
	       if(associated(tens_in%data_cmplx4)) then
	        ls=size(tens_in%data_cmplx4)
	        if(size(tens_out%data_cmplx4).ne.ls.or.tens_out%tensor_block_size.ne.ls) then
	         tensor_block_compatible=.FALSE.; ierr=4; return
	        endif
	       endif
	      endif
	      if((associated(tens_in%data_cmplx8).and.(.not.associated(tens_out%data_cmplx8))).or.&
	        &((.not.associated(tens_in%data_cmplx8)).and.associated(tens_out%data_cmplx8))) then
	       tensor_block_compatible=.FALSE.; return
	      else
	       if(associated(tens_in%data_cmplx8)) then
	        ls=size(tens_in%data_cmplx8)
	        if(size(tens_out%data_cmplx8).ne.ls.or.tens_out%tensor_block_size.ne.ls) then
	         tensor_block_compatible=.FALSE.; ierr=5; return
	        endif
	       endif
	      endif
	     endif
	    endif
	   endif
	  else
	   tensor_block_compatible=.FALSE.; ierr=6 !some of the %tensor_shape arrays were not allocated
	  endif
	 endif
	else
	 tensor_block_compatible=.FALSE.
	endif
	return
	end function tensor_block_compatible
!------------------------------------------------------------------
	subroutine tensor_block_mimic(tens_in,tens_out,ierr,transp) !SERIAL
!This subroutine copies the internal structure of a tensor block without copying the actual data.
!Optionally, it can also initialize the tensor shape according to a given permutation (O2N).
!INPUT:
! - tens_in - tensor block being mimicked;
! - transp(0:) - (optional) if present, the tensor shape will also be initialized according to this permutation (O2N);
!OUTPUT:
! - tens_out - output tensor block;
! - ierr - error code (0:success).
!NOTES:
! - Tensor block storage layouts are ignored.
	implicit none
	type(tensor_block_t), intent(in):: tens_in
	type(tensor_block_t), intent(inout):: tens_out
	integer, intent(in), optional:: transp(0:*)
	integer, intent(inout):: ierr
	integer i,j,k,l,m,n,k0,k1,k2,k3,ks,kf
	logical res

	ierr=0
	n=tens_in%tensor_shape%num_dim
	if(tens_out%tensor_shape%num_dim.ne.n.or.tens_out%tensor_block_size.ne.tens_in%tensor_block_size) then
	 call tensor_block_destroy(tens_out,ierr); if(ierr.ne.0) then; ierr=1; return; endif
!Allocate tensor shape:
	 if(n.gt.0) then
	  allocate(tens_out%tensor_shape%dim_extent(1:n),STAT=ierr); if(ierr.ne.0) then; ierr=2; return; endif
	  allocate(tens_out%tensor_shape%dim_divider(1:n),STAT=ierr); if(ierr.ne.0) then; ierr=3; return; endif
	  allocate(tens_out%tensor_shape%dim_group(1:n),STAT=ierr); if(ierr.ne.0) then; ierr=4; return; endif
	  res=tensor_block_alloc(tens_out,'sp',ierr,.TRUE.); if(ierr.ne.0) then; ierr=5; return; endif
	 endif
	 tens_out%tensor_shape%num_dim=n; tens_out%tensor_block_size=tens_in%tensor_block_size
	endif
	if(n.gt.0) then
	 if(present(transp)) then !adopt the tensor shape in full according to the given permutation (O2N)
	  tens_out%tensor_shape%dim_extent(transp(1:n))=tens_in%tensor_shape%dim_extent(1:n)
	  tens_out%tensor_shape%dim_divider(transp(1:n))=tens_in%tensor_shape%dim_divider(1:n)
	  tens_out%tensor_shape%dim_group(transp(1:n))=tens_in%tensor_shape%dim_group(1:n)
	 endif
!Allocate data arrays, if needed:
 !REAL4:
	 if(associated(tens_in%data_real4)) then
	  if(size(tens_in%data_real4).eq.tens_in%tensor_block_size) then
	   if(.not.associated(tens_out%data_real4)) then
!	    allocate(tens_out%data_real4(0:tens_in%tensor_block_size-1),STAT=ierr)
	    ierr=array_alloc(tens_out%data_real4,tens_in%tensor_block_size,base=0_LONGINT)
	    if(ierr.ne.0) then; ierr=6; return; endif
	    res=tensor_block_alloc(tens_out,'r4',ierr,.TRUE.); if(ierr.ne.0) then; ierr=7; return; endif
	   endif
	  else
	   ierr=8; return
	  endif
	 endif
 !REAL8:
	 if(associated(tens_in%data_real8)) then
	  if(size(tens_in%data_real8).eq.tens_in%tensor_block_size) then
	   if(.not.associated(tens_out%data_real8)) then
!	    allocate(tens_out%data_real8(0:tens_in%tensor_block_size-1),STAT=ierr)
	    ierr=array_alloc(tens_out%data_real8,tens_in%tensor_block_size,base=0_LONGINT)
	    if(ierr.ne.0) then; ierr=9; return; endif
	    res=tensor_block_alloc(tens_out,'r8',ierr,.TRUE.); if(ierr.ne.0) then; ierr=10; return; endif
	   endif
	  else
	   ierr=11; return
	  endif
	 endif
 !CMPLX4:
	 if(associated(tens_in%data_cmplx4)) then
	  if(size(tens_in%data_cmplx4).eq.tens_in%tensor_block_size) then
	   if(.not.associated(tens_out%data_cmplx4)) then
!	    allocate(tens_out%data_cmplx4(0:tens_in%tensor_block_size-1),STAT=ierr)
	    ierr=array_alloc(tens_out%data_cmplx4,tens_in%tensor_block_size,base=0_LONGINT)
	    if(ierr.ne.0) then; ierr=12; return; endif
	    res=tensor_block_alloc(tens_out,'c4',ierr,.TRUE.); if(ierr.ne.0) then; ierr=13; return; endif
	   endif
	  else
	   ierr=14; return
	  endif
	 endif
 !CMPLX8:
	 if(associated(tens_in%data_cmplx8)) then
	  if(size(tens_in%data_cmplx8).eq.tens_in%tensor_block_size) then
	   if(.not.associated(tens_out%data_cmplx8)) then
!	    allocate(tens_out%data_cmplx8(0:tens_in%tensor_block_size-1),STAT=ierr)
	    ierr=array_alloc(tens_out%data_cmplx8,tens_in%tensor_block_size,base=0_LONGINT)
	    if(ierr.ne.0) then; ierr=15; return; endif
	    res=tensor_block_alloc(tens_out,'c8',ierr,.TRUE.); if(ierr.ne.0) then; ierr=16; return; endif
	   endif
	  else
	   ierr=17; return
	  endif
	 endif
	endif
	return
	end subroutine tensor_block_mimic
!------------------------------------------------------------------------------------------------------
	subroutine tensor_block_create(shape_str,data_kind,tens_block,ierr,val_r4,val_r8,val_c4,val_c8) !PARALLEL
!This subroutine creates a tensor block <tens_block> based on the tensor shape specification string (TSSS) <shape_str>.
!FORMAT of <shape_str>:
!"(E1/D1{G1},E2/D2{G2},...)":
!  Ex is the extent of the dimension x (segment);
!  /Dx specifies an optional segment divider for the dimension x (lm_segment_size), 1<=Dx<=Ex (DEFAULT = Ex);
!      Ex MUST be a multiple of Dx.
!  {Gx} optionally specifies the symmetric group the dimension belongs to, Gx>=0 (default group 0 has no symmetry ordering).
!       Dimensions grouped together (group#>0) will obey a non-descending ordering from left to right.
!By default, the 1st dimension is the most minor one while the last is the most senior (Fortran-like).
!If the number of dimensions equals to zero, the %scalar_value field will be initialized.
!INPUT:
! - shape_str - tensor shape specification string (SSS);
! - data_kind - requested data kind, one of {"r4","r8","c4","c8"};
! - tens_block - tensor block;
! - val_r4/val_r8/val_c4/val_c8 - (optional) initialization value for different data kinds (otherwise a random fill will be invoked);
!OUTPUT:
! - tens_block - filled tensor_block;
! - ierr - error code (0: success);
!NOTES:
! - If the tensor block has already been allocated before, it will be reshaped and reinitialized.
! - If ordered dimensions are present, the data feed may not reflect a proper symmetry (antisymmetry)!
	implicit none
	character(*), intent(in):: shape_str
	character(2), intent(in):: data_kind
	type(tensor_block_t), intent(inout):: tens_block
	real(4), intent(in), optional:: val_r4
	real(8), intent(in), optional:: val_r8
	complex(4), intent(in), optional:: val_c4
	complex(8), intent(in), optional:: val_c8
	integer, intent(inout):: ierr

	ierr=0
	call tensor_block_shape_create(tens_block,shape_str,ierr); if(ierr.ne.0) then; ierr=1; return; endif
	ierr=tensor_block_shape_ok(tens_block); if(ierr.ne.0) then; ierr=2; return; endif
	select case(data_kind)
	case('r4','R4')
	 if(present(val_r4)) then
	  call tensor_block_init(data_kind,tens_block,ierr,val_r4=val_r4); if(ierr.ne.0) then; ierr=3; return; endif
	 else
	  if(present(val_r8)) then
	   call tensor_block_init(data_kind,tens_block,ierr,val_r8=val_r8); if(ierr.ne.0) then; ierr=4; return; endif
	  else
	   if(present(val_c4)) then
            call tensor_block_init(data_kind,tens_block,ierr,val_c4=val_c4); if(ierr.ne.0) then; ierr=5; return; endif
	   else
	    if(present(val_c8)) then
	     call tensor_block_init(data_kind,tens_block,ierr,val_c8=val_c8); if(ierr.ne.0) then; ierr=6; return; endif
	    else
	     call tensor_block_init(data_kind,tens_block,ierr); if(ierr.ne.0) then; ierr=7; return; endif
	    endif
	   endif
	  endif
	 endif
	case('r8','R8')
	 if(present(val_r8)) then
	  call tensor_block_init(data_kind,tens_block,ierr,val_r8=val_r8); if(ierr.ne.0) then; ierr=8; return; endif
	 else
	  if(present(val_r4)) then
	   call tensor_block_init(data_kind,tens_block,ierr,val_r4=val_r4); if(ierr.ne.0) then; ierr=9; return; endif
	  else
	   if(present(val_c4)) then
	    call tensor_block_init(data_kind,tens_block,ierr,val_c4=val_c4); if(ierr.ne.0) then; ierr=10; return; endif
	   else
	    if(present(val_c8)) then
	     call tensor_block_init(data_kind,tens_block,ierr,val_c8=val_c8); if(ierr.ne.0) then; ierr=11; return; endif
	    else
	     call tensor_block_init(data_kind,tens_block,ierr); if(ierr.ne.0) then; ierr=12; return; endif
	    endif
	   endif
	  endif
	 endif
	case('c4','C4')
	 if(present(val_c4)) then
	  call tensor_block_init(data_kind,tens_block,ierr,val_c4=val_c4); if(ierr.ne.0) then; ierr=13; return; endif
	 else
	  if(present(val_c8)) then
	   call tensor_block_init(data_kind,tens_block,ierr,val_c8=val_c8); if(ierr.ne.0) then; ierr=14; return; endif
	  else
	   if(present(val_r8)) then
	    call tensor_block_init(data_kind,tens_block,ierr,val_r8=val_r8); if(ierr.ne.0) then; ierr=15; return; endif
	   else
	    if(present(val_r4)) then
	     call tensor_block_init(data_kind,tens_block,ierr,val_r4=val_r4); if(ierr.ne.0) then; ierr=16; return; endif
	    else
	     call tensor_block_init(data_kind,tens_block,ierr); if(ierr.ne.0) then; ierr=17; return; endif
	    endif
	   endif
	  endif
	 endif
	case('c8','C8')
	 if(present(val_c8)) then
	  call tensor_block_init(data_kind,tens_block,ierr,val_c8=val_c8); if(ierr.ne.0) then; ierr=18; return; endif
	 else
	  if(present(val_c4)) then
	   call tensor_block_init(data_kind,tens_block,ierr,val_c4=val_c4); if(ierr.ne.0) then; ierr=19; return; endif
	  else
	   if(present(val_r8)) then
	    call tensor_block_init(data_kind,tens_block,ierr,val_r8=val_r8); if(ierr.ne.0) then; ierr=20; return; endif
	   else
	    if(present(val_r4)) then
	     call tensor_block_init(data_kind,tens_block,ierr,val_r4=val_r4); if(ierr.ne.0) then; ierr=21; return; endif
	    else
	     call tensor_block_init(data_kind,tens_block,ierr); if(ierr.ne.0) then; ierr=22; return; endif
	    endif
	   endif
	  endif
	 endif
	case default
	 ierr=23
	end select
	return
	end subroutine tensor_block_create
!------------------------------------------------------------------------------------------
	subroutine tensor_block_init(data_kind,tens_block,ierr,val_r4,val_r8,val_c4,val_c8) !PARALLEL
!This subroutine initializes a tensor block <tens_block> with either some value or random numbers.
!INPUT:
! - data_kind - requested data kind, one of {"r4","r8","c4","c8"};
! - tens_block - tensor block;
! - val_r4/val_r8/val_c4/val_c8 - (optional) if present, the tensor block is assigned the value <val> (otherwise, a random fill);
!OUTPUT:
! - tens_block - filled tensor block;
! - ierr - error code (0: success):
!                     -1: invalid (negative) tensor rank;
!                     -2: negative tensor size returned;
!                    x>0: invalid <tensor_shape> (zero/negative xth dimension extent);
!                    666: invalid <data_kind>;
!                    667: memory allocation failed;
!NOTES:
! - For tensors with a non-zero rank, the %scalar_value field will be set to the Euclidean norm of the tensor block.
! - Scalar tensors will be initialized with the <val_XX> value (if present), regardless of the <data_kind>.
! - In general, a tensor block may have dimension ordering (symmetry) restrictions.
!   In this case, the number fill done here might not reflect the proper symmetry (e.g., antisymmetry)!
	implicit none
!-----------------------------------------------------
	integer(LONGINT), parameter:: chunk_size=2**10
	integer(LONGINT), parameter:: vec_size=2**8
!--------------------------------------------------
	character(2), intent(in):: data_kind
	type(tensor_block_t), intent(inout):: tens_block
	real(4), intent(in), optional:: val_r4
	real(8), intent(in), optional:: val_r8
	complex(4), intent(in), optional:: val_c4
	complex(8), intent(in), optional:: val_c8
	integer, intent(inout):: ierr
	integer i,j,k,l,m,n,k0,k1,k2,k3,k4,ks,kf
	integer(LONGINT) tens_size,l0,l1
	real(8) vec_r8(0:vec_size-1),valr8,val,rnd_buf(2)
	real(4) vec_r4(0:vec_size-1),valr4
	complex(4) vec_c4(0:vec_size-1),valc4
	complex(8) vec_c8(0:vec_size-1),valc8
	logical res

	ierr=0; tens_block%tensor_block_size=tensor_block_shape_size(tens_block,ierr)
	if(ierr.ne.0) then; ierr=1; return; endif
	if(tens_block%tensor_block_size.le.0_LONGINT) then; ierr=2; return; endif
	if(tens_block%tensor_shape%num_dim.eq.0) then !scalar tensor
	 if(associated(tens_block%data_real4)) then
	  if(tensor_block_alloc(tens_block,'r4',ierr)) then
	   if(ierr.ne.0) then; ierr=3; return; endif
!	   deallocate(tens_block%data_real4,STAT=ierr)
	   call array_free(tens_block%data_real4,ierr)
	   if(ierr.ne.0) then; ierr=4; return; endif
	   res=tensor_block_alloc(tens_block,'r4',ierr,.FALSE.); if(ierr.ne.0) then; ierr=5; return; endif
	  else
	   if(ierr.ne.0) then; ierr=6; return; endif
	   nullify(tens_block%data_real4)
	  endif
	 endif
	 if(associated(tens_block%data_real8)) then
	  if(tensor_block_alloc(tens_block,'r8',ierr)) then
	   if(ierr.ne.0) then; ierr=7; return; endif
!	   deallocate(tens_block%data_real8,STAT=ierr)
	   call array_free(tens_block%data_real8,ierr)
	   if(ierr.ne.0) then; ierr=8; return; endif
	   res=tensor_block_alloc(tens_block,'r8',ierr,.FALSE.); if(ierr.ne.0) then; ierr=9; return; endif
	  else
	   if(ierr.ne.0) then; ierr=10; return; endif
	   nullify(tens_block%data_real8)
	  endif
	 endif
	 if(associated(tens_block%data_cmplx4)) then
	  if(tensor_block_alloc(tens_block,'c4',ierr)) then
	   if(ierr.ne.0) then; ierr=11; return; endif
!	   deallocate(tens_block%data_cmplx4,STAT=ierr)
	   call array_free(tens_block%data_cmplx4,ierr)
	   if(ierr.ne.0) then; ierr=12; return; endif
	   res=tensor_block_alloc(tens_block,'c4',ierr,.FALSE.); if(ierr.ne.0) then; ierr=13; return; endif
	  else
	   if(ierr.ne.0) then; ierr=14; return; endif
	   nullify(tens_block%data_cmplx4)
	  endif
	 endif
	 if(associated(tens_block%data_cmplx8)) then
	  if(tensor_block_alloc(tens_block,'c8',ierr)) then
	   if(ierr.ne.0) then; ierr=15; return; endif
!	   deallocate(tens_block%data_cmplx8,STAT=ierr)
	   call array_free(tens_block%data_cmplx8,ierr)
	   if(ierr.ne.0) then; ierr=16; return; endif
	   res=tensor_block_alloc(tens_block,'c8',ierr,.FALSE.); if(ierr.ne.0) then; ierr=17; return; endif
	  else
	   if(ierr.ne.0) then; ierr=18; return; endif
	   nullify(tens_block%data_cmplx8)
	  endif
	 endif
	 if(tens_block%tensor_block_size.ne.1_LONGINT) then; ierr=19; return; endif
	endif
	select case(data_kind)
	case('r4','R4')
	 if(tens_block%tensor_shape%num_dim.gt.0) then !true tensor
	  if(associated(tens_block%data_real4)) then
	   if(size(tens_block%data_real4).ne.tens_block%tensor_block_size) then
	    if(tensor_block_alloc(tens_block,'r4',ierr)) then
	     if(ierr.ne.0) then; ierr=16; return; endif
!	     deallocate(tens_block%data_real4,STAT=ierr)
	     call array_free(tens_block%data_real4,ierr)
	     if(ierr.ne.0) then; ierr=17; return; endif
	     res=tensor_block_alloc(tens_block,'r4',ierr,.FALSE.); if(ierr.ne.0) then; ierr=18; return; endif
	    else
	     if(ierr.ne.0) then; ierr=19; return; endif
	     nullify(tens_block%data_real4)
	    endif
!	    allocate(tens_block%data_real4(0:tens_block%tensor_block_size-1),STAT=ierr)
	    ierr=array_alloc(tens_block%data_real4,tens_block%tensor_block_size,base=0_LONGINT)
	    if(ierr.ne.0) then; ierr=20; return; endif
	    res=tensor_block_alloc(tens_block,'r4',ierr,.TRUE.); if(ierr.ne.0) then; ierr=21; return; endif
	   endif
	  else
!	   allocate(tens_block%data_real4(0:tens_block%tensor_block_size-1),STAT=ierr)
	   ierr=array_alloc(tens_block%data_real4,tens_block%tensor_block_size,base=0_LONGINT)
	   if(ierr.ne.0) then; ierr=22; return; endif
	   res=tensor_block_alloc(tens_block,'r4',ierr,.TRUE.); if(ierr.ne.0) then; ierr=23; return; endif
	  endif
	  if(present(val_r4).or.present(val_r8).or.present(val_c4).or.present(val_c8)) then !constant fill
	   if(present(val_r4)) then
	    valr4=val_r4
	   else
	    if(present(val_r8)) then
	     valr4=real(val_r8,4)
	    else
	     if(present(val_c4)) then
	      valr4=cmplx4_to_real4(val_c4)
	     else
	      if(present(val_c8)) then
	       valr4=real(cmplx8_to_real8(val_c8),4)
	      endif
	     endif
	    endif
	   endif
	   vec_r4(0_LONGINT:vec_size-1_LONGINT)=valr4
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(l0,l1)
!$OMP DO SCHEDULE(GUIDED)
	   do l0=0_LONGINT,tens_block%tensor_block_size-1_LONGINT-mod(tens_block%tensor_block_size,vec_size),vec_size
	    do l1=0_LONGINT,vec_size-1_LONGINT; tens_block%data_real4(l0+l1)=vec_r4(l1); enddo
	   enddo
!$OMP END DO NOWAIT
!$OMP MASTER
	   tens_block%data_real4(tens_block%tensor_block_size-mod(tens_block%tensor_block_size,vec_size):&
	                        &tens_block%tensor_block_size-1_LONGINT)=valr4
!$OMP END MASTER
!$OMP END PARALLEL
	  else !random fill
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(l0,l1)
!$OMP DO SCHEDULE(GUIDED)
	   do l0=0_LONGINT,tens_block%tensor_block_size-1_LONGINT,chunk_size
	    l1=min(l0+chunk_size-1_LONGINT,tens_block%tensor_block_size-1_LONGINT)
	    call random_number(tens_block%data_real4(l0:l1))
	   enddo
!$OMP END DO
!$OMP END PARALLEL
	  endif
	  if(DATA_KIND_SYNC) then
	   valr8=tensor_block_norm2(tens_block,ierr,'r4'); if(ierr.ne.0) then; ierr=24; return; endif
	   tens_block%scalar_value=cmplx(dsqrt(valr8),0d0,kind=8)
	  endif
	 else !scalar
	  if(present(val_r4)) then
	   tens_block%scalar_value=cmplx(real(val_r4,8),0d0,kind=8)
	  else
	   if(present(val_r8)) then
	    tens_block%scalar_value=cmplx(val_r8,0d0,kind=8)
	   else
	    if(present(val_c4)) then
	     tens_block%scalar_value=cmplx(val_c4,kind=8)
	    else
	     if(present(val_c8)) then
	      tens_block%scalar_value=val_c8
	     else
	      call random_number(val); tens_block%scalar_value=cmplx(val,0d0,kind=8)
	     endif
	    endif
	   endif
	  endif
	 endif
	case('r8','R8')
	 if(tens_block%tensor_shape%num_dim.gt.0) then !true tensor
	  if(associated(tens_block%data_real8)) then
	   if(size(tens_block%data_real8).ne.tens_block%tensor_block_size) then
	    if(tensor_block_alloc(tens_block,'r8',ierr)) then
	     if(ierr.ne.0) then; ierr=25; return; endif
!	     deallocate(tens_block%data_real8,STAT=ierr)
	     call array_free(tens_block%data_real8,ierr)
	     if(ierr.ne.0) then; ierr=26; return; endif
	     res=tensor_block_alloc(tens_block,'r8',ierr,.FALSE.); if(ierr.ne.0) then; ierr=27; return; endif
	    else
	     if(ierr.ne.0) then; ierr=28; return; endif
	     nullify(tens_block%data_real8)
	    endif
!	    allocate(tens_block%data_real8(0:tens_block%tensor_block_size-1),STAT=ierr)
	    ierr=array_alloc(tens_block%data_real8,tens_block%tensor_block_size,base=0_LONGINT)
	    if(ierr.ne.0) then; ierr=29; return; endif
	    res=tensor_block_alloc(tens_block,'r8',ierr,.TRUE.); if(ierr.ne.0) then; ierr=30; return; endif
	   endif
	  else
!	   allocate(tens_block%data_real8(0:tens_block%tensor_block_size-1),STAT=ierr)
	   ierr=array_alloc(tens_block%data_real8,tens_block%tensor_block_size,base=0_LONGINT)
	   if(ierr.ne.0) then; ierr=31; return; endif
	   res=tensor_block_alloc(tens_block,'r8',ierr,.TRUE.); if(ierr.ne.0) then; ierr=32; return; endif
	  endif
	  if(present(val_r4).or.present(val_r8).or.present(val_c4).or.present(val_c8)) then !constant fill
	   if(present(val_r8)) then
	    valr8=val_r8
	   else
	    if(present(val_r4)) then
	     valr8=real(val_r4,8)
	    else
	     if(present(val_c4)) then
	      valr8=real(cmplx4_to_real4(val_c4),8)
	     else
	      if(present(val_c8)) then
	       valr8=cmplx8_to_real8(val_c8)
	      endif
	     endif
	    endif
	   endif
	   vec_r8(0_LONGINT:vec_size-1_LONGINT)=valr8
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(l0,l1)
!$OMP DO SCHEDULE(GUIDED)
	   do l0=0_LONGINT,tens_block%tensor_block_size-1_LONGINT-mod(tens_block%tensor_block_size,vec_size),vec_size
	    do l1=0_LONGINT,vec_size-1_LONGINT; tens_block%data_real8(l0+l1)=vec_r8(l1); enddo
	   enddo
!$OMP END DO NOWAIT
!$OMP MASTER
	   tens_block%data_real8(tens_block%tensor_block_size-mod(tens_block%tensor_block_size,vec_size):&
	                        &tens_block%tensor_block_size-1_LONGINT)=valr8
!$OMP END MASTER
!$OMP END PARALLEL
	  else !random fill
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(l0,l1)
!$OMP DO SCHEDULE(GUIDED)
	   do l0=0_LONGINT,tens_block%tensor_block_size-1_LONGINT,chunk_size
	    l1=min(l0+chunk_size-1_LONGINT,tens_block%tensor_block_size-1_LONGINT)
	    call random_number(tens_block%data_real8(l0:l1))
	   enddo
!$OMP END DO
!$OMP END PARALLEL
	  endif
	  if(DATA_KIND_SYNC) then
	   valr8=tensor_block_norm2(tens_block,ierr,'r8'); if(ierr.ne.0) then; ierr=33; return; endif
	   tens_block%scalar_value=cmplx(dsqrt(valr8),0d0,kind=8)
	  endif
	 else !scalar
	  if(present(val_r8)) then
	   tens_block%scalar_value=cmplx(val_r8,0d0,kind=8)
	  else
	   if(present(val_r4)) then
	    tens_block%scalar_value=cmplx(real(val_r4,8),0d0,kind=8)
	   else
	    if(present(val_c4)) then
	     tens_block%scalar_value=cmplx(val_c4,kind=8)
	    else
	     if(present(val_c8)) then
	      tens_block%scalar_value=val_c8
	     else
	      call random_number(val); tens_block%scalar_value=cmplx(val,0d0,kind=8)
	     endif
	    endif
	   endif
	  endif
	 endif
	case('c4','C4')
	 if(tens_block%tensor_shape%num_dim.gt.0) then !true tensor
	  if(associated(tens_block%data_cmplx4)) then
	   if(size(tens_block%data_cmplx4).ne.tens_block%tensor_block_size) then
	    if(tensor_block_alloc(tens_block,'c4',ierr)) then
	     if(ierr.ne.0) then; ierr=34; return; endif
!	     deallocate(tens_block%data_cmplx4,STAT=ierr)
	     call array_free(tens_block%data_cmplx4,ierr)
	     if(ierr.ne.0) then; ierr=35; return; endif
	     res=tensor_block_alloc(tens_block,'c4',ierr,.FALSE.); if(ierr.ne.0) then; ierr=36; return; endif
	    else
	     if(ierr.ne.0) then; ierr=37; return; endif
	     nullify(tens_block%data_cmplx4)
	    endif
!	    allocate(tens_block%data_cmplx4(0:tens_block%tensor_block_size-1),STAT=ierr)
	    ierr=array_alloc(tens_block%data_cmplx4,tens_block%tensor_block_size,base=0_LONGINT)
	    if(ierr.ne.0) then; ierr=38; return; endif
	    res=tensor_block_alloc(tens_block,'c4',ierr,.TRUE.); if(ierr.ne.0) then; ierr=39; return; endif
	   endif
	  else
!	   allocate(tens_block%data_cmplx4(0:tens_block%tensor_block_size-1),STAT=ierr)
	   ierr=array_alloc(tens_block%data_cmplx4,tens_block%tensor_block_size,base=0_LONGINT)
	   if(ierr.ne.0) then; ierr=40; return; endif
	   res=tensor_block_alloc(tens_block,'c4',ierr,.TRUE.); if(ierr.ne.0) then; ierr=41; return; endif
	  endif
	  if(present(val_r4).or.present(val_r8).or.present(val_c4).or.present(val_c8)) then !constant fill
	   if(present(val_c4)) then
	    valc4=val_c4
	   else
	    if(present(val_c8)) then
	     valc4=cmplx(val_c8,kind=4)
	    else
	     if(present(val_r8)) then
	      valc4=cmplx(val_r8,0d0,kind=4)
	     else
	      if(present(val_r4)) then
	       valc4=cmplx(val_r4,0.0,kind=4)
	      endif
	     endif
	    endif
	   endif
	   vec_c4(0_LONGINT:vec_size-1_LONGINT)=valc4
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(l0,l1)
!$OMP DO SCHEDULE(GUIDED)
	   do l0=0_LONGINT,tens_block%tensor_block_size-1_LONGINT-mod(tens_block%tensor_block_size,vec_size),vec_size
	    do l1=0_LONGINT,vec_size-1_LONGINT; tens_block%data_cmplx4(l0+l1)=vec_c4(l1); enddo
	   enddo
!$OMP END DO NOWAIT
!$OMP MASTER
	   tens_block%data_cmplx4(tens_block%tensor_block_size-mod(tens_block%tensor_block_size,vec_size):&
	                         &tens_block%tensor_block_size-1_LONGINT)=valc4
!$OMP END MASTER
!$OMP END PARALLEL
	  else !random fill
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(l0,rnd_buf)
!$OMP DO SCHEDULE(GUIDED)
	   do l0=0_LONGINT,tens_block%tensor_block_size-1_LONGINT
	    call random_number(rnd_buf(1:2)); tens_block%data_cmplx4(l0)=cmplx(rnd_buf(1),rnd_buf(2),kind=4)
	   enddo
!$OMP END DO
!$OMP END PARALLEL
	  endif
	  if(DATA_KIND_SYNC) then
	   valr8=tensor_block_norm2(tens_block,ierr,'c4'); if(ierr.ne.0) then; ierr=42; return; endif
	   tens_block%scalar_value=cmplx(dsqrt(valr8),0d0,kind=8)
	  endif
	 else !scalar
	  if(present(val_c4)) then
	   tens_block%scalar_value=cmplx(val_c4,kind=8)
	  else
	   if(present(val_c8)) then
	    tens_block%scalar_value=val_c8
	   else
	    if(present(val_r8)) then
	     tens_block%scalar_value=cmplx(val_r8,0d0,kind=8)
	    else
	     if(present(val_r4)) then
	      tens_block%scalar_value=cmplx(real(val_r4,8),0d0,kind=8)
	     else
	      call random_number(val); call random_number(valr8); tens_block%scalar_value=cmplx(val,valr8,kind=8)
	     endif
	    endif
	   endif
	  endif
	 endif
	case('c8','C8')
	 if(tens_block%tensor_shape%num_dim.gt.0) then !true tensor
	  if(associated(tens_block%data_cmplx8)) then
	   if(size(tens_block%data_cmplx8).ne.tens_block%tensor_block_size) then
	    if(tensor_block_alloc(tens_block,'c8',ierr)) then
	     if(ierr.ne.0) then; ierr=43; return; endif
!	     deallocate(tens_block%data_cmplx8,STAT=ierr)
	     call array_free(tens_block%data_cmplx8,ierr)
	     if(ierr.ne.0) then; ierr=44; return; endif
	     res=tensor_block_alloc(tens_block,'c8',ierr,.FALSE.); if(ierr.ne.0) then; ierr=45; return; endif
	    else
	     if(ierr.ne.0) then; ierr=46; return; endif
	     nullify(tens_block%data_cmplx8)
	    endif
!	    allocate(tens_block%data_cmplx8(0:tens_block%tensor_block_size-1),STAT=ierr)
	    ierr=array_alloc(tens_block%data_cmplx8,tens_block%tensor_block_size,base=0_LONGINT)
	    if(ierr.ne.0) then; ierr=47; return; endif
	    res=tensor_block_alloc(tens_block,'c8',ierr,.TRUE.); if(ierr.ne.0) then; ierr=48; return; endif
	   endif
	  else
!	   allocate(tens_block%data_cmplx8(0:tens_block%tensor_block_size-1),STAT=ierr)
	   ierr=array_alloc(tens_block%data_cmplx8,tens_block%tensor_block_size,base=0_LONGINT)
	   if(ierr.ne.0) then; ierr=49; return; endif
	   res=tensor_block_alloc(tens_block,'c8',ierr,.TRUE.); if(ierr.ne.0) then; ierr=50; return; endif
	  endif
	  if(present(val_r4).or.present(val_r8).or.present(val_c4).or.present(val_c8)) then !constant fill
	   if(present(val_c8)) then
	    valc8=val_c8
	   else
	    if(present(val_c4)) then
	     valc8=cmplx(val_c4,kind=8)
	    else
	     if(present(val_r8)) then
	      valc8=cmplx(val_r8,0d0,kind=8)
	     else
	      if(present(val_r4)) then
	       valc8=cmplx(real(val_r4,8),0d0,kind=8)
	      endif
	     endif
	    endif
	   endif
	   vec_c8(0_LONGINT:vec_size-1_LONGINT)=valc8
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(l0,l1)
!$OMP DO SCHEDULE(GUIDED)
	   do l0=0_LONGINT,tens_block%tensor_block_size-1_LONGINT-mod(tens_block%tensor_block_size,vec_size),vec_size
	    do l1=0_LONGINT,vec_size-1_LONGINT; tens_block%data_cmplx8(l0+l1)=vec_c8(l1); enddo
	   enddo
!$OMP END DO NOWAIT
!$OMP MASTER
	   tens_block%data_cmplx8(tens_block%tensor_block_size-mod(tens_block%tensor_block_size,vec_size):&
	                         &tens_block%tensor_block_size-1_LONGINT)=valc8
!$OMP END MASTER
!$OMP END PARALLEL
	  else !random fill
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(l0,rnd_buf)
!$OMP DO SCHEDULE(GUIDED)
	   do l0=0_LONGINT,tens_block%tensor_block_size-1_LONGINT
	    call random_number(rnd_buf(1:2)); tens_block%data_cmplx8(l0)=cmplx(rnd_buf(1),rnd_buf(2),kind=8)
	   enddo
!$OMP END DO
!$OMP END PARALLEL
	  endif
	  if(DATA_KIND_SYNC) then
	   valr8=tensor_block_norm2(tens_block,ierr,'c8'); if(ierr.ne.0) then; ierr=51; return; endif
	   tens_block%scalar_value=cmplx(dsqrt(valr8),0d0,kind=8)
	  endif
	 else !scalar
	  if(present(val_c8)) then
	   tens_block%scalar_value=val_c8
	  else
	   if(present(val_c4)) then
	    tens_block%scalar_value=cmplx(val_c4,kind=8)
	   else
	    if(present(val_r8)) then
	     tens_block%scalar_value=cmplx(val_r8,0d0,kind=8)
	    else
	     if(present(val_r4)) then
	      tens_block%scalar_value=cmplx(real(val_r4,8),0d0,kind=8)
	     else
	      call random_number(val); call random_number(valr8); tens_block%scalar_value=cmplx(val,valr8,kind=8)
	     endif
	    endif
	   endif
	  endif
	 endif
	case default
	 ierr=52
	end select
	return
	end subroutine tensor_block_init
!--------------------------------------------------------
        logical function tensor_block_is_empty(tens,ierr)
!Returns TRUE if the tensor block is empty, FALSE otherwise.
         implicit none
         type(tensor_block_t), intent(in):: tens !in: tensor block
         integer, intent(out), optional:: ierr   !out: error code (0:success)
         integer:: errc

         errc=0; tensor_block_is_empty=.TRUE.
         if(tens%tensor_shape%num_dim.ge.0) then
          tensor_block_is_empty=.FALSE.
          if(tens%tensor_shape%num_dim.le.MAX_TENSOR_RANK) then
           if(.not.associated(tens%tensor_shape%dim_extent).and.tens%tensor_shape%num_dim.gt.0) then
            errc=2; tensor_block_is_empty=.TRUE.
           endif
          else
           errc=1; tensor_block_is_empty=.TRUE.
          endif
         endif
         if(present(ierr)) ierr=errc
         return
        end function tensor_block_is_empty
!------------------------------------------------------------------------------
        subroutine tensor_block_assoc(tens,tens_shape,data_kind,tens_body,ierr)
!Constructs a <tensor_block_t> object based on externally provided data.
!No new memory is allocated (only pointer association occurs).
         implicit none
         type(tensor_block_t), intent(inout):: tens    !inout: tensor block (empty on entrance, defined on exit)
         type(tensor_shape_t), intent(in):: tens_shape !in: tensor block shape
         integer(C_INT), intent(in):: data_kind        !in: data kind
         type(C_PTR), intent(in):: tens_body           !in: tensor body (elements of <data_kind>)
         integer(C_INT), intent(out), optional:: ierr  !out: error code (0:success)
         real(4), pointer, contiguous:: r4p(:)
         real(8), pointer, contiguous:: r8p(:)
         complex(4), pointer, contiguous:: c4p(:)
         complex(8), pointer, contiguous:: c8p(:)

         ierr=0
         if(tensor_block_is_empty(tens)) then
          if(tens_shape%num_dim.ge.0.and.tens_shape%num_dim.le.MAX_TENSOR_RANK) then
           if(c_associated(tens_body)) then
            tens%ptr_alloc=0
            tens%tensor_shape=tens_shape !pointer components are pointer associated only
            tens%tensor_block_size=tensor_block_shape_size(tens,ierr); if(ierr.ne.0) then; ierr=1; return; endif
            if(tens%tensor_block_size.le.0) then; ierr=2; return; endif
            if(tens%tensor_shape%num_dim.ge.0) then
             select case(data_kind)
             case(R4)
              call c_f_pointer(tens_body,r4p,shape=(/tens%tensor_block_size/))
              if(tens%tensor_shape%num_dim.gt.0) then
               tens%data_real4(0:)=>r4p; tens%scalar_value=(0d0,0d0)
              else
               tens%scalar_value=cmplx(real(r4p(lbound(r4p,1)),8),0d0,kind=8)
              endif
              r4p=>NULL()
             case(R8)
              call c_f_pointer(tens_body,r8p,shape=(/tens%tensor_block_size/))
              if(tens%tensor_shape%num_dim.gt.0) then
               tens%data_real8(0:)=>r8p; tens%scalar_value=(0d0,0d0)
              else
               tens%scalar_value=cmplx(r8p(lbound(r8p,1)),0d0,kind=8)
              endif
              r8p=>NULL()
             case(C4)
              call c_f_pointer(tens_body,c4p,shape=(/tens%tensor_block_size/))
              if(tens%tensor_shape%num_dim.gt.0) then
               tens%data_cmplx4(0:)=>c4p; tens%scalar_value=(0d0,0d0)
              else
               tens%scalar_value=c4p(lbound(c4p,1))
              endif
              c4p=>NULL()
             case(C8)
              call c_f_pointer(tens_body,c8p,shape=(/tens%tensor_block_size/))
              if(tens%tensor_shape%num_dim.gt.0) then
               tens%data_cmplx8(0:)=>c8p; tens%scalar_value=(0d0,0d0)
              else
               tens%scalar_value=c8p(lbound(c8p,1))
              endif
              c8p=>NULL()
             case default
              ierr=3
             end select
            else
             ierr=4
            endif
           else
            ierr=5 !tensor body is absent
           endif
          else
           ierr=6 !tensor shape is empty
          endif
         else
          ierr=7 !tensor block is not empty
         endif
         return
        end subroutine tensor_block_assoc
!-------------------------------------------------------
	subroutine tensor_block_destroy(tens_block,ierr) !SERIAL
!This subroutine destroys a tensor block <tens_block>.
	implicit none
	type(tensor_block_t), intent(inout):: tens_block
	integer, intent(inout):: ierr
	integer i
	logical res

	ierr=0; tens_block%tensor_block_size=0_LONGINT
!REAL4:
	if(associated(tens_block%data_real4)) then
	 if(tensor_block_alloc(tens_block,'r4',ierr)) then
	  if(ierr.ne.0) then; ierr=1; return; endif
!	  deallocate(tens_block%data_real4,STAT=ierr)
	  call array_free(tens_block%data_real4,ierr)
	  if(ierr.ne.0) then; ierr=2; return; endif
	  res=tensor_block_alloc(tens_block,'r4',ierr,.FALSE.); if(ierr.ne.0) then; ierr=3; return; endif
	 else
	  if(ierr.ne.0) then; ierr=4; return; endif
!	  deallocate(tens_block%data_real4,STAT=i) !debug
	  nullify(tens_block%data_real4)
	 endif
!	else
!	 res=tensor_block_alloc(tens_block,'r4',ierr,.FALSE.); if(ierr.ne.0) then; ierr=5; return; endif
	endif
!REAL8:
	if(associated(tens_block%data_real8)) then
	 if(tensor_block_alloc(tens_block,'r8',ierr)) then
	  if(ierr.ne.0) then; ierr=6; return; endif
!	  deallocate(tens_block%data_real8,STAT=ierr)
	  call array_free(tens_block%data_real8,ierr)
	  if(ierr.ne.0) then; ierr=7; return; endif
	  res=tensor_block_alloc(tens_block,'r8',ierr,.FALSE.); if(ierr.ne.0) then; ierr=8; return; endif
	 else
	  if(ierr.ne.0) then; ierr=9; return; endif
!	  deallocate(tens_block%data_real8,STAT=i) !debug
	  nullify(tens_block%data_real8)
	 endif
!	else
!	 res=tensor_block_alloc(tens_block,'r8',ierr,.FALSE.); if(ierr.ne.0) then; ierr=10; return; endif
	endif
!COMPLEX4:
	if(associated(tens_block%data_cmplx4)) then
	 if(tensor_block_alloc(tens_block,'c4',ierr)) then
	  if(ierr.ne.0) then; ierr=11; return; endif
!	  deallocate(tens_block%data_cmplx4,STAT=ierr)
	  call array_free(tens_block%data_cmplx4,ierr)
	  if(ierr.ne.0) then; ierr=12; return; endif
	  res=tensor_block_alloc(tens_block,'c4',ierr,.FALSE.); if(ierr.ne.0) then; ierr=13; return; endif
	 else
	  if(ierr.ne.0) then; ierr=14; return; endif
!	  deallocate(tens_block%data_cmplx4,STAT=i) !debug
	  nullify(tens_block%data_cmplx4)
	 endif
!	else
!	 res=tensor_block_alloc(tens_block,'c4',ierr,.FALSE.); if(ierr.ne.0) then; ierr=15; return; endif
	endif
!COMPLEX8:
	if(associated(tens_block%data_cmplx8)) then
	 if(tensor_block_alloc(tens_block,'c8',ierr)) then
	  if(ierr.ne.0) then; ierr=16; return; endif
!	  deallocate(tens_block%data_cmplx8,STAT=ierr)
	  call array_free(tens_block%data_cmplx8,ierr)
	  if(ierr.ne.0) then; ierr=17; return; endif
	  res=tensor_block_alloc(tens_block,'c8',ierr,.FALSE.); if(ierr.ne.0) then; ierr=18; return; endif
	 else
	  if(ierr.ne.0) then; ierr=19; return; endif
!	  deallocate(tens_block%data_cmplx8,STAT=i) !debug
	  nullify(tens_block%data_cmplx8)
	 endif
!	else
!	 res=tensor_block_alloc(tens_block,'c8',ierr,.FALSE.); if(ierr.ne.0) then; ierr=20; return; endif
	endif
!SCALAR:
	tens_block%scalar_value=(0d0,0d0)
!TENSOR SHAPE:
	tens_block%tensor_shape%num_dim=-1
	if(tensor_block_alloc(tens_block,'sp',ierr)) then
	 if(ierr.ne.0) then; ierr=21; return; endif
	 deallocate(tens_block%tensor_shape%dim_extent,STAT=ierr); if(ierr.ne.0) then; ierr=22; return; endif
	 deallocate(tens_block%tensor_shape%dim_divider,STAT=ierr); if(ierr.ne.0) then; ierr=23; return; endif
	 deallocate(tens_block%tensor_shape%dim_group,STAT=ierr); if(ierr.ne.0) then; ierr=24; return; endif
	 res=tensor_block_alloc(tens_block,'sp',ierr,.FALSE.); if(ierr.ne.0) then; ierr=25; return; endif
	else
	 if(ierr.ne.0) then; ierr=26; return; endif
	 nullify(tens_block%tensor_shape%dim_extent)
	 nullify(tens_block%tensor_shape%dim_divider)
	 nullify(tens_block%tensor_shape%dim_group)
	endif
	tens_block%ptr_alloc=0
	return
	end subroutine tensor_block_destroy
!-------------------------------------------------------------------
	subroutine tensor_block_sync(tens,mast_kind,ierr,slave_kind) !PARALLEL
!This subroutine allocates and/or synchronizes different data kinds within a tensor block.
!For tensors of positive rank, the %scalar_value field will contain the Euclidean (Frobenius) norm of the tensor block.
!Note that basic tensor operations do not have to keep different data kinds consistent, so it is fully on the user!
!INPUT:
! - tens - tensor block;
! - mast_kind - master data kind, one of {'r4','r8','c4','c8'};
! - slave_kind - (optional) slave data kind, one of {'r4','r8','c4','c8', or '--'} (the latter means to destroy the master data kind);
!                If absent, all allocated data kinds will be syncronized with the master data kind.
!OUTPUT:
! - tens - modified tensor block;
! - ierr - error code (0:success).
!NOTES:
! - All data kinds in rank-0 tensor blocks (scalars) are mapped to 'c8'. An attempt to destroy it will cause an error.
! - An attempt to destroy the only data kind present in the tensor block will cause an arror.
! - Non-allocated tensor blocks are ignored.
! - The tensor block storage layout does not matter here since it only affects the access pattern.
	implicit none
	type(tensor_block_t), intent(inout):: tens
	character(2), intent(in):: mast_kind
	character(2), intent(in), optional:: slave_kind
	integer, intent(inout):: ierr
	integer i,j,k,l,m
	character(2) slk
	integer(LONGINT) l0,ls
	real(4) val_r4
	real(8) val_r8
	complex(4) val_c4
	complex(8) val_c8
	logical res

	ierr=0
	if(mast_kind.ne.'r4'.and.mast_kind.ne.'R4'.and.&
	  &mast_kind.ne.'r8'.and.mast_kind.ne.'R8'.and.&
	  &mast_kind.ne.'c4'.and.mast_kind.ne.'C4'.and.&
	  &mast_kind.ne.'c8'.and.mast_kind.ne.'C8') then; ierr=1; return; endif
	if(present(slave_kind)) then; slk=slave_kind; else; slk='  '; endif
	if(tens%tensor_shape%num_dim.gt.0) then !true tensor
	 if(((mast_kind.eq.'r4'.or.mast_kind.eq.'R4').and.associated(tens%data_real4)).or.&
	   &((mast_kind.eq.'r8'.or.mast_kind.eq.'R8').and.associated(tens%data_real8)).or.&
	   &((mast_kind.eq.'c4'.or.mast_kind.eq.'C4').and.associated(tens%data_cmplx4)).or.&
	   &((mast_kind.eq.'c8'.or.mast_kind.eq.'C8').and.associated(tens%data_cmplx8))) then
	  ls=tens%tensor_block_size
	  if(slk.eq.'--') then !destroy master data kind
	   select case(mast_kind)
	   case('r4','R4')
	    if(associated(tens%data_real8).or.associated(tens%data_cmplx4).or.associated(tens%data_cmplx8)) then
	     if(tensor_block_alloc(tens,'r4',ierr)) then
	      if(ierr.ne.0) then; ierr=2; return; endif
!	      deallocate(tens%data_real4,STAT=ierr)
	      call array_free(tens%data_real4,ierr)
	      if(ierr.ne.0) then; ierr=3; return; endif
	      res=tensor_block_alloc(tens,'r4',ierr,.FALSE.); if(ierr.ne.0) then; ierr=4; return; endif
	     else
	      if(ierr.ne.0) then; ierr=5; return; endif
	      nullify(tens%data_real4)
	     endif
	    else
	     ierr=6; return
	    endif
	   case('r8','R8')
	    if(associated(tens%data_real4).or.associated(tens%data_cmplx4).or.associated(tens%data_cmplx8)) then
	     if(tensor_block_alloc(tens,'r8',ierr)) then
	      if(ierr.ne.0) then; ierr=7; return; endif
!	      deallocate(tens%data_real8,STAT=ierr)
	      call array_free(tens%data_real8,ierr)
	      if(ierr.ne.0) then; ierr=8; return; endif
	      res=tensor_block_alloc(tens,'r8',ierr,.FALSE.); if(ierr.ne.0) then; ierr=9; return; endif
	     else
	      if(ierr.ne.0) then; ierr=10; return; endif
	      nullify(tens%data_real8)
	     endif
	    else
	     ierr=11; return
	    endif
	   case('c4','C4')
	    if(associated(tens%data_real4).or.associated(tens%data_real8).or.associated(tens%data_cmplx8)) then
	     if(tensor_block_alloc(tens,'c4',ierr)) then
	      if(ierr.ne.0) then; ierr=12; return; endif
!	      deallocate(tens%data_cmplx4,STAT=ierr)
	      call array_free(tens%data_cmplx4,ierr)
	      if(ierr.ne.0) then; ierr=13; return; endif
	      res=tensor_block_alloc(tens,'c4',ierr,.FALSE.); if(ierr.ne.0) then; ierr=14; return; endif
	     else
	      if(ierr.ne.0) then; ierr=15; return; endif
	      nullify(tens%data_cmplx4)
	     endif
	    else
	     ierr=16; return
	    endif
	   case('c8','C8')
	    if(associated(tens%data_real4).or.associated(tens%data_real8).or.associated(tens%data_cmplx4)) then
	     if(tensor_block_alloc(tens,'c8',ierr)) then
	      if(ierr.ne.0) then; ierr=17; return; endif
!	      deallocate(tens%data_cmplx8,STAT=ierr)
	      call array_free(tens%data_cmplx8,ierr)
	      if(ierr.ne.0) then; ierr=18; return; endif
	      res=tensor_block_alloc(tens,'c8',ierr,.FALSE.); if(ierr.ne.0) then; ierr=19; return; endif
	     else
	      if(ierr.ne.0) then; ierr=20; return; endif
	      nullify(tens%data_cmplx8)
	     endif
	    else
	     ierr=21; return
	    endif
	   end select
	  else
!Set the tensor block norm based on the master kind:
	   val_r8=tensor_block_norm2(tens,ierr,mast_kind); if(ierr.ne.0) then; ierr=22; return; endif
	   tens%scalar_value=cmplx(dsqrt(val_r8),0d0,kind=8) !Euclidean (Frobenius) norm of the tensor block
!Proceed:
	   if(slk.ne.mast_kind) then
 !REAL4:
	    if((slk.eq.'r4'.or.slk.eq.'R4').or.&
	      &((mast_kind.ne.'r4'.and.mast_kind.ne.'R4').and.slk.eq.'  '.and.associated(tens%data_real4))) then
	     if(slk.ne.'  '.and.(.not.associated(tens%data_real4))) then
!	      allocate(tens%data_real4(0:ls-1),STAT=ierr)
	      ierr=array_alloc(tens%data_real4,ls,base=0_LONGINT)
	      if(ierr.ne.0) then; ierr=23; return; endif
	      res=tensor_block_alloc(tens,'r4',ierr,.TRUE.); if(ierr.ne.0) then; ierr=24; return; endif
	     endif
	     if(size(tens%data_real4).eq.ls) then
	      select case(mast_kind)
	      case('r8','R8')
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED)
	       do l0=0_LONGINT,ls-1_LONGINT; tens%data_real4(l0)=real(tens%data_real8(l0),4); enddo
!$OMP END PARALLEL DO
	      case('c4','C4')
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED)
	       do l0=0_LONGINT,ls-1_LONGINT; tens%data_real4(l0)=cmplx4_to_real4(tens%data_cmplx4(l0)); enddo
!$OMP END PARALLEL DO
	      case('c8','C8')
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED)
	       do l0=0_LONGINT,ls-1_LONGINT; tens%data_real4(l0)=real(cmplx8_to_real8(tens%data_cmplx8(l0)),4); enddo
!$OMP END PARALLEL DO
	      end select
	     else
	      ierr=25; return !array size mismatch
	     endif
	    endif
 !REAL8:
	    if((slk.eq.'r8'.or.slk.eq.'R8').or.&
	      &((mast_kind.ne.'r8'.and.mast_kind.ne.'R8').and.slk.eq.'  '.and.associated(tens%data_real8))) then
	     if(slk.ne.'  '.and.(.not.associated(tens%data_real8))) then
!	      allocate(tens%data_real8(0:ls-1),STAT=ierr)
	      ierr=array_alloc(tens%data_real8,ls,base=0_LONGINT)
	      if(ierr.ne.0) then; ierr=26; return; endif
	      res=tensor_block_alloc(tens,'r8',ierr,.TRUE.); if(ierr.ne.0) then; ierr=27; return; endif
	     endif
	     if(size(tens%data_real8).eq.ls) then
	      select case(mast_kind)
	      case('r4','R4')
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED)
	       do l0=0_LONGINT,ls-1_LONGINT; tens%data_real8(l0)=tens%data_real4(l0); enddo
!$OMP END PARALLEL DO
	      case('c4','C4')
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED)
	       do l0=0_LONGINT,ls-1_LONGINT; tens%data_real8(l0)=real(cmplx4_to_real4(tens%data_cmplx4(l0)),8); enddo
!$OMP END PARALLEL DO
	      case('c8','C8')
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED)
	       do l0=0_LONGINT,ls-1_LONGINT; tens%data_real8(l0)=cmplx8_to_real8(tens%data_cmplx8(l0)); enddo
!$OMP END PARALLEL DO
	      end select
	     else
	      ierr=28; return !array size mismatch
	     endif
	    endif
 !COMPLEX4:
	    if((slk.eq.'c4'.or.slk.eq.'C4').or.&
	      &((mast_kind.ne.'c4'.and.mast_kind.ne.'C4').and.slk.eq.'  '.and.associated(tens%data_cmplx4))) then
	     if(slk.ne.'  '.and.(.not.associated(tens%data_cmplx4))) then
!	      allocate(tens%data_cmplx4(0:ls-1),STAT=ierr)
	      ierr=array_alloc(tens%data_cmplx4,ls,base=0_LONGINT)
	      if(ierr.ne.0) then; ierr=29; return; endif
	      res=tensor_block_alloc(tens,'c4',ierr,.TRUE.); if(ierr.ne.0) then; ierr=30; return; endif
	     endif
	     if(size(tens%data_cmplx4).eq.ls) then
	      select case(mast_kind)
	      case('r4','R4')
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED)
	       do l0=0_LONGINT,ls-1_LONGINT; tens%data_cmplx4(l0)=cmplx(tens%data_real4(l0),0.0,kind=4); enddo
!$OMP END PARALLEL DO
	      case('r8','R8')
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED)
	       do l0=0_LONGINT,ls-1_LONGINT; tens%data_cmplx4(l0)=cmplx(tens%data_real8(l0),0d0,kind=4); enddo
!$OMP END PARALLEL DO
	      case('c8','C8')
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED)
	       do l0=0_LONGINT,ls-1_LONGINT; tens%data_cmplx4(l0)=cmplx(tens%data_cmplx8(l0),kind=4); enddo
!$OMP END PARALLEL DO
	      end select
	     else
	      ierr=31; return !array size mismatch
	     endif
	    endif
 !COMPLEX8:
	    if((slk.eq.'c8'.or.slk.eq.'C8').or.&
	      &((mast_kind.ne.'c8'.and.mast_kind.ne.'C8').and.slk.eq.'  '.and.associated(tens%data_cmplx8))) then
	     if(slk.ne.'  '.and.(.not.associated(tens%data_cmplx8))) then
!	      allocate(tens%data_cmplx8(0:ls-1),STAT=ierr)
	      ierr=array_alloc(tens%data_cmplx8,ls,base=0_LONGINT)
	      if(ierr.ne.0) then; ierr=32; return; endif
	      res=tensor_block_alloc(tens,'c8',ierr,.TRUE.); if(ierr.ne.0) then; ierr=33; return; endif
	     endif
	     if(size(tens%data_cmplx8).eq.ls) then
	      select case(mast_kind)
	      case('r4','R4')
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED)
	       do l0=0_LONGINT,ls-1_LONGINT; tens%data_cmplx8(l0)=cmplx(tens%data_real4(l0),0.0,kind=8); enddo
!$OMP END PARALLEL DO
	      case('r8','R8')
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED)
	       do l0=0_LONGINT,ls-1_LONGINT; tens%data_cmplx8(l0)=cmplx(tens%data_real8(l0),0d0,kind=8); enddo
!$OMP END PARALLEL DO
	      case('c4','C4')
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED)
	       do l0=0_LONGINT,ls-1_LONGINT; tens%data_cmplx8(l0)=cmplx(tens%data_cmplx4(l0),kind=8); enddo
!$OMP END PARALLEL DO
	      end select
	     else
	      ierr=34; return !array size mismatch
	     endif
	    endif
	   endif
	  endif
	 else
	  ierr=35 !master data kind is not allocated
	 endif
	elseif(tens%tensor_shape%num_dim.eq.0) then !scalar
	 if(slk.eq.'--') ierr=36 !the scalar data kind cannot be deleted
	endif
	return
	end subroutine tensor_block_sync
!---------------------------------------------------------
	subroutine tensor_block_scale(tens,scale_fac,ierr) !PARALLEL
!This subroutine multiplies a tensor block by <scale_fac>.
!INPUT:
! - tens - tensor block;
! - scale_fac - scaling factor;
!OUTPUT:
! - tens - scaled tensor block;
! - ierr - error code (0:success).
!NOTES:
! - All allocated data kinds will be scaled (no further sycnronization is required).
	implicit none
	type(tensor_block_t), intent(inout):: tens
	complex(8), intent(in):: scale_fac
	integer, intent(inout):: ierr
	integer i,j,k,l,m,n
	integer(LONGINT) l0,l1,ls
	real(4) fac_r4
	real(8) fac_r8
	complex(4) fac_c4
	complex(8) fac_c8

	ierr=0; ls=tens%tensor_block_size
	if(ls.gt.0_LONGINT) then
!REAL4:
	 if(associated(tens%data_real4)) then
	  if(size(tens%data_real4).eq.ls) then
	   fac_r4=real(cmplx8_to_real8(scale_fac),4)
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) FIRSTPRIVATE(fac_r4) SCHEDULE(GUIDED)
	   do l0=0_LONGINT,ls-1_LONGINT; tens%data_real4(l0)=tens%data_real4(l0)*fac_r4; enddo
!$OMP END PARALLEL DO
	  else
	   ierr=1; return
	  endif
	 endif
!REAL8:
	 if(associated(tens%data_real8)) then
	  if(size(tens%data_real8).eq.ls) then
	   fac_r8=cmplx8_to_real8(scale_fac)
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) FIRSTPRIVATE(fac_r8) SCHEDULE(GUIDED)
	   do l0=0_LONGINT,ls-1_LONGINT; tens%data_real8(l0)=tens%data_real8(l0)*fac_r8; enddo
!$OMP END PARALLEL DO
	  else
	   ierr=2; return
	  endif
	 endif
!CMPLX4:
	 if(associated(tens%data_cmplx4)) then
	  if(size(tens%data_cmplx4).eq.ls) then
	   fac_c4=cmplx(scale_fac,kind=4)
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) FIRSTPRIVATE(fac_c4) SCHEDULE(GUIDED)
	   do l0=0_LONGINT,ls-1_LONGINT; tens%data_cmplx4(l0)=tens%data_cmplx4(l0)*fac_c4; enddo
!$OMP END PARALLEL DO
	  else
	   ierr=3; return
	  endif
	 endif
!CMPLX8:
	 if(associated(tens%data_cmplx8)) then
	  if(size(tens%data_cmplx8).eq.ls) then
	   fac_c8=scale_fac
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) FIRSTPRIVATE(fac_c8) SCHEDULE(GUIDED)
	   do l0=0_LONGINT,ls-1_LONGINT; tens%data_cmplx8(l0)=tens%data_cmplx8(l0)*fac_c8; enddo
!$OMP END PARALLEL DO
	  else
	   ierr=4; return
	  endif
	 endif
	endif
	return
	end subroutine tensor_block_scale
!-----------------------------------------------
	subroutine tensor_block_conjg(tens,ierr) !PARALLEL
!This subroutine complex conjugates a tensor block.
!INPUT:
! - tens - tensor block;
!OUTPUT:
! - tens - complex conjugated tensor block;
! - ierr - error code (0:success).
!NOTES:
! - All allocated complex data kinds will be complex conjugated (no further sycnronization is required).
	implicit none
	type(tensor_block_t), intent(inout):: tens
	integer, intent(inout):: ierr
	integer:: i,j,k,l,m,n
	integer(LONGINT):: l0,ls

	ierr=0; ls=tens%tensor_block_size
	if(ls.gt.0_LONGINT) then
!CMPLX4:
	 if(associated(tens%data_cmplx4)) then
	  if(size(tens%data_cmplx4).eq.ls) then
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED)
	   do l0=0_LONGINT,ls-1_LONGINT; tens%data_cmplx4(l0)=conjg(tens%data_cmplx4(l0)); enddo
!$OMP END PARALLEL DO
	  else
	   ierr=1; return
	  endif
	 endif
!CMPLX8:
	 if(associated(tens%data_cmplx8)) then
	  if(size(tens%data_cmplx8).eq.ls) then
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED)
	   do l0=0_LONGINT,ls-1_LONGINT; tens%data_cmplx8(l0)=conjg(tens%data_cmplx8(l0)); enddo
!$OMP END PARALLEL DO
	  else
	   ierr=2; return
	  endif
	 endif
	endif
	return
	end subroutine tensor_block_conjg
!---------------------------------------------------------------
	real(8) function tensor_block_norm1(tens,ierr,data_kind) !PARALLEL
!This function computes the 1-norm of a tensor block.
!INPUT:
! - tens - tensor block;
! - data_kind - (optional) data kind, one of {'r4','r8','c4','c8'};
!               If <data_kind> is not specified, the maximal one will be used (r4->r8->c4->c8).
!OUTPUT:
! - tensor_block_norm1 - 1-norm of the tensor block;
! - ierr - error code (0:success).
	implicit none
	type(tensor_block_t), intent(in):: tens
	character(2), intent(in), optional:: data_kind
	integer, intent(inout):: ierr
	integer i,j,k,l,m,n
	integer(LONGINT) l0,ls
	character(2) datk
	real(4) val_r4
	real(8) val_r8

	ierr=0; tensor_block_norm1=0d0
	if(present(data_kind)) then
	 datk=data_kind
	else
	 datk=tensor_master_data_kind(tens,ierr); if(ierr.ne.0) then; ierr=1; return; endif
	endif
	if(tens%tensor_shape%num_dim.gt.0) then !true tensor
	 ls=tens%tensor_block_size
	 if(ls.gt.0_LONGINT) then
	  select case(datk)
	  case('r4','R4')
	   if(associated(tens%data_real4)) then
	    if(size(tens%data_real4).eq.ls) then
	     val_r4=0.0
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED) REDUCTION(+:val_r4)
	     do l0=0_LONGINT,ls-1_LONGINT; val_r4=val_r4+abs(tens%data_real4(l0)); enddo
!$OMP END PARALLEL DO
	     tensor_block_norm1=real(val_r4,8)
	    else
	     ierr=2
	    endif
	   else
	    ierr=3
	   endif
	  case('r8','R8')
	   if(associated(tens%data_real8)) then
	    if(size(tens%data_real8).eq.ls) then
	     val_r8=0d0
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED) REDUCTION(+:val_r8)
	     do l0=0_LONGINT,ls-1_LONGINT; val_r8=val_r8+abs(tens%data_real8(l0)); enddo
!$OMP END PARALLEL DO
	     tensor_block_norm1=val_r8
	    else
	     ierr=4
	    endif
	   else
	    ierr=5
	   endif
	  case('c4','C4')
	   if(associated(tens%data_cmplx4)) then
	    if(size(tens%data_cmplx4).eq.ls) then
	     val_r4=0.0
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED) REDUCTION(+:val_r4)
	     do l0=0_LONGINT,ls-1_LONGINT; val_r4=val_r4+abs(tens%data_cmplx4(l0)); enddo
!$OMP END PARALLEL DO
	     tensor_block_norm1=val_r4
	    else
	     ierr=6
	    endif
	   else
	    ierr=7
	   endif
	  case('c8','C8')
	   if(associated(tens%data_cmplx8)) then
	    if(size(tens%data_cmplx8).eq.ls) then
	     val_r8=0d0
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED) REDUCTION(+:val_r8)
	     do l0=0_LONGINT,ls-1_LONGINT; val_r8=val_r8+abs(tens%data_cmplx8(l0)); enddo
!$OMP END PARALLEL DO
	     tensor_block_norm1=val_r8
	    else
	     ierr=8
	    endif
	   else
	    ierr=9
	   endif
	  case default
	   ierr=10
	  end select
	 else
	  ierr=11
	 endif
	elseif(tens%tensor_shape%num_dim.eq.0) then !scalar tensor
	 tensor_block_norm1=abs(tens%scalar_value)
	else !empty tensor
	 ierr=12
	endif
	return
	end function tensor_block_norm1
!---------------------------------------------------------------
	real(8) function tensor_block_norm2(tens,ierr,data_kind) !PARALLEL
!This function computes the squared Euclidean (Frobenius) 2-norm of a tensor block.
!INPUT:
! - tens - tensor block;
! - data_kind - (optional) data kind, one of {'r4','r8','c4','c8'};
!               If <data_kind> is not specified, the maximal one will be used (r4->r8->c4->c8).
!OUTPUT:
! - tensor_block_norm2 - squared 2-norm of the tensor block;
! - ierr - error code (0:success).
	implicit none
	type(tensor_block_t), intent(in):: tens
	character(2), intent(in), optional:: data_kind
	integer, intent(inout):: ierr
	integer i,j,k,l,m,n
	integer(LONGINT) l0,ls
	character(2) datk
	real(4) val_r4
	real(8) val_r8

	ierr=0; tensor_block_norm2=0d0
	if(present(data_kind)) then
	 datk=data_kind
	else
	 datk=tensor_master_data_kind(tens,ierr); if(ierr.ne.0) then; ierr=1; return; endif
	endif
	if(tens%tensor_shape%num_dim.gt.0) then !true tensor
	 ls=tens%tensor_block_size
	 if(ls.gt.0_LONGINT) then
	  select case(datk)
	  case('r4','R4')
	   if(associated(tens%data_real4)) then
	    if(size(tens%data_real4).eq.ls) then
	     val_r4=0.0
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED) REDUCTION(+:val_r4)
	     do l0=0_LONGINT,ls-1_LONGINT; val_r4=val_r4+tens%data_real4(l0)**2; enddo
!$OMP END PARALLEL DO
	     tensor_block_norm2=real(val_r4,8)
	    else
	     ierr=2
	    endif
	   else
	    ierr=3
	   endif
	  case('r8','R8')
	   if(associated(tens%data_real8)) then
	    if(size(tens%data_real8).eq.ls) then
	     val_r8=0d0
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED) REDUCTION(+:val_r8)
	     do l0=0_LONGINT,ls-1_LONGINT; val_r8=val_r8+tens%data_real8(l0)**2; enddo
!$OMP END PARALLEL DO
	     tensor_block_norm2=val_r8
	    else
	     ierr=4
	    endif
	   else
	    ierr=5
	   endif
	  case('c4','C4')
	   if(associated(tens%data_cmplx4)) then
	    if(size(tens%data_cmplx4).eq.ls) then
	     val_r4=0d0
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED) REDUCTION(+:val_r4)
	     do l0=0_LONGINT,ls-1_LONGINT; val_r4=val_r4+abs(tens%data_cmplx4(l0))**2; enddo
!$OMP END PARALLEL DO
	     tensor_block_norm2=val_r4
	    else
	     ierr=6
	    endif
	   else
	    ierr=7
	   endif
	  case('c8','C8')
	   if(associated(tens%data_cmplx8)) then
	    if(size(tens%data_cmplx8).eq.ls) then
	     val_r8=0d0
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED) REDUCTION(+:val_r8)
	     do l0=0_LONGINT,ls-1_LONGINT; val_r8=val_r8+abs(tens%data_cmplx8(l0))**2; enddo
!$OMP END PARALLEL DO
	     tensor_block_norm2=val_r8
	    else
	     ierr=8
	    endif
	   else
	    ierr=9
	   endif
	  case default
	   ierr=10
	  end select
	 else
	  ierr=11
	 endif
	elseif(tens%tensor_shape%num_dim.eq.0) then !scalar tensor
	 tensor_block_norm2=abs(tens%scalar_value)**2
	else !empty tensor
	 ierr=12
	endif
	return
	end function tensor_block_norm2
!-------------------------------------------------------------
	real(8) function tensor_block_max(tens,ierr,data_kind) !PARALLEL
!This function finds the largest by modulus element in a tensor block.
!INPUT:
! - tens - tensor block;
! - data_kind - (optional) requested data kind;
!OUTPUT:
! - tensor_block_max - modulus of the max element(s);
! - ierr - error code (0:success).
	implicit none
	type(tensor_block_t), intent(inout):: tens !(out) because of <tensor_block_sync>
	character(2), intent(in), optional:: data_kind
	integer, intent(inout):: ierr
	character(2) dtk
	integer(LONGINT) l0
	real(4) valr4
	real(8) valr8

	ierr=0
	if(tens%tensor_shape%num_dim.gt.0) then !true tensor
	 if(present(data_kind)) then
	  dtk=data_kind
	 else
	  dtk=tensor_master_data_kind(tens,ierr); if(ierr.ne.0) then; ierr=1; return; endif
	 endif
	 select case(dtk)
	 case('r4','R4')
	  if(associated(tens%data_real4)) then
	   if(size(tens%data_real4).eq.tens%tensor_block_size) then
	    valr4=0.0
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED) REDUCTION(max:valr4)
	    do l0=0_LONGINT,tens%tensor_block_size-1_LONGINT; valr4=max(valr4,abs(tens%data_real4(l0))); enddo
!$OMP END PARALLEL DO
	    tensor_block_max=valr4
	   else
	    ierr=3
	   endif
	  else
	   ierr=4
	  endif
	 case('r8','R8')
	  if(associated(tens%data_real8)) then
	   if(size(tens%data_real8).eq.tens%tensor_block_size) then
	    valr8=0d0
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED) REDUCTION(max:valr8)
	    do l0=0_LONGINT,tens%tensor_block_size-1_LONGINT; valr8=max(valr8,abs(tens%data_real8(l0))); enddo
!$OMP END PARALLEL DO
	    tensor_block_max=valr8
	   else
	    ierr=5
	   endif
	  else
	   ierr=6
	  endif
	 case('c4','C4')
	  if(associated(tens%data_cmplx4)) then
	   if(size(tens%data_cmplx4).eq.tens%tensor_block_size) then
	    valr4=0.0
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED) REDUCTION(max:valr4)
	    do l0=0_LONGINT,tens%tensor_block_size-1_LONGINT; valr4=max(valr4,abs(tens%data_cmplx4(l0))); enddo
!$OMP END PARALLEL DO
	    tensor_block_max=valr4
	   else
	    ierr=7
	   endif
	  else
	   ierr=8
	  endif
	 case('c8','C8')
	  if(associated(tens%data_cmplx8)) then
	   if(size(tens%data_cmplx8).eq.tens%tensor_block_size) then
	    valr8=0d0
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED) REDUCTION(max:valr8)
	    do l0=0_LONGINT,tens%tensor_block_size-1_LONGINT; valr8=max(valr8,abs(tens%data_cmplx8(l0))); enddo
!$OMP END PARALLEL DO
	    tensor_block_max=valr8
	   else
	    ierr=9
	   endif
	  else
	   ierr=10
	  endif
	 case default
	  ierr=11; return
	 end select
	elseif(tens%tensor_shape%num_dim.eq.0) then !scalar
	 tensor_block_max=abs(tens%scalar_value)
	else
	 ierr=12
	endif
	return
	end function tensor_block_max
!-------------------------------------------------------------
	real(8) function tensor_block_min(tens,ierr,data_kind) !PARALLEL
!This function finds the smallest by modulus element in a tensor block.
!INPUT:
! - tens - tensor block;
! - data_kind - (optional) requested data kind;
!OUTPUT:
! - tensor_block_min - modulus of the min element(s);
! - ierr - error code (0:success).
	implicit none
	type(tensor_block_t), intent(inout):: tens !(out) because of <tensor_block_sync>
	character(2), intent(in), optional:: data_kind
	integer, intent(inout):: ierr
	character(2) dtk
	integer(LONGINT) l0
	real(4) valr4
	real(8) valr8

	ierr=0
	if(tens%tensor_shape%num_dim.gt.0) then !true tensor
	 if(present(data_kind)) then
	  dtk=data_kind
	 else
	  dtk=tensor_master_data_kind(tens,ierr); if(ierr.ne.0) then; ierr=1; return; endif
	 endif
	 select case(dtk)
	 case('r4','R4')
	  if(associated(tens%data_real4)) then
	   if(size(tens%data_real4).eq.tens%tensor_block_size) then
	    valr4=abs(tens%data_real4(0))
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED) REDUCTION(min:valr4)
	    do l0=0_LONGINT,tens%tensor_block_size-1_LONGINT; valr4=min(valr4,abs(tens%data_real4(l0))); enddo
!$OMP END PARALLEL DO
	    tensor_block_min=valr4
	   else
	    ierr=3
	   endif
	  else
	   ierr=4
	  endif
	 case('r8','R8')
	  if(associated(tens%data_real8)) then
	   if(size(tens%data_real8).eq.tens%tensor_block_size) then
	    valr8=abs(tens%data_real8(0))
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED) REDUCTION(min:valr8)
	    do l0=0_LONGINT,tens%tensor_block_size-1_LONGINT; valr8=min(valr8,abs(tens%data_real8(l0))); enddo
!$OMP END PARALLEL DO
	    tensor_block_min=valr8
	   else
	    ierr=5
	   endif
	  else
	   ierr=6
	  endif
	 case('c4','C4')
	  if(associated(tens%data_cmplx4)) then
	   if(size(tens%data_cmplx4).eq.tens%tensor_block_size) then
	    valr4=abs(tens%data_cmplx4(0))
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED) REDUCTION(min:valr4)
	    do l0=0_LONGINT,tens%tensor_block_size-1_LONGINT; valr4=min(valr4,abs(tens%data_cmplx4(l0))); enddo
!$OMP END PARALLEL DO
	    tensor_block_min=valr4
	   else
	    ierr=7
	   endif
	  else
	   ierr=8
	  endif
	 case('c8','C8')
	  if(associated(tens%data_cmplx8)) then
	   if(size(tens%data_cmplx8).eq.tens%tensor_block_size) then
	    valr8=abs(tens%data_cmplx8(0))
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED) REDUCTION(min:valr8)
	    do l0=0_LONGINT,tens%tensor_block_size-1_LONGINT; valr8=min(valr8,abs(tens%data_cmplx8(l0))); enddo
!$OMP END PARALLEL DO
	    tensor_block_min=valr8
	   else
	    ierr=9
	   endif
	  else
	   ierr=10
	  endif
	 case default
	  ierr=11; return
	 end select
	elseif(tens%tensor_shape%num_dim.eq.0) then !scalar
	 tensor_block_min=abs(tens%scalar_value)
	else
	 ierr=12
	endif
	return
	end function tensor_block_min
!-----------------------------------------------------------------------
	subroutine tensor_block_slice(tens,slice,ext_beg,ierr,data_kind) !PARALLEL
!This subroutine extracts a slice from a tensor block.
!Tensor block <slice> must have its shape defined on input!
!INPUT:
! - tens - tensor block;
! - slice - tensor block which will contain the slice (its shape specifies the slice dimensions);
! - ext_beg(1:) - beginning offset of each tensor dimension (numeration starts at 0) to slice from;
! - data_kind - (optional) requested data_kind, one of {'r4','r8','c4','c8'};
!OUTPUT:
! - slice - filled tensor block slice;
! - ierr - error code (0:success).
!NOTES:
! - <slice> must have its shape defined and the same layout as <tens>!
!   <slice> data does not have to be allocated and initialized.
! - For scalar tensors, slicing reduces to copying the scalar value.
! - If no <data_kind> is specified, the highest possible data kind will be used from <tens>.
! - <slice> is syncronized at the end.
	implicit none
	type(tensor_block_t), intent(inout):: tens !(out) because of <tensor_block_copy> because of <tensor_block_layout> because of <tensor_block_shape_ok>
	type(tensor_block_t), intent(inout):: slice
	integer, intent(in):: ext_beg(1:*)
	character(2), intent(in), optional:: data_kind
	integer, intent(inout):: ierr
	integer i,j,k,l,m,n,ks,kf,tlt,slt
	integer(LONGINT) ls
	character(2) dtk
	logical res

	ierr=0
	if(tens%tensor_shape%num_dim.eq.slice%tensor_shape%num_dim) then
	 n=tens%tensor_shape%num_dim
	 if(n.gt.0) then !true tensor
!Check and possibly adjust arguments:
	  ls=tensor_block_shape_size(slice,ierr); if(ierr.ne.0) then; ierr=1; return; endif
	  if(slice%tensor_block_size.le.0_LONGINT.or.slice%tensor_block_size.ne.ls) then; ierr=2; return; endif !invalid size of the tensor slice
	  if(present(data_kind)) then
	   dtk=data_kind
	  else
	   dtk=tensor_master_data_kind(tens,ierr); if(ierr.ne.0) then; ierr=3; return; endif
	  endif
	  select case(dtk)
	  case('r4','R4')
	   if(.not.associated(tens%data_real4)) then; ierr=4; return; endif
	   if(.not.associated(slice%data_real4)) then
!	    allocate(slice%data_real4(0:ls-1),STAT=ierr)
	    ierr=array_alloc(slice%data_real4,ls,base=0_LONGINT)
	    if(ierr.ne.0) then; ierr=5; return; endif
	    res=tensor_block_alloc(slice,'r4',ierr,.TRUE.); if(ierr.ne.0) then; ierr=6; return; endif
	   endif
	  case('r8','R8')
	   if(.not.associated(tens%data_real8)) then; ierr=7; return; endif
	   if(.not.associated(slice%data_real8)) then
!	    allocate(slice%data_real8(0:ls-1),STAT=ierr)
	    ierr=array_alloc(slice%data_real8,ls,base=0_LONGINT)
	    if(ierr.ne.0) then; ierr=8; return; endif
	    res=tensor_block_alloc(slice,'r8',ierr,.TRUE.); if(ierr.ne.0) then; ierr=9; return; endif
	   endif
	  case('c4','C4')
	   if(.not.associated(tens%data_cmplx4)) then; ierr=10; return; endif
	   if(.not.associated(slice%data_cmplx4)) then
!	    allocate(slice%data_cmplx4(0:ls-1),STAT=ierr)
	    ierr=array_alloc(slice%data_cmplx4,ls,base=0_LONGINT)
	    if(ierr.ne.0) then; ierr=11; return; endif
	    res=tensor_block_alloc(slice,'c4',ierr,.TRUE.); if(ierr.ne.0) then; ierr=12; return; endif
	   endif
	  case('c8','C8')
	   if(.not.associated(tens%data_cmplx8)) then; ierr=13; return; endif
	   if(.not.associated(slice%data_cmplx8)) then
!	    allocate(slice%data_cmplx8(0:ls-1),STAT=ierr)
	    ierr=array_alloc(slice%data_cmplx8,ls,base=0_LONGINT)
	    if(ierr.ne.0) then; ierr=14; return; endif
	    res=tensor_block_alloc(slice,'c8',ierr,.TRUE.); if(ierr.ne.0) then; ierr=15; return; endif
	   endif
	  case default
	   ierr=16; return !invalid data kind
	  end select
!Check whether the slice is trivial:
	  kf=0
	  do i=1,n
	   if(ext_beg(i).lt.0.or.ext_beg(i).ge.tens%tensor_shape%dim_extent(i).or.slice%tensor_shape%dim_extent(i).le.0.or.&
	     &ext_beg(i)+slice%tensor_shape%dim_extent(i)-1.ge.tens%tensor_shape%dim_extent(i)) then
	    ierr=17; return
	   endif
	   if(slice%tensor_shape%dim_extent(i).ne.tens%tensor_shape%dim_extent(i)) kf=1 !non-trivial
	  enddo
!Slicing:
	  if(kf.eq.0) then !one-to-one copy
	   call tensor_block_copy(tens,slice,ierr); if(ierr.ne.0) then; ierr=18; return; endif
	  else !true slicing
	   tlt=tensor_block_layout(tens,ierr); if(ierr.ne.0) then; ierr=19; return; endif
	   slt=tensor_block_layout(slice,ierr,.TRUE.); if(ierr.ne.0) then; ierr=20; return; endif
	   if(slt.eq.tlt) then !layouts coinside
	    select case(tlt)
	    case(dimension_led)
	     select case(dtk)
	     case('r4','R4')
	      call tensor_block_slice_dlf(n,tens%data_real4,tens%tensor_shape%dim_extent,&
	            &slice%data_real4,slice%tensor_shape%dim_extent,ext_beg,ierr); if(ierr.ne.0) then; ierr=21; return; endif
	     case('r8','R8')
	      call tensor_block_slice_dlf(n,tens%data_real8,tens%tensor_shape%dim_extent,&
	            &slice%data_real8,slice%tensor_shape%dim_extent,ext_beg,ierr); if(ierr.ne.0) then; ierr=22; return; endif
	     case('c4','C4')
	      call tensor_block_slice_dlf(n,tens%data_cmplx4,tens%tensor_shape%dim_extent,&
	            &slice%data_cmplx4,slice%tensor_shape%dim_extent,ext_beg,ierr); if(ierr.ne.0) then; ierr=23; return; endif
	     case('c8','C8')
	      call tensor_block_slice_dlf(n,tens%data_cmplx8,tens%tensor_shape%dim_extent,&
	            &slice%data_cmplx8,slice%tensor_shape%dim_extent,ext_beg,ierr); if(ierr.ne.0) then; ierr=24; return; endif
	     end select
	    case(bricked_dense,bricked_ordered)
	     !`Future
	    case(sparse_list)
	     !`Future
	    case(compressed)
	     !`Future
	    case default
	     ierr=25; return !invalid tensor layout
	    end select
	    if(DATA_KIND_SYNC) then
	     call tensor_block_sync(slice,dtk,ierr); if(ierr.ne.0) then; ierr=26; return; endif
	    endif
	   else
	    ierr=27 !tensor layouts differ
	   endif
	  endif
	 elseif(n.eq.0) then !scalar
	  slice%scalar_value=tens%scalar_value
	 else !empty tensors
	  ierr=28
	 endif
	else
	 ierr=29 !tensor block and its slice have different ranks
	endif
	return
	end subroutine tensor_block_slice
!------------------------------------------------------------------------
	subroutine tensor_block_insert(tens,slice,ext_beg,ierr,data_kind) !PARALLEL
!This subroutine inserts a slice into a tensor block.
!INPUT:
! - tens - tensor block;
! - slice - slice to be inserted;
! - ext_beg(1:) - beginning offset of each tensor dimension (numeration starts at 0) where to insert;
! - data_kind - (optional) requested data_kind, one of {'r4','r8','c4','c8'};
!OUTPUT:
! - tens - modified tensor block;
! - ierr - error code (0:success).
!NOTES:
! - <slice> must have the same layout as <tens>!
! - For scalar tensors, insertion reduces to copying the scalar value.
! - If no <data_kind> is specified, the highest possible common data kind will be used.
! - <tens> is syncronized at the end.
	implicit none
	type(tensor_block_t), intent(inout):: tens
	type(tensor_block_t), intent(inout):: slice !(out) because of <tensor_block_copy> because of <tensor_block_layout> because of <tensor_block_shape_ok>
	integer, intent(in):: ext_beg(1:*)
	character(2), intent(in), optional:: data_kind
	integer, intent(inout):: ierr
	integer i,j,k,l,m,n,ks,kf,tlt,slt
	integer(LONGINT) ls
	character(2) dtk,stk

	ierr=0
	if(tens%tensor_shape%num_dim.eq.slice%tensor_shape%num_dim) then
	 n=tens%tensor_shape%num_dim
	 if(n.gt.0) then !true tensor
!Check and possibly adjust arguments:
	  ls=tensor_block_shape_size(slice,ierr); if(ierr.ne.0) then; ierr=1; return; endif
	  if(slice%tensor_block_size.le.0_LONGINT.or.slice%tensor_block_size.ne.ls) then; ierr=2; return; endif !invalid size of the tensor slice
	  stk=tensor_master_data_kind(slice,ierr); if(ierr.ne.0) then; ierr=3; return; endif
	  if(stk.eq.'--') then; ierr=4; return; endif !empty slice
	  if(present(data_kind)) then
	   dtk=data_kind
	  else
	   dtk=tensor_common_data_kind(tens,slice,ierr); if(ierr.ne.0) then; ierr=5; return; endif
	  endif
	  select case(dtk)
	  case('r4','R4')
	   if(.not.associated(tens%data_real4)) then; ierr=6; return; endif
	   if(.not.associated(slice%data_real4)) then
	    call tensor_block_sync(slice,stk,ierr,'r4'); if(ierr.ne.0) then; ierr=7; return; endif
	   endif
	  case('r8','R8')
	   if(.not.associated(tens%data_real8)) then; ierr=8; return; endif
	   if(.not.associated(slice%data_real8)) then
	    call tensor_block_sync(slice,stk,ierr,'r8'); if(ierr.ne.0) then; ierr=9; return; endif
	   endif
	  case('c4','C4')
	   if(.not.associated(tens%data_cmplx4)) then; ierr=10; return; endif
	   if(.not.associated(slice%data_cmplx4)) then
	    call tensor_block_sync(slice,stk,ierr,'c4'); if(ierr.ne.0) then; ierr=11; return; endif
	   endif
	  case('c8','C8')
	   if(.not.associated(tens%data_cmplx8)) then; ierr=12; return; endif
	   if(.not.associated(slice%data_cmplx8)) then
	    call tensor_block_sync(slice,stk,ierr,'c8'); if(ierr.ne.0) then; ierr=13; return; endif
	   endif
	  case('--')
	   dtk=tensor_master_data_kind(tens,ierr); if(ierr.ne.0) then; ierr=14; return; endif
	   select case(dtk)
	   case('r4','R4')
	    if(.not.associated(slice%data_real4)) then
	     call tensor_block_sync(slice,stk,ierr,'r4'); if(ierr.ne.0) then; ierr=15; return; endif
	    endif
	   case('r8','R8')
	    if(.not.associated(slice%data_real8)) then
	     call tensor_block_sync(slice,stk,ierr,'r8'); if(ierr.ne.0) then; ierr=16; return; endif
	    endif
	   case('c4','C4')
	    if(.not.associated(slice%data_cmplx4)) then
	     call tensor_block_sync(slice,stk,ierr,'c4'); if(ierr.ne.0) then; ierr=17; return; endif
	    endif
	   case('c8','C8')
	    if(.not.associated(slice%data_cmplx8)) then
	     call tensor_block_sync(slice,stk,ierr,'c8'); if(ierr.ne.0) then; ierr=18; return; endif
	    endif
	   case default
	    ierr=19; return !no master data kind found in <tens>
	   end select
	  case default
	   ierr=20; return !invalid data kind
	  end select
!Check whether the slice is trivial:
	  kf=0
	  do i=1,n
	   if(ext_beg(i).lt.0.or.ext_beg(i).ge.tens%tensor_shape%dim_extent(i).or.slice%tensor_shape%dim_extent(i).le.0.or.&
	     &ext_beg(i)+slice%tensor_shape%dim_extent(i)-1.ge.tens%tensor_shape%dim_extent(i)) then
	    ierr=21; return
	   endif
	   if(slice%tensor_shape%dim_extent(i).ne.tens%tensor_shape%dim_extent(i)) kf=1 !non-trivial
	  enddo
!Insertion:
	  if(kf.eq.0) then !one-to-one copy
	   call tensor_block_copy(slice,tens,ierr); if(ierr.ne.0) then; ierr=22; return; endif
	  else !true insertion
	   tlt=tensor_block_layout(tens,ierr); if(ierr.ne.0) then; ierr=23; return; endif
	   slt=tensor_block_layout(slice,ierr,.TRUE.); if(ierr.ne.0) then; ierr=24; return; endif
	   if(slt.eq.tlt) then
	    select case(tlt)
	    case(dimension_led)
	     select case(dtk)
	     case('r4','R4')
	      call tensor_block_insert_dlf(n,tens%data_real4,tens%tensor_shape%dim_extent,&
	            &slice%data_real4,slice%tensor_shape%dim_extent,ext_beg,ierr); if(ierr.ne.0) then; ierr=25; return; endif
	     case('r8','R8')
	      call tensor_block_insert_dlf(n,tens%data_real8,tens%tensor_shape%dim_extent,&
	            &slice%data_real8,slice%tensor_shape%dim_extent,ext_beg,ierr); if(ierr.ne.0) then; ierr=26; return; endif
	     case('c4','C4')
	      call tensor_block_insert_dlf(n,tens%data_cmplx4,tens%tensor_shape%dim_extent,&
	            &slice%data_cmplx4,slice%tensor_shape%dim_extent,ext_beg,ierr); if(ierr.ne.0) then; ierr=27; return; endif
	     case('c8','C8')
	      call tensor_block_insert_dlf(n,tens%data_cmplx8,tens%tensor_shape%dim_extent,&
	            &slice%data_cmplx8,slice%tensor_shape%dim_extent,ext_beg,ierr); if(ierr.ne.0) then; ierr=28; return; endif
	     end select
	    case(bricked_dense,bricked_ordered)
	     !`Future
	    case(sparse_list)
	     !`Future
	    case(compressed)
	     !`Future
	    case default
	     ierr=29; return !invalid tensor layout
	    end select
	    if(DATA_KIND_SYNC) then
	     call tensor_block_sync(tens,dtk,ierr); if(ierr.ne.0) then; ierr=30; return; endif
	    endif
	   else
	    ierr=31 !tensor layouts differ
	   endif
	  endif
	 elseif(n.eq.0) then !scalar
	  tens%scalar_value=slice%scalar_value
	 else !empty tensors
	  ierr=32
	 endif
	else
	 ierr=33 !tensor block and the slice have different ranks
	endif
	return
	end subroutine tensor_block_insert
!--------------------------------------------------------------------------------------------
	subroutine tensor_block_print(ifh,head_line,ext_beg,tens,ierr,data_kind,print_thresh) !SERIAL
!This subroutine prints all non-zero elements of a tensor block.
!INPUT:
! - ifh - output file handle;
! - head_line - header line;
! - ext_beg(1:) - beginnings for each dimension;
! - tens - tensor block;
! - data_kind - (optional) requested data kind;
! - print_thresh - printing threshold;
!OUTPUT:
! - Output in the file <ifh>;
! - ierr - error code (0:success).
	implicit none
	integer, intent(in):: ifh,ext_beg(1:*)
	character(*), intent(in):: head_line
	type(tensor_block_t), intent(inout):: tens !(out) because of <tensor_block_layout>
	character(2), intent(in), optional:: data_kind
	real(8), intent(in), optional:: print_thresh
	integer, intent(inout):: ierr
	integer i,j,k,l,m,n,tst,im(1:max_tensor_rank)
	integer(LONGINT) l0,l1,bases(1:max_tensor_rank)
	character(2) dtk
	real(8) prth

	ierr=0
	write(ifh,'("#")',advance='no',err=2000)
	call printl(ifh,head_line(1:len_trim(head_line)))
	if(tens%tensor_shape%num_dim.ge.0.and.tens%tensor_shape%num_dim.le.max_tensor_rank) then
	 if(tens%tensor_shape%num_dim.gt.0) then !true tensor
	  if(present(data_kind)) then
	   dtk=data_kind
	  else
	   dtk=tensor_master_data_kind(tens,ierr); if(ierr.ne.0) then; ierr=1; return; endif
	  endif
	  if(present(print_thresh)) then; prth=print_thresh; else; prth=ABS_CMP_THRESH; endif
	  tst=tensor_block_layout(tens,ierr); if(ierr.ne.0) then; ierr=2; return; endif
	  select case(tst)
	  case(dimension_led)
	   l0=1_LONGINT; do i=1,tens%tensor_shape%num_dim; bases(i)=l0; l0=l0*tens%tensor_shape%dim_extent(i); enddo
	   if(l0.eq.tens%tensor_block_size) then
	    select case(dtk)
	    case('r4','R4')
	     do l0=0_LONGINT,tens%tensor_block_size-1_LONGINT
	      if(abs(tens%data_real4(l0)).gt.real(prth,4)) then
	       im(1:tens%tensor_shape%num_dim)=ext_beg(1:tens%tensor_shape%num_dim)
	       l1=l0; do i=tens%tensor_shape%num_dim,1,-1; im(i)=im(i)+l1/bases(i); l1=mod(l1,bases(i)); enddo
	       write(ifh,'(D15.7,64(1x,i5))') tens%data_real4(l0),im(1:tens%tensor_shape%num_dim)
	      endif
	     enddo
	    case('r8','R8')
	     do l0=0_LONGINT,tens%tensor_block_size-1_LONGINT
	      if(abs(tens%data_real8(l0)).gt.prth) then
	       im(1:tens%tensor_shape%num_dim)=ext_beg(1:tens%tensor_shape%num_dim)
	       l1=l0; do i=tens%tensor_shape%num_dim,1,-1; im(i)=im(i)+l1/bases(i); l1=mod(l1,bases(i)); enddo
	       write(ifh,'(D23.15,64(1x,i5))') tens%data_real8(l0),im(1:tens%tensor_shape%num_dim)
	      endif
	     enddo
	    case('c4','C4')
	     do l0=0_LONGINT,tens%tensor_block_size-1_LONGINT
	      if(abs(tens%data_cmplx4(l0)).gt.prth) then
	       im(1:tens%tensor_shape%num_dim)=ext_beg(1:tens%tensor_shape%num_dim)
	       l1=l0; do i=tens%tensor_shape%num_dim,1,-1; im(i)=im(i)+l1/bases(i); l1=mod(l1,bases(i)); enddo
	       write(ifh,'("(",D23.15,",",D23.15,")",64(1x,i5))') tens%data_cmplx4(l0),im(1:tens%tensor_shape%num_dim)
	      endif
	     enddo
	    case('c8','C8')
	     do l0=0_LONGINT,tens%tensor_block_size-1_LONGINT
	      if(abs(tens%data_cmplx8(l0)).gt.prth) then
	       im(1:tens%tensor_shape%num_dim)=ext_beg(1:tens%tensor_shape%num_dim)
	       l1=l0; do i=tens%tensor_shape%num_dim,1,-1; im(i)=im(i)+l1/bases(i); l1=mod(l1,bases(i)); enddo
	       write(ifh,'("(",D23.15,",",D23.15,")",64(1x,i5))') tens%data_cmplx8(l0),im(1:tens%tensor_shape%num_dim)
	      endif
	     enddo
	    case default
	     ierr=3; return
	    end select
	   else
	    ierr=4; return
	   endif
	  case(bricked_dense,bricked_ordered)
	   !`Future
	  case(sparse_list)
	   !`Future
	  case(compressed)
	   !`Future
	  case default
	   ierr=5; return
	  end select
	 else !scalar
	  write(ifh,'("Complex scalar value = (",D23.15,",",D23.15,")")') tens%scalar_value
	 endif
	else
	 ierr=6; call printl(ifh,'ERROR(tensor_algebra::tensor_block_print): negative or too high tensor rank!')
	endif
	return
2000	ierr=7; return
	end subroutine tensor_block_print
!-----------------------------------------------------------------------------------------
	subroutine tensor_block_trace(contr_ptrn,tens_in,tens_out,ierr,data_kind,ord_rest) !PARALLEL
!This subroutine executes an intra-tensor index contraction (accumulative partial or full trace):
!tens_out(:)+=TRACE(tens_in(:))
!INPUT:
! - contr_ptrn(1:input_rank) - index contraction pattern (similar to the one used by <tensor_block_contract>);
! - tens_in - input tensor block;
! - data_kind - (optional) requested data kind;
! - ord_rest(1:input_rank) - (optional) index ordering restrictions (for contracted indices only);
!OUTPUT:
! - tens_out - initialized! output tensor block (where the result of partial/full tracing will be accumulated);
! - ierr - error code (0:success).
!NOTES:
! - Both tensor blocks must have the same storage layout.
	implicit none
	integer, intent(in):: contr_ptrn(1:*)
	integer, intent(in), optional:: ord_rest(1:*)
	character(2), intent(in), optional:: data_kind
	type(tensor_block_t), intent(inout):: tens_in !(out) because of <tensor_block_layout>
	type(tensor_block_t), intent(inout):: tens_out
	integer, intent(inout):: ierr
	integer i,j,k,l,m,n,ks,kf
	integer rank_in,rank_out,im(1:max_tensor_rank)
	integer(LONGINT) ls,l0
	character(2) dtk,slk,dlt
	logical cptrn_ok
	real(4) valr4
	real(8) valr8
	complex(4) valc4
	complex(8) valc8

	ierr=0
	rank_in=tens_in%tensor_shape%num_dim; rank_out=tens_out%tensor_shape%num_dim
	if(rank_in.gt.0.and.rank_out.ge.0.and.rank_out.le.rank_in) then
	 cptrn_ok=contr_ptrn_ok(contr_ptrn,rank_in,rank_out)
	 if(present(ord_rest)) cptrn_ok=cptrn_ok.and.ord_rest_ok(ord_rest,contr_ptrn,rank_in,rank_out)
	 if(present(data_kind)) then
	  dtk=data_kind
	 else
	  dtk=tensor_master_data_kind(tens_in,ierr); if(ierr.ne.0) then; ierr=1; return; endif
	 endif
	 if(dtk.ne.'r4'.and.dtk.ne.'R4'.and.&
	   &dtk.ne.'r8'.and.dtk.ne.'R8'.and.&
	   &dtk.ne.'c4'.and.dtk.ne.'C4'.and.&
	   &dtk.ne.'c8'.and.dtk.ne.'C8') then; ierr=2; return; endif
	 ks=tensor_block_layout(tens_in,ierr); if(ierr.ne.0) then; ierr=3; return; endif
	 kf=tensor_block_layout(tens_out,ierr); if(ierr.ne.0) then; ierr=4; return; endif
	 if(ks.eq.kf.or.kf.eq.scalar_tensor) then !the same storage layout
	  select case(ks)
	  case(dimension_led)
	   select case(dtk)
	   case('r4','R4')
	    if(associated(tens_in%data_real4)) then
	     if(rank_out.gt.0.and.(.not.associated(tens_out%data_real4))) then
	      slk=tensor_master_data_kind(tens_out,ierr); if(ierr.ne.0) then; ierr=5; return; endif
	      if(slk.eq.'--') then; ierr=6; return; endif
	      dlt='r4'; call tensor_block_sync(tens_out,slk,ierr,dlt); if(ierr.ne.0) then; ierr=7; return; endif
	     else
	      dlt='  '
	     endif
	     if(rank_out.gt.0) then !partial trace
	      call tensor_block_ptrace_dlf(contr_ptrn,ord_rest,tens_in%data_real4,rank_in,tens_in%tensor_shape%dim_extent,&
	            &tens_out%data_real4,rank_out,tens_out%tensor_shape%dim_extent,ierr)
	      if(ierr.ne.0) then; ierr=8; return; endif
	     else !full trace
	      valr4=real(cmplx8_to_real8(tens_out%scalar_value),4)
	      call tensor_block_ftrace_dlf(contr_ptrn,ord_rest,tens_in%data_real4,rank_in,&
	            &tens_in%tensor_shape%dim_extent,valr4,ierr)
	      if(ierr.ne.0) then; ierr=9; return; endif
	      tens_out%scalar_value=cmplx(real(valr4,8),0d0,kind=8)
	     endif
	    else
	     ierr=10; return
	    endif
	   case('r8','R8')
	    if(associated(tens_in%data_real8)) then
	     if(rank_out.gt.0.and.(.not.associated(tens_out%data_real8))) then
	      slk=tensor_master_data_kind(tens_out,ierr); if(ierr.ne.0) then; ierr=11; return; endif
	      if(slk.eq.'--') then; ierr=12; return; endif
	      dlt='r8'; call tensor_block_sync(tens_out,slk,ierr,dlt); if(ierr.ne.0) then; ierr=13; return; endif
	     else
	      dlt='  '
	     endif
	     if(rank_out.gt.0) then !partial trace
	      call tensor_block_ptrace_dlf(contr_ptrn,ord_rest,tens_in%data_real8,rank_in,tens_in%tensor_shape%dim_extent,&
	            &tens_out%data_real8,rank_out,tens_out%tensor_shape%dim_extent,ierr)
	      if(ierr.ne.0) then; ierr=14; return; endif
	     else !full trace
	      valr8=cmplx8_to_real8(tens_out%scalar_value)
	      call tensor_block_ftrace_dlf(contr_ptrn,ord_rest,tens_in%data_real8,rank_in,&
	            &tens_in%tensor_shape%dim_extent,valr8,ierr)
	      if(ierr.ne.0) then; ierr=15; return; endif
	      tens_out%scalar_value=cmplx(valr8,0d0,kind=8)
	     endif
	    else
	     ierr=16; return
	    endif
	   case('c4','C4')
	    if(associated(tens_in%data_cmplx4)) then
	     if(rank_out.gt.0.and.(.not.associated(tens_out%data_cmplx4))) then
	      slk=tensor_master_data_kind(tens_out,ierr); if(ierr.ne.0) then; ierr=17; return; endif
	      if(slk.eq.'--') then; ierr=18; return; endif
	      dlt='c4'; call tensor_block_sync(tens_out,slk,ierr,dlt); if(ierr.ne.0) then; ierr=19; return; endif
	     else
	      dlt='  '
	     endif
	     if(rank_out.gt.0) then !partial trace
	      call tensor_block_ptrace_dlf(contr_ptrn,ord_rest,tens_in%data_cmplx4,rank_in,tens_in%tensor_shape%dim_extent,&
	            &tens_out%data_cmplx4,rank_out,tens_out%tensor_shape%dim_extent,ierr)
	      if(ierr.ne.0) then; ierr=20; return; endif
	     else !full trace
	      valc4=cmplx(tens_out%scalar_value,kind=4)
	      call tensor_block_ftrace_dlf(contr_ptrn,ord_rest,tens_in%data_cmplx4,rank_in,&
	            &tens_in%tensor_shape%dim_extent,valc4,ierr)
	      if(ierr.ne.0) then; ierr=21; return; endif
	      tens_out%scalar_value=cmplx(valc4,kind=8)
	     endif
	    else
	     ierr=22; return
	    endif
	   case('c8','C8')
	    if(associated(tens_in%data_cmplx8)) then
	     if(rank_out.gt.0.and.(.not.associated(tens_out%data_cmplx8))) then
	      slk=tensor_master_data_kind(tens_out,ierr); if(ierr.ne.0) then; ierr=23; return; endif
	      if(slk.eq.'--') then; ierr=24; return; endif
	      dlt='c8'; call tensor_block_sync(tens_out,slk,ierr,dlt); if(ierr.ne.0) then; ierr=25; return; endif
	     else
	      dlt='  '
	     endif
	     if(rank_out.gt.0) then !partial trace
	      call tensor_block_ptrace_dlf(contr_ptrn,ord_rest,tens_in%data_cmplx8,rank_in,tens_in%tensor_shape%dim_extent,&
	            &tens_out%data_cmplx8,rank_out,tens_out%tensor_shape%dim_extent,ierr)
	      if(ierr.ne.0) then; ierr=26; return; endif
	     else !full trace
	      valc8=tens_out%scalar_value
	      call tensor_block_ftrace_dlf(contr_ptrn,ord_rest,tens_in%data_cmplx8,rank_in,&
	            &tens_in%tensor_shape%dim_extent,valc8,ierr)
	      if(ierr.ne.0) then; ierr=27; return; endif
	      tens_out%scalar_value=valc8
	     endif
	    else
	     ierr=28; return
	    endif
	   end select
	  case(bricked_dense,bricked_ordered)
	   !`Future
	  case(sparse_list)
	   !`Future
	  case(compressed)
	   !`Future
	  case default
	   ierr=29; return
	  end select
	  if(DATA_KIND_SYNC) then
	   call tensor_block_sync(tens_out,dtk,ierr); if(ierr.ne.0) then; ierr=30; return; endif
	  endif
	  if(dlt.ne.'  ') then
	   call tensor_block_sync(tens_out,dlt,ierr,'--'); if(ierr.ne.0) then; ierr=31; return; endif
	  endif
	 else
	  ierr=32 !tensor storage layouts differ
	 endif
	elseif(rank_in.eq.0.and.rank_out.eq.0) then !two scalars
	 tens_out%scalar_value=tens_in%scalar_value
	else
	 ierr=33
	endif
	return
	contains

	 logical function contr_ptrn_ok(cptrn,rank_in,rank_out)
	 integer, intent(in):: rank_in,rank_out,cptrn(1:rank_in)
	 integer j0,j1,jbus(1:rank_out)
	 contr_ptrn_ok=.TRUE.; jbus(1:rank_out)=0
	 do j0=1,rank_in
	  j1=cptrn(j0)
	  if(j1.gt.0) then !uncontracted index
	   if(j1.gt.rank_out) then; contr_ptrn_ok=.FALSE.; return; endif
	   if(jbus(j1).ne.0) then; contr_ptrn_ok=.FALSE.; return; else; jbus(j1)=jbus(j1)+1; endif
	   if(tens_in%tensor_shape%dim_extent(j0).ne.tens_out%tensor_shape%dim_extent(j1)) then
	    contr_ptrn_ok=.FALSE.; return
	   endif
	  elseif(j1.lt.0) then !contracted index
	   if(-j1.gt.rank_in.or.-j1.eq.j0) then; contr_ptrn_ok=.FALSE.; return; endif
	   if(cptrn(-j1).ne.-j0) then; contr_ptrn_ok=.FALSE.; return; endif
	   if(tens_in%tensor_shape%dim_extent(j0).ne.tens_in%tensor_shape%dim_extent(-j1)) then
	    contr_ptrn_ok=.FALSE.; return
	   endif
	  else
	   contr_ptrn_ok=.FALSE.
	  endif
	 enddo
	 do j0=1,rank_out; if(jbus(j0).ne.1) then; contr_ptrn_ok=.FALSE.; return; endif; enddo
	 return
	 end function contr_ptrn_ok

	 logical function ord_rest_ok(ordr,cptrn,rank_in,rank_out) !`Finish
	 integer, intent(in):: rank_in,rank_out,cptrn(1:rank_in),ordr(1:rank_in)
	 integer j0
	 ord_rest_ok=.TRUE.
	 return
	 end function ord_rest_ok

	end subroutine tensor_block_trace
!----------------------------------------------------------------------------------------------
	logical function tensor_block_cmp(tens1,tens2,ierr,data_kind,rel,cmp_thresh,diff_count) !PARALLEL
!This function compares two tensor blocks.
!INPUT:
! - tens1, tens2 - two tensor blocks to compare;
! - data_kind - (optional) requested data kind, one of {'r4','r8','c4','c8'};
! - rel - if .TRUE., a relative comparison will be invoked: DIFF(a,b)/ABSMAX(a,b), (default=.FALSE.:absolute comparison);
! - cmp_thresh - (optional) numerical comparison threshold (real8);
!OUTPUT:
! - tensor_block_cmp = .TRUE. if tens1 = tens2 to within the given tolerance, .FALSE. otherwise;
! - diff_count - (optional) number of elements that differ (only for compatible tensors with the same layout);
! - ierr - error code (0:success);
!NOTES:
! - The two tensor blocks must have the same storage layout.
! - If <data_kind> is not specified explicitly, this function will try to use the <tensor_common_data_kind>;
!   if the latter does not exist, the result will be .FALSE.!
	implicit none
	type(tensor_block_t), intent(inout):: tens1,tens2 !(out) because of <tensor_block_layout>
	character(2), intent(in), optional:: data_kind
	logical, intent(in), optional:: rel
	real(8), intent(in), optional:: cmp_thresh
	integer(LONGINT), intent(out), optional:: diff_count
	integer, intent(inout):: ierr
!-----------------------------------------------------
	integer(LONGINT), parameter:: chunk_size=2**17 !chunk size
!-----------------------------------------------------
	integer i,j,k,l,m,n,k0,k1,k2,k3,ks,kf
	integer(LONGINT) l0,l1,l2,diffc
	character(2) dtk
	real(4) cmp_thr4,f1,f2
	real(8) cmp_thr8,d1,d2
	logical no_exit,rel_comp

	ierr=0
	if(present(rel)) then; rel_comp=rel; else; rel_comp=.FALSE.; endif !default is the absolute comparison (not relative)
	if(present(cmp_thresh)) then
	 cmp_thr8=cmp_thresh
	else
	 if(rel_comp) then; cmp_thr8=REL_CMP_THRESH; else; cmp_thr8=ABS_CMP_THRESH; endif
	endif
	if(present(diff_count)) then; diff_count=0_LONGINT; no_exit=.TRUE.; else; no_exit=.FALSE.; endif
	tensor_block_cmp=tensor_block_compatible(tens1,tens2,ierr,no_check_data_kinds=.TRUE.)
	if(ierr.ne.0) then; tensor_block_cmp=.FALSE.; ierr=1; return; endif
	if(tensor_block_cmp.and.tens1%tensor_shape%num_dim.gt.0) then !two tensors
	 k1=tensor_block_layout(tens1,ierr); if(ierr.ne.0) then; ierr=2; return; endif
	 k2=tensor_block_layout(tens2,ierr); if(ierr.ne.0) then; ierr=3; return; endif
	 if(k1.ne.k2) then; tensor_block_cmp=.FALSE.; ierr=4; return; endif !storage layouts differ
	 if(present(data_kind)) then !find common data kind
	  dtk=data_kind
	 else
	  dtk=tensor_common_data_kind(tens1,tens2,ierr); if(ierr.ne.0) then; tensor_block_cmp=.FALSE.; ierr=5; return; endif
	 endif
	 diffc=0_LONGINT
	 select case(k1)
	 case(not_allocated,scalar_tensor) !this case is treated separately
	  tensor_block_cmp=.FALSE.; ierr=6
	 case(dimension_led,bricked_dense,bricked_ordered)
	  select case(dtk)
	  case('--') !tensor blocks do not have a common data kind (cannot be directly compared)
	   tensor_block_cmp=.FALSE.
	  case('r4','R4')
	   cmp_thr4=real(cmp_thr8,4)
	   if(associated(tens1%data_real4).and.associated(tens2%data_real4)) then
	    l1=size(tens1%data_real4); l2=size(tens2%data_real4)
	    if(l1.eq.l2.and.l1.eq.tens1%tensor_block_size.and.l1.eq.tens2%tensor_block_size.and.l1.gt.0) then
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(l0,l2) FIRSTPRIVATE(cmp_thr4) REDUCTION(+:diffc)
	     do l0=0_LONGINT,l1-1_LONGINT,chunk_size
	      if(rel_comp) then !relative
!$OMP DO SCHEDULE(GUIDED)
	       do l2=l0,min(l0+chunk_size-1_LONGINT,l1-1_LONGINT)
	        f1=abs(tens1%data_real4(l2)); f2=abs(tens2%data_real4(l2))
	        if(abs(tens1%data_real4(l2)-tens2%data_real4(l2))/max(f1,f2).gt.cmp_thr4) diffc=diffc+1_LONGINT
	       enddo
!$OMP END DO
	      else !absolute
!$OMP DO SCHEDULE(GUIDED)
	       do l2=l0,min(l0+chunk_size-1_LONGINT,l1-1_LONGINT)
	        if(abs(tens1%data_real4(l2)-tens2%data_real4(l2)).gt.cmp_thr4) diffc=diffc+1_LONGINT
	       enddo
!$OMP END DO
	      endif
!$OMP CRITICAL
	      if(diffc.gt.0_LONGINT.and.tensor_block_cmp) tensor_block_cmp=.FALSE.
!$OMP END CRITICAL
!$OMP BARRIER
!$OMP FLUSH(tensor_block_cmp)
	      if(.not.(tensor_block_cmp.or.no_exit)) exit
	     enddo
!$OMP END PARALLEL
	    else
	     tensor_block_cmp=.FALSE.; ierr=7
	    endif
	   else
	    tensor_block_cmp=.FALSE.; ierr=8
	   endif
	  case('r8','R8')
	   if(associated(tens1%data_real8).and.associated(tens2%data_real8)) then
	    l1=size(tens1%data_real8); l2=size(tens2%data_real8)
	    if(l1.eq.l2.and.l1.eq.tens1%tensor_block_size.and.l1.eq.tens2%tensor_block_size.and.l1.gt.0) then
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(l0,l2) FIRSTPRIVATE(cmp_thr8) REDUCTION(+:diffc)
	     do l0=0_LONGINT,l1-1_LONGINT,chunk_size
	      if(rel_comp) then !relative
!$OMP DO SCHEDULE(GUIDED)
	       do l2=l0,min(l0+chunk_size-1_LONGINT,l1-1_LONGINT)
	        d1=abs(tens1%data_real8(l2)); d2=abs(tens2%data_real8(l2))
	        if(abs(tens1%data_real8(l2)-tens2%data_real8(l2))/max(d1,d2).gt.cmp_thr8) diffc=diffc+1_LONGINT
	       enddo
!$OMP END DO
	      else !absolute
!$OMP DO SCHEDULE(GUIDED)
	       do l2=l0,min(l0+chunk_size-1_LONGINT,l1-1_LONGINT)
	        if(abs(tens1%data_real8(l2)-tens2%data_real8(l2)).gt.cmp_thr8) diffc=diffc+1_LONGINT
	       enddo
!$OMP END DO
	      endif
!$OMP CRITICAL
	      if(diffc.gt.0_LONGINT.and.tensor_block_cmp) tensor_block_cmp=.FALSE.
!$OMP END CRITICAL
!$OMP BARRIER
!$OMP FLUSH(tensor_block_cmp)
	      if(.not.(tensor_block_cmp.or.no_exit)) exit
	     enddo
!$OMP END PARALLEL
	    else
	     tensor_block_cmp=.FALSE.; ierr=9
	    endif
	   else
	    tensor_block_cmp=.FALSE.; ierr=10
	   endif
	  case('c4','C4')
	   cmp_thr4=real(cmp_thr8,4)
	   if(associated(tens1%data_cmplx4).and.associated(tens2%data_cmplx4)) then
	    l1=size(tens1%data_cmplx4); l2=size(tens2%data_cmplx4)
	    if(l1.eq.l2.and.l1.eq.tens1%tensor_block_size.and.l1.eq.tens2%tensor_block_size.and.l1.gt.0) then
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(l0,l2) FIRSTPRIVATE(cmp_thr4)
	     do l0=0_LONGINT,l1-1_LONGINT,chunk_size
	      if(rel_comp) then !relative
!$OMP DO SCHEDULE(GUIDED)
	       do l2=l0,min(l0+chunk_size-1_LONGINT,l1-1_LONGINT)
	        f1=abs(tens1%data_cmplx4(l2)); f2=abs(tens2%data_cmplx4(l2))
	        if(abs(tens1%data_cmplx4(l2)-tens2%data_cmplx4(l2))/max(f1,f2).gt.cmp_thr4) diffc=diffc+1_LONGINT
	       enddo
!$OMP END DO
	      else !absolute
!$OMP DO SCHEDULE(GUIDED)
	       do l2=l0,min(l0+chunk_size-1_LONGINT,l1-1_LONGINT)
	        if(abs(tens1%data_cmplx4(l2)-tens2%data_cmplx4(l2)).gt.cmp_thr4) diffc=diffc+1_LONGINT
	       enddo
!$OMP END DO
	      endif
!$OMP CRITICAL
	      if(diffc.gt.0_LONGINT.and.tensor_block_cmp) tensor_block_cmp=.FALSE.
!$OMP END CRITICAL
!$OMP BARRIER
!$OMP FLUSH(tensor_block_cmp)
	      if(.not.(tensor_block_cmp.or.no_exit)) exit
	     enddo
!$OMP END PARALLEL
	    else
	     tensor_block_cmp=.FALSE.; ierr=11
	    endif
	   else
	    tensor_block_cmp=.FALSE.; ierr=12
	   endif
	  case('c8','C8')
	   if(associated(tens1%data_cmplx8).and.associated(tens2%data_cmplx8)) then
	    l1=size(tens1%data_cmplx8); l2=size(tens2%data_cmplx8)
	    if(l1.eq.l2.and.l1.eq.tens1%tensor_block_size.and.l1.eq.tens2%tensor_block_size.and.l1.gt.0) then
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(l0,l2) FIRSTPRIVATE(cmp_thr8)
	     do l0=0_LONGINT,l1-1_LONGINT,chunk_size
	      if(rel_comp) then !relative
!$OMP DO SCHEDULE(GUIDED)
	       do l2=l0,min(l0+chunk_size-1_LONGINT,l1-1_LONGINT)
	        d1=abs(tens1%data_cmplx8(l2)); d2=abs(tens2%data_cmplx8(l2))
	        if(abs(tens1%data_cmplx8(l2)-tens2%data_cmplx8(l2))/max(d1,d2).gt.cmp_thr8) diffc=diffc+1_LONGINT
	       enddo
!$OMP END DO
	      else !absolute
!$OMP DO SCHEDULE(GUIDED)
	       do l2=l0,min(l0+chunk_size-1_LONGINT,l1-1_LONGINT)
	        if(abs(tens1%data_cmplx8(l2)-tens2%data_cmplx8(l2)).gt.cmp_thr8) diffc=diffc+1_LONGINT
	       enddo
!$OMP END DO
	      endif
!$OMP CRITICAL
	      if(diffc.gt.0_LONGINT.and.tensor_block_cmp) tensor_block_cmp=.FALSE.
!$OMP END CRITICAL
!$OMP BARRIER
!$OMP FLUSH(tensor_block_cmp)
	      if(.not.(tensor_block_cmp.or.no_exit)) exit
	     enddo
!$OMP END PARALLEL
	    else
	     tensor_block_cmp=.FALSE.; ierr=13
	    endif
	   else
	    tensor_block_cmp=.FALSE.; ierr=14
	   endif
	  case default
	   tensor_block_cmp=.FALSE.; ierr=15
	  end select
	 case(sparse_list)
	  !`Future
	 case(compressed)
	  !`Future
	 case default
	  tensor_block_cmp=.FALSE.; ierr=16
	 end select
	 if(present(diff_count)) diff_count=diffc
	elseif(tensor_block_cmp.and.tens1%tensor_shape%num_dim.eq.0) then !two scalars
	 if(rel_comp) then !relative
	  if(abs(tens1%scalar_value-tens2%scalar_value)/max(abs(tens1%scalar_value),abs(tens2%scalar_value)).gt.cmp_thr8) then
	   tensor_block_cmp=.FALSE.; if(present(diff_count)) diff_count=1_LONGINT
	  endif
	 else !absolute
	  if(abs(tens1%scalar_value-tens2%scalar_value).gt.cmp_thr8) then
	   tensor_block_cmp=.FALSE.; if(present(diff_count)) diff_count=1_LONGINT
	  endif
	 endif
	endif
	return
	end function tensor_block_cmp
!--------------------------------------------------------------------------
	subroutine tensor_block_copy(tens_in,tens_out,ierr,transp,arg_conj) !PARALLEL
!This subroutine makes a copy of a tensor block with an optional index permutation.
!INPUT:
! - tens_in - input tensor;
! - transp(0:*) - (optional) O2N index permutation;
! - arg_conj - (optional) argument complex conjugation (Bit 0 -> Destination, Bit 1 -> Left);
!OUTPUT:
! - tens_out - output tensor;
! - ierr - error code (0:success).
!NOTE:
! - All allocated data kinds will be copied (no further sync is required).
	implicit none
	type(tensor_block_t), intent(inout):: tens_in !(out) because of <tensor_block_layout> because of <tensor_block_shape_ok>
	type(tensor_block_t), intent(inout):: tens_out
	integer, intent(in), optional:: transp(0:*)
	integer, intent(in), optional:: arg_conj
	integer, intent(inout):: ierr
	integer:: i,j,k,l,m,n,k0,k1,k2,k3,ks,kf
	integer:: trn(0:max_tensor_rank)
	logical:: compat,trivial,dconj,lconj

	ierr=0; n=tens_in%tensor_shape%num_dim
	if(present(arg_conj)) then
	 k=arg_conj
	 dconj=(mod(k,2).eq.1); k=k/2
	 lconj=(mod(k,2).eq.1)
	 if(dconj) then; dconj=.FALSE.; lconj=.not.lconj; endif
	else
	 dconj=.FALSE.; lconj=.FALSE.
	endif
	if(n.gt.0) then !true tensor
	 if(present(transp)) then
	  trn(0:n)=transp(0:n); if(.not.perm_ok(n,trn)) then; ierr=1; return; endif
	  trivial=perm_trivial(n,trn)
	 else
	  trn(0:n)=(/+1,(j,j=1,n)/); trivial=.TRUE.
	 endif
!Check tensor block shapes:
	 compat=tensor_block_compatible(tens_in,tens_out,ierr,trn); if(ierr.ne.0) then; ierr=2; return; endif
	 if(.not.compat) then
	  call tensor_block_mimic(tens_in,tens_out,ierr); if(ierr.ne.0) then; ierr=3; return; endif
	 endif
!Scalar value:
	 if(lconj) then
	  tens_out%scalar_value=conjg(tens_in%scalar_value)
	 else
	  tens_out%scalar_value=tens_in%scalar_value
	 endif
!Tensor shape:
	 tens_out%tensor_shape%dim_extent(trn(1:n))=tens_in%tensor_shape%dim_extent(1:n)
	 tens_out%tensor_shape%dim_divider(trn(1:n))=tens_in%tensor_shape%dim_divider(1:n)
	 tens_out%tensor_shape%dim_group(trn(1:n))=tens_in%tensor_shape%dim_group(1:n)
!Determine the tensor block storage layout:
	 ks=tensor_block_layout(tens_in,ierr); if(ierr.ne.0) then; ierr=4; return; endif
	 kf=tensor_block_layout(tens_out,ierr); if(ierr.ne.0) then; ierr=5; return; endif
	 if(ks.ne.kf) then; ierr=6; return; endif !tensor block storage layouts differ
	 if(trivial) ks=dimension_led !for a direct copy the storage layout is irrelevant
	 select case(ks)
	 case(dimension_led)
!Data:
 !REAL4:
	  if(associated(tens_in%data_real4)) then
	   if(tens_in%tensor_block_size.gt.1_LONGINT) then
	    if(TRANS_SHMEM) then
	     call tensor_block_copy_dlf(n,tens_in%tensor_shape%dim_extent,trn,tens_in%data_real4,tens_out%data_real4,ierr)
	     if(ierr.ne.0) then; ierr=7; return; endif
	    else
	     call tensor_block_copy_scatter_dlf(n,tens_in%tensor_shape%dim_extent,trn,tens_in%data_real4,tens_out%data_real4,ierr)
	     if(ierr.ne.0) then; ierr=8; return; endif
	    endif
	   elseif(tens_in%tensor_block_size.eq.1_LONGINT) then
	    tens_out%data_real4(0)=tens_in%data_real4(0)
	   else
	    ierr=9; return
	   endif
	  endif
 !REAL8:
	  if(associated(tens_in%data_real8)) then
	   if(tens_in%tensor_block_size.gt.1_LONGINT) then
	    if(TRANS_SHMEM) then
	     call tensor_block_copy_dlf(n,tens_in%tensor_shape%dim_extent,trn,tens_in%data_real8,tens_out%data_real8,ierr)
	     if(ierr.ne.0) then; ierr=10; return; endif
	    else
	     call tensor_block_copy_scatter_dlf(n,tens_in%tensor_shape%dim_extent,trn,tens_in%data_real8,tens_out%data_real8,ierr)
	     if(ierr.ne.0) then; ierr=11; return; endif
	    endif
	   elseif(tens_in%tensor_block_size.eq.1_LONGINT) then
	    tens_out%data_real8(0)=tens_in%data_real8(0)
	   else
	    ierr=12; return
	   endif
	  endif
 !COMPLEX4:
	  if(associated(tens_in%data_cmplx4)) then
	   if(tens_in%tensor_block_size.gt.1_LONGINT) then
	    if(TRANS_SHMEM) then
	     call tensor_block_copy_dlf(n,tens_in%tensor_shape%dim_extent,trn,tens_in%data_cmplx4,tens_out%data_cmplx4,&
	                               &ierr,lconj)
	     if(ierr.ne.0) then; ierr=13; return; endif
	    else
	     call tensor_block_copy_scatter_dlf(n,tens_in%tensor_shape%dim_extent,trn,tens_in%data_cmplx4,tens_out%data_cmplx4,&
	                                       &ierr,lconj)
	     if(ierr.ne.0) then; ierr=14; return; endif
	    endif
	   elseif(tens_in%tensor_block_size.eq.1_LONGINT) then
	    if(lconj) then
	     tens_out%data_cmplx4(0)=conjg(tens_in%data_cmplx4(0))
	    else
	     tens_out%data_cmplx4(0)=tens_in%data_cmplx4(0)
	    endif
	   else
	    ierr=15; return
	   endif
	  endif
 !COMPLEX8:
	  if(associated(tens_in%data_cmplx8)) then
	   if(tens_in%tensor_block_size.gt.1_LONGINT) then
	    if(TRANS_SHMEM) then
	     call tensor_block_copy_dlf(n,tens_in%tensor_shape%dim_extent,trn,tens_in%data_cmplx8,tens_out%data_cmplx8,&
	                               &ierr,lconj)
	     if(ierr.ne.0) then; ierr=16; return; endif
	    else
	     call tensor_block_copy_scatter_dlf(n,tens_in%tensor_shape%dim_extent,trn,tens_in%data_cmplx8,tens_out%data_cmplx8,&
	                                       &ierr,lconj)
	     if(ierr.ne.0) then; ierr=17; return; endif
	    endif
	   elseif(tens_in%tensor_block_size.eq.1_LONGINT) then
	    if(lconj) then
	     tens_out%data_cmplx8(0)=conjg(tens_in%data_cmplx8(0))
	    else
	     tens_out%data_cmplx8(0)=tens_in%data_cmplx8(0)
	    endif
	   else
	    ierr=18; return
	   endif
	  endif
	 case(bricked_dense,bricked_ordered)
	  !`Future
	 case(sparse_list)
	  !`Future
	 case(compressed)
	  !`Future
	 case default
	  ierr=19; return
	 end select
	elseif(n.eq.0) then !scalar (0-dimension tensor)
	 if(tens_out%tensor_shape%num_dim.gt.0) then
	  call tensor_block_destroy(tens_out,ierr); if(ierr.ne.0) then; ierr=20; return; endif
	 endif
	 tens_out%tensor_shape%num_dim=0; tens_out%tensor_block_size=tens_in%tensor_block_size
	 if(lconj) then
	  tens_out%scalar_value=conjg(tens_in%scalar_value)
	 else
	  tens_out%scalar_value=tens_in%scalar_value
	 endif
	else !tens_in%tensor_shape%num_dim<0
	 call tensor_block_destroy(tens_out,ierr); if(ierr.ne.0) then; ierr=21; return; endif
	endif
	return
	end subroutine tensor_block_copy
!----------------------------------------------------------------------------------------------
	subroutine tensor_block_add(tens0,tens1,ierr,scale_fac,arg_conj,data_kind,accumulative) !PARALLEL
!This subroutine adds tensor block <tens1> to tensor block <tens0>:
!tens0(:)+=tens1(:)*scale_fac
!INPUT:
! - tens0, tens1 - initialized! tensor blocks;
! - scale_fac - (optional) scaling factor;
! - arg_conj - (optional) argument complex conjugation (Bit 0 -> Destination, Bit 1 -> Left);
! - data_kind - (optional) requested data kind, one of {'r4','r8','c4','c8'};
! - accumulative - (optional) whether or not the tensor addition is accumulative in the destination tensor;
!OUTPUT:
! - tens0 - modified tensor block;
! - ierr - error code (0:success);
!NOTES:
! - If the <data_kind> is not specified explicitly, then all common data kinds will be processed;
!   otherwise, the data kinds syncronization will be invoked.
! - Not-allocated (still compatible) tensor blocks will be simply ignored.
        implicit none
        type(tensor_block_t), intent(inout):: tens0    !inout: destination tensor
        type(tensor_block_t), intent(inout):: tens1    !in: left tensor: (out) because <tens1> might need data kinds syncronization to become compatible with <tens0>
        integer, intent(inout):: ierr                  !out: error code
        complex(8), intent(in), optional:: scale_fac   !in: scaling prefactor
        integer, intent(in), optional:: arg_conj       !in: argument complex conjugation (Bit 0 -> Destination, Bit 1 -> Left);
        character(2), intent(in), optional:: data_kind !in: requested data kind, one of {'r4','r8','c4','c8'};
        logical, intent(in), optional:: accumulative   !in: whether or not the tensor addition is accumulative in the destination tensor
        integer:: i,j,k,l,m,n,ks,kf
        integer(LONGINT):: l0,l1,ls
        character(2):: dtk,slk,dlt
        logical:: tencom,scale_present,dconj,lconj,accum
        real(4):: val_r4
        real(8):: val_r8
        complex(4):: val_c4,l_c4
        complex(8):: val_c8,l_c8

	ierr=0
	accum=.TRUE.; if(present(accumulative)) accum=accumulative
	ks=tensor_block_layout(tens0,ierr); if(ierr.ne.0) then; ierr=1; return; endif
	kf=tensor_block_layout(tens1,ierr); if(ierr.ne.0) then; ierr=2; return; endif
	if(ks.ne.kf) then; ierr=3; return; endif !tensor storage layouts differ
	if(present(scale_fac)) then
	 val_c8=scale_fac; scale_present=.TRUE.
	else
	 val_c8=(1d0,0d0); scale_present=.FALSE.
	endif
	if(present(arg_conj)) then
	 k=arg_conj
	 dconj=(mod(k,2).eq.1); k=k/2
	 lconj=(mod(k,2).eq.1)
	 if(dconj) then; dconj=.FALSE.; lconj=.not.lconj; endif
	else
	 dconj=.FALSE.; lconj=.FALSE.
	endif
	if(present(data_kind)) then; dtk=data_kind; else; dtk='  '; endif
	tencom=tensor_block_compatible(tens0,tens1,ierr,no_check_data_kinds=.TRUE.); if(ierr.ne.0) then; ierr=4; return; endif
	if(tencom) then
	 if(tens0%tensor_shape%num_dim.eq.0) then !scalars
	  if(.not.accum) tens0%scalar_value=(0d0,0d0)
	  if(lconj) then; l_c8=conjg(tens1%scalar_value); else; l_c8=tens1%scalar_value; endif
	  tens0%scalar_value=tens0%scalar_value+l_c8*val_c8
	 elseif(tens0%tensor_shape%num_dim.gt.0) then !true tensors
	  select case(ks)
	  case(dimension_led,bricked_dense,bricked_ordered)
	   ls=tens0%tensor_block_size
	   if(ls.gt.0_LONGINT) then
 !REAL4:
	    if(dtk.eq.'r4'.or.dtk.eq.'R4'.or.dtk.eq.'  ') then
	     if(associated(tens0%data_real4)) then
	      if(.not.associated(tens1%data_real4)) then
	       slk=tensor_master_data_kind(tens1,ierr); if(ierr.ne.0) then; ierr=5; return; endif
	       if(slk.eq.'--') then; ierr=6; return; endif
	       dlt='r4'; call tensor_block_sync(tens1,slk,ierr,dlt); if(ierr.ne.0) then; ierr=7; return; endif
	      else
	       dlt='  '
	      endif
	      if(size(tens0%data_real4).eq.ls.and.tens1%tensor_block_size.eq.ls) then
               if(accum) then !accumulating
                if(scale_present) then !scaling present
                 val_r4=real(cmplx8_to_real8(val_c8),4)
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED) FIRSTPRIVATE(val_r4)
                 do l0=0_LONGINT,ls-1_LONGINT
                  tens0%data_real4(l0)=tens0%data_real4(l0)+tens1%data_real4(l0)*val_r4
                 enddo
!$OMP END PARALLEL DO
                else !no scaling
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED)
                 do l0=0_LONGINT,ls-1_LONGINT
                  tens0%data_real4(l0)=tens0%data_real4(l0)+tens1%data_real4(l0)
                 enddo
!$OMP END PARALLEL DO
                endif
               else !overwriting
                if(scale_present) then !scaling present
                 val_r4=real(cmplx8_to_real8(val_c8),4)
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED) FIRSTPRIVATE(val_r4)
                 do l0=0_LONGINT,ls-1_LONGINT
                  tens0%data_real4(l0)=tens1%data_real4(l0)*val_r4
                 enddo
!$OMP END PARALLEL DO
                else !no scaling
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED)
                 do l0=0_LONGINT,ls-1_LONGINT
                  tens0%data_real4(l0)=tens1%data_real4(l0)
                 enddo
!$OMP END PARALLEL DO
                endif
               endif
	       if(dlt.ne.'  ') then
	        call tensor_block_sync(tens1,dlt,ierr,'--'); if(ierr.ne.0) then; ierr=8; return; endif
	       endif
	      else
	       ierr=9; return
	      endif
	     else
	      if(dtk.eq.'r4'.or.dtk.eq.'R4') then; ierr=10; return; endif
	     endif
	    endif
 !REAL8:
	    if(dtk.eq.'r8'.or.dtk.eq.'R8'.or.dtk.eq.'  ') then
	     if(associated(tens0%data_real8)) then
	      if(.not.associated(tens1%data_real8)) then
	       slk=tensor_master_data_kind(tens1,ierr); if(ierr.ne.0) then; ierr=11; return; endif
	       if(slk.eq.'--') then; ierr=12; return; endif
	       dlt='r8'; call tensor_block_sync(tens1,slk,ierr,dlt); if(ierr.ne.0) then; ierr=13; return; endif
	      else
	       dlt='  '
	      endif
	      if(size(tens0%data_real8).eq.ls.and.tens1%tensor_block_size.eq.ls) then
               if(accum) then !accumulating
                if(scale_present) then !scaling present
                 val_r8=cmplx8_to_real8(val_c8)
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED) FIRSTPRIVATE(val_r8)
                 do l0=0_LONGINT,ls-1_LONGINT
                  tens0%data_real8(l0)=tens0%data_real8(l0)+tens1%data_real8(l0)*val_r8
                 enddo
!$OMP END PARALLEL DO
                else !no scaling
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED)
                 do l0=0_LONGINT,ls-1_LONGINT
                  tens0%data_real8(l0)=tens0%data_real8(l0)+tens1%data_real8(l0)
                 enddo
!$OMP END PARALLEL DO
                endif
               else !overwriting
                if(scale_present) then !scaling present
                 val_r8=cmplx8_to_real8(val_c8)
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED) FIRSTPRIVATE(val_r8)
                 do l0=0_LONGINT,ls-1_LONGINT
                  tens0%data_real8(l0)=tens1%data_real8(l0)*val_r8
                 enddo
!$OMP END PARALLEL DO
                else !no scaling
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED)
                 do l0=0_LONGINT,ls-1_LONGINT
                  tens0%data_real8(l0)=tens1%data_real8(l0)
                 enddo
!$OMP END PARALLEL DO
                endif
               endif
	       if(dlt.ne.'  ') then
	        call tensor_block_sync(tens1,dlt,ierr,'--'); if(ierr.ne.0) then; ierr=14; return; endif
	       endif
	      else
	       ierr=15; return
	      endif
	     else
	      if(dtk.eq.'r8'.or.dtk.eq.'R8') then; ierr=16; return; endif
	     endif
	    endif
 !COMPLEX4:
	    if(dtk.eq.'c4'.or.dtk.eq.'C4'.or.dtk.eq.'  ') then
	     if(associated(tens0%data_cmplx4)) then
	      if(.not.associated(tens1%data_cmplx4)) then
	       slk=tensor_master_data_kind(tens1,ierr); if(ierr.ne.0) then; ierr=17; return; endif
	       if(slk.eq.'--') then; ierr=18; return; endif
	       dlt='c4'; call tensor_block_sync(tens1,slk,ierr,dlt); if(ierr.ne.0) then; ierr=19; return; endif
	      else
	       dlt='  '
	      endif
	      if(size(tens0%data_cmplx4).eq.ls.and.tens1%tensor_block_size.eq.ls) then
               if(accum) then !accumlating
                if(lconj) then !left tensor is conjugated
                 if(scale_present) then !scaling present
                  val_c4=cmplx(val_c8,kind=4)
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED) FIRSTPRIVATE(val_c4)
                  do l0=0_LONGINT,ls-1_LONGINT
                   tens0%data_cmplx4(l0)=tens0%data_cmplx4(l0)+conjg(tens1%data_cmplx4(l0))*val_c4
                  enddo
!$OMP END PARALLEL DO
                 else !no scaling
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED)
                  do l0=0_LONGINT,ls-1_LONGINT
                   tens0%data_cmplx4(l0)=tens0%data_cmplx4(l0)+conjg(tens1%data_cmplx4(l0))
                  enddo
!$OMP END PARALLEL DO
                 endif
                else !left tensor is normal
                 if(scale_present) then !scaling present
                  val_c4=cmplx(val_c8,kind=4)
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED) FIRSTPRIVATE(val_c4)
                  do l0=0_LONGINT,ls-1_LONGINT;
                   tens0%data_cmplx4(l0)=tens0%data_cmplx4(l0)+tens1%data_cmplx4(l0)*val_c4
                  enddo
!$OMP END PARALLEL DO
                 else !no scaling
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED)
                  do l0=0_LONGINT,ls-1_LONGINT
                   tens0%data_cmplx4(l0)=tens0%data_cmplx4(l0)+tens1%data_cmplx4(l0)
                  enddo
!$OMP END PARALLEL DO
                 endif
                endif
               else !overwriting
                if(lconj) then !left tensor is conjugated
                 if(scale_present) then !scaling present
                  val_c4=cmplx(val_c8,kind=4)
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED) FIRSTPRIVATE(val_c4)
                  do l0=0_LONGINT,ls-1_LONGINT
                   tens0%data_cmplx4(l0)=conjg(tens1%data_cmplx4(l0))*val_c4
                  enddo
!$OMP END PARALLEL DO
                 else !no scaling
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED)
                  do l0=0_LONGINT,ls-1_LONGINT
                   tens0%data_cmplx4(l0)=conjg(tens1%data_cmplx4(l0))
                  enddo
!$OMP END PARALLEL DO
                 endif
                else !left tensor is normal
                 if(scale_present) then !scaling present
                  val_c4=cmplx(val_c8,kind=4)
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED) FIRSTPRIVATE(val_c4)
                  do l0=0_LONGINT,ls-1_LONGINT;
                   tens0%data_cmplx4(l0)=tens1%data_cmplx4(l0)*val_c4
                  enddo
!$OMP END PARALLEL DO
                 else !no scaling
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED)
                  do l0=0_LONGINT,ls-1_LONGINT
                   tens0%data_cmplx4(l0)=tens1%data_cmplx4(l0)
                  enddo
!$OMP END PARALLEL DO
                 endif
                endif
               endif
	       if(dlt.ne.'  ') then
	        call tensor_block_sync(tens1,dlt,ierr,'--'); if(ierr.ne.0) then; ierr=20; return; endif
	       endif
	      else
	       ierr=21; return
	      endif
	     else
	      if(dtk.eq.'c4'.or.dtk.eq.'C4') then; ierr=22; return; endif
	     endif
	    endif
 !COMPLEX8:
	    if(dtk.eq.'c8'.or.dtk.eq.'C8'.or.dtk.eq.'  ') then
	     if(associated(tens0%data_cmplx8)) then
	      if(.not.associated(tens1%data_cmplx8)) then
	       slk=tensor_master_data_kind(tens1,ierr); if(ierr.ne.0) then; ierr=23; return; endif
	       if(slk.eq.'--') then; ierr=24; return; endif
	       dlt='c8'; call tensor_block_sync(tens1,slk,ierr,dlt); if(ierr.ne.0) then; ierr=25; return; endif
	      else
	       dlt='  '
	      endif
	      if(size(tens0%data_cmplx8).eq.ls.and.tens1%tensor_block_size.eq.ls) then
               if(accum) then !accumulating
                if(lconj) then !left tensor is conjugated
                 if(scale_present) then !scaling present
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED) FIRSTPRIVATE(val_c8)
                  do l0=0_LONGINT,ls-1_LONGINT
                   tens0%data_cmplx8(l0)=tens0%data_cmplx8(l0)+conjg(tens1%data_cmplx8(l0))*val_c8
                  enddo
!$OMP END PARALLEL DO
                 else !no scaling
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED)
                  do l0=0_LONGINT,ls-1_LONGINT
                   tens0%data_cmplx8(l0)=tens0%data_cmplx8(l0)+conjg(tens1%data_cmplx8(l0))
                  enddo
!$OMP END PARALLEL DO
                 endif
                else !left tensor is normal
                 if(scale_present) then !scaling present
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED) FIRSTPRIVATE(val_c8)
                  do l0=0_LONGINT,ls-1_LONGINT
                   tens0%data_cmplx8(l0)=tens0%data_cmplx8(l0)+tens1%data_cmplx8(l0)*val_c8
                  enddo
!$OMP END PARALLEL DO
                 else !no scaling
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED)
                  do l0=0_LONGINT,ls-1_LONGINT
                   tens0%data_cmplx8(l0)=tens0%data_cmplx8(l0)+tens1%data_cmplx8(l0)
                  enddo
!$OMP END PARALLEL DO
                 endif
                endif
               else !overwriting
                if(lconj) then !left tensor is conjugated
                 if(scale_present) then !scaling present
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED) FIRSTPRIVATE(val_c8)
                  do l0=0_LONGINT,ls-1_LONGINT
                   tens0%data_cmplx8(l0)=conjg(tens1%data_cmplx8(l0))*val_c8
                  enddo
!$OMP END PARALLEL DO
                 else !no scaling
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED)
                  do l0=0_LONGINT,ls-1_LONGINT
                   tens0%data_cmplx8(l0)=conjg(tens1%data_cmplx8(l0))
                  enddo
!$OMP END PARALLEL DO
                 endif
                else !left tensor is normal
                 if(scale_present) then !scaling present
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED) FIRSTPRIVATE(val_c8)
                  do l0=0_LONGINT,ls-1_LONGINT
                   tens0%data_cmplx8(l0)=tens1%data_cmplx8(l0)*val_c8
                  enddo
!$OMP END PARALLEL DO
                 else !no scaling
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED)
                  do l0=0_LONGINT,ls-1_LONGINT
                   tens0%data_cmplx8(l0)=tens1%data_cmplx8(l0)
                  enddo
!$OMP END PARALLEL DO
                 endif
                endif
               endif
	       if(dlt.ne.'  ') then
	        call tensor_block_sync(tens1,dlt,ierr,'--'); if(ierr.ne.0) then; ierr=26; return; endif
	       endif
	      else
	       ierr=27; return
	      endif
	     else
	      if(dtk.eq.'c8'.or.dtk.eq.'C8') then; ierr=28; return; endif
	     endif
	    endif
!Sync the destination tensor:
	    if(dtk.ne.'  ') then
	     if(DATA_KIND_SYNC) then
	      call tensor_block_sync(tens0,dtk,ierr); if(ierr.ne.0) then; ierr=29; return; endif
	     endif
	    else
	     slk=tensor_master_data_kind(tens0,ierr); if(ierr.ne.0) then; ierr=30; return; endif
	     if(slk.eq.'--') then; ierr=31; return; endif
	     val_r8=tensor_block_norm2(tens0,ierr,slk); if(ierr.ne.0) then; ierr=32; return; endif
	     tens0%scalar_value=cmplx(dsqrt(val_r8),0d0,kind=8) !Euclidean norm of the destination tensor block
	    endif
	   else
	    ierr=33 !%tensor_block_size less or equal to zero for an allocated tensor block
	   endif
	  case(sparse_list)
	   !`Future
	  case(compressed)
	   !`Future
	  case default
	   ierr=34 !invalid storage layout
	  end select
	 endif
	else
	 ierr=35 !incompatible shapes of the tensor blocks
	endif
	return
	end subroutine tensor_block_add
!-------------------------------------------------------------------------------------------------------------------------
	subroutine tensor_block_contract(contr_ptrn,ltens,rtens,dtens,ierr,alpha,arg_conj,data_kind,ord_rest,accumulative) !PARALLEL
!This subroutine contracts two tensor blocks and accumulates the result into another tensor block:
!dtens(:)+=ltens(:)*rtens(:)
!Author: Dmitry I. Lyakh (Liakh): quant4me@gmail.com
!Possible cases:
! A) tensor+=tensor*tensor (no traces!): all tensor operands can be transposed;
! B) scalar+=tensor*tensor (no traces!): only the left tensor operand can be transposed;
! C) tensor+=tensor*scalar OR tensor+=scalar*tensor (no traces!): no transpose;
! D) scalar+=scalar*scalar: no transpose.
!INPUT:
! - contr_ptrn(1:left_rank+right_rank) - contraction pattern:
!                                        contr_ptrn(1:left_rank) refers to indices of the left tensor argument;
!                                        contr_ptrn(left_rank+1:left_rank+right_rank) refers to indices of the right tensor argument;
!                                        contr_ptrn(x)>0 means that the index is uncontracted and shows the position where it goes;
!                                        contr_ptrn(x)<0 means that the index is contracted and shows the position in the other argument where it is located.
! - ltens - left tensor argument (tensor block);
! - rtens - right tensor argument (tensor block);
! - dtens - initialized! destination tensor argument (tensor block);
! - alpha - BLAS alpha (complex);
! - arg_conj - argument complex conjugation flags: Bit 0 -> Destination, Bit 1 -> Left, Bit 2 -> Right tensor argument;
! - data_kind - (optional) requested data kind, one of {'r4','r8','c4','c8'};
! - ord_rest(1:left_rank+right_rank) - (optional) index ordering restrictions (for contracted indices only);
! - accumulative - (optional) whether or not the tensor contraction is accumulative;
!OUTPUT:
! - dtens - modified destination tensor (tensor block);
! - ierr - error code (0: success);
!NOTES:
! - If <data_kind> is not specified then only the highest present data kind will be processed
!   whereas the present lower-level data kinds of the destination tensor will be syncronized.
        implicit none
        integer, intent(in):: contr_ptrn(1:*)                     !in: digital contraction pattern (see above)
        type(tensor_block_t), intent(inout), target:: ltens,rtens !inout: left and right tensors: (out) because of <tensor_block_layout> because of <tensor_block_shape_ok>
        type(tensor_block_t), intent(inout), target:: dtens       !inout: destination tensor
        integer, intent(inout):: ierr                             !out: error code
        complex(8), intent(in), optional:: alpha                  !in: scaling prefactor
        integer, intent(in), optional:: arg_conj                  !in: argument complex conjugation (Bit 0 -> Destination, Bit 1 -> Left, Bit 2 -> Right)
        character(2), intent(in), optional:: data_kind            !in: preferred data kind
        integer, intent(in), optional:: ord_rest(1:*)             !in: index ordering restrictions (for contracted indices only)
        logical, intent(in), optional:: accumulative              !in: whether or not the tensor contraction is accumulative (into destination tensor)
!----------------------------------------------------
        integer, parameter:: PARTIAL_CONTRACTION=1
        integer, parameter:: FULL_CONTRACTION=2
        integer, parameter:: ADD_TENSOR=3
        integer, parameter:: MULTIPLY_SCALARS=4
!------------------------------------------------
        integer:: i,j,k,l,m,n,k0,k1,k2,k3,ks,kf
        integer(LONGINT):: l0,l1,l2,l3,lld,lrd,lcd
        integer:: ltb,rtb,dtb,lrank,nthr,rrank,drank,nlu,nru,ncd,tst,contr_case,conj,dn2o(0:max_tensor_rank)
        integer, target:: lo2n(0:max_tensor_rank),ro2n(0:max_tensor_rank),do2n(0:max_tensor_rank)
        integer, pointer:: trn(:)
        type(tensor_block_t), pointer:: tens_in,tens_out,ltp,rtp,dtp
        type(tensor_block_t), target:: lta,rta,dta
        character(2):: dtk
        character(1):: ltrm,rtrm
        real(4):: d_r4
        real(8):: d_r8,start_gemm,finish_gemm
        complex(4):: d_c4,l_c4,r_c4
        complex(8):: d_c8,l_c8,r_c8,alf,beta
        logical:: contr_ok,ltransp,rtransp,dtransp,transp,lconj,rconj,dconj,accum

        ierr=0
        nthr=omp_get_max_threads()
#ifdef USE_MKL
        call mkl_set_num_threads(nthr)
#endif
!Get the argument types:
        ltb=tensor_block_layout(ltens,ierr); if(ierr.ne.0) then; ierr=1; return; endif !left-tensor storage layout type
        rtb=tensor_block_layout(rtens,ierr); if(ierr.ne.0) then; ierr=2; return; endif !right-tensor storage layout type
        dtb=tensor_block_layout(dtens,ierr); if(ierr.ne.0) then; ierr=3; return; endif !destination-tensor storage layout type
!Determine the contraction case:
        if(ltb.eq.not_allocated.or.rtb.eq.not_allocated.or.dtb.eq.not_allocated) then; ierr=4; return; endif
        if(ltb.eq.scalar_tensor.and.rtb.eq.scalar_tensor.and.dtb.eq.scalar_tensor) then !multiplication of scalars
         contr_case=MULTIPLY_SCALARS
        elseif((ltb.ne.scalar_tensor.and.rtb.eq.scalar_tensor.and.dtb.ne.scalar_tensor).or.&
              &(ltb.eq.scalar_tensor.and.rtb.ne.scalar_tensor.and.dtb.ne.scalar_tensor)) then
         contr_case=ADD_TENSOR
        elseif(ltb.ne.scalar_tensor.and.rtb.ne.scalar_tensor.and.dtb.eq.scalar_tensor) then
         contr_case=FULL_CONTRACTION
        elseif(ltb.ne.scalar_tensor.and.rtb.ne.scalar_tensor.and.dtb.ne.scalar_tensor) then
         contr_case=PARTIAL_CONTRACTION
        else
         ierr=5; return
        endif
!Check tensor ranks:
        lrank=ltens%tensor_shape%num_dim; rrank=rtens%tensor_shape%num_dim; drank=dtens%tensor_shape%num_dim
        if(lrank.ge.0.and.lrank.le.max_tensor_rank.and.rrank.ge.0.and.rrank.le.max_tensor_rank.and.&
          &drank.ge.0.and.drank.le.max_tensor_rank) then
         if(present(data_kind)) then
          dtk=data_kind
         else
          call determine_data_kind(dtk,ierr); if(ierr.ne.0) then; ierr=6; return; endif
         endif
 !Check the requested contraction pattern:
         contr_ok=contr_ptrn_ok(contr_ptrn,lrank,rrank,drank)
         if(present(ord_rest)) contr_ok=contr_ok.and.ord_rest_ok(ord_rest,contr_ptrn,lrank,rrank,drank)
         if(.not.contr_ok) then; ierr=7; return; endif
!        write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_contract): contraction pattern accepted:",128(1x,i2))') &
!         contr_ptrn(1:lrank+rrank) !debug
!        write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_contract): tensor layouts (left, right, dest): ",i2,1x,i2,1x,i2)') &
!         ltb,rtb,dtb !debug
 !Determine index permutations for all tensor operands together with the numbers of contracted/uncontraced indices (ncd/{nlu,nru}):
         call determine_index_permutations !sets {dtransp,ltransp,rtransp},{do2n,lo2n,ro2n},{ncd,nlu,nru}
!        write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_contract): left uncontr, right uncontr, contr dims: "&
!         &,i2,1x,i2,1x,i2)') nlu,nru,ncd !debug
!        write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_contract): left index permutation (O2N)  :"&
!         &,128(1x,i2))') lo2n(1:lrank) !debug
!        write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_contract): right index permutation (O2N) :"&
!         &,128(1x,i2))') ro2n(1:rrank) !debug
!        write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_contract): result index permutation (O2N):"&
!         &,128(1x,i2))') do2n(1:drank) !debug
 !Determine complex conjugation for all tensor arguments:
         ltrm='T'; rtrm='N'
         if(dtk(1:1).eq.'c'.or.dtk(1:1).eq.'C') then !only complex data types (C4,C8)
  !Determine the need for complex conjugation for all arguments:
          k=arg_conj
          dconj=(mod(k,2).eq.1); k=k/2
          lconj=(mod(k,2).eq.1); k=k/2
          rconj=(mod(k,2).eq.1)
          if(dconj) then; dconj=.FALSE.; lconj=.not.lconj; rconj=.not.rconj; endif
  !Modify right tensor index permutation if needed:
          if(lconj) ltrm='C' !'T' -> 'C'
          if(rconj) then     !'N' -> 'C'
           if(ncd.gt.0.and.nru.gt.0) then
            dn2o(0)=ro2n(0); do k=1,rrank; dn2o(ro2n(k))=k; enddo
            do k=1,ncd; ro2n(dn2o(k))=nru+k; enddo
            do k=ncd+1,rrank; ro2n(dn2o(k))=k-ncd; enddo
            rtransp=.not.perm_trivial(rrank,ro2n)
           endif
           rtrm='C'
          endif
         else
          dconj=.FALSE.; lconj=.FALSE.; rconj=.FALSE.
         endif
 !Transpose/conjugate tensor arguments, if needed:
         nullify(ltp); nullify(rtp); nullify(dtp)
         do k=1,2 !left/right tensor argument switch
          if(k.eq.1) then
           tst=ltb; transp=ltransp; tens_in=>ltens
#ifdef NO_BLAS
           if(lconj.and.(contr_case.eq.PARTIAL_CONTRACTION.or.contr_case.eq.FULL_CONTRACTION)) then
#else
           if(lconj.and.((contr_case.eq.PARTIAL_CONTRACTION.and.DISABLE_BLAS).or.contr_case.eq.FULL_CONTRACTION)) then
#endif
            conj=0+1*2 !this conjugation mask will be used in tensor_block_copy(): Bit X is a conjugation flag for argument X
           else
            conj=0 !all bits are zero => no argument conjugation
           endif
          else
           tst=rtb; transp=rtransp; tens_in=>rtens
#ifdef NO_BLAS
           if(rconj.and.(contr_case.eq.PARTIAL_CONTRACTION.or.contr_case.eq.FULL_CONTRACTION)) then
#else
           if(rconj.and.((contr_case.eq.PARTIAL_CONTRACTION.and.DISABLE_BLAS).or.contr_case.eq.FULL_CONTRACTION)) then
#endif
            conj=0+1*2 !this conjugation mask will be used in tensor_block_copy(): Bit X is a conjugation flag for argument X
           else
            conj=0 !all bits are zero => no argument conjugation
           endif
          endif
          if(tens_in%tensor_shape%num_dim.gt.0.and.(transp.or.(conj.ne.0))) then !true tensor which requires a transpose
!          write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_contract): permutation to be performed for ",i2)') k !debug
           if(k.eq.1) then; trn=>lo2n; tens_out=>lta; else; trn=>ro2n; tens_out=>rta; endif
           select case(tst)
           case(scalar_tensor)
           case(dimension_led)
            call tensor_block_copy(tens_in,tens_out,ierr,transp=trn,arg_conj=conj)
            if(ierr.ne.0) then; ierr=8; goto 999; endif
           case(bricked_dense,bricked_ordered)
            !`Future
           case(sparse_list)
            !`Future
           case(compressed)
            !`Future
           case default
            ierr=9; goto 999
           end select
           if(k.eq.1) then; ltp=>lta; else; rtp=>rta; endif
           nullify(tens_out); nullify(trn)
          elseif(tens_in%tensor_shape%num_dim.eq.0.or.(.not.transp)) then !scalar tensor or no transpose required
           if(k.eq.1) then; ltp=>ltens; else; rtp=>rtens; endif
          else
           ierr=10; goto 999
          endif
          nullify(tens_in)
         enddo !k
         if(dtransp) then !transpose the destination tensor
!         write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_contract): permutation to be performed for ",i2)') 0 !debug
          dn2o(0)=+1; do k=1,drank; dn2o(do2n(k))=k; enddo
          select case(dtb)
          case(scalar_tensor)
          case(dimension_led)
           call tensor_block_copy(dtens,dta,ierr,transp=dn2o); if(ierr.ne.0) then; ierr=11; goto 999; endif
          case(bricked_dense,bricked_ordered)
           !`Future
          case(sparse_list)
           !`Future
          case(compressed)
           !`Future
          case default
           ierr=12; goto 999
          end select
          dtp=>dta
         else !no transpose for the destination tensor
          dtp=>dtens
         endif
!        write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_contract): arguments are ready to be processed!")') !debug
 !Calculate matrix dimensions:
!        write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_contract): argument pointer status (l,r,d): ",l1,1x,l1,1x,l1)') &
!         associated(ltp),associated(rtp),associated(dtp) !debug
!        write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_contract): left index extents  :",128(1x,i4))') &
!         ltp%tensor_shape%dim_extent(1:lrank) !debug
!        write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_contract): right index extents :",128(1x,i4))') &
!         rtp%tensor_shape%dim_extent(1:rrank) !debug
!        write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_contract): result index extents:",128(1x,i4))') &
!         dtp%tensor_shape%dim_extent(1:drank) !debug
         call calculate_matrix_dimensions(dtb,nlu,nru,dtp,lld,lrd,ierr); if(ierr.ne.0) then; ierr=13; goto 999; endif
         call calculate_matrix_dimensions(ltb,ncd,nlu,ltp,lcd,l0,ierr); if(ierr.ne.0) then; ierr=14; goto 999; endif
         if(rtrm.eq.'C') then !R(r,c) matrix shape
          call calculate_matrix_dimensions(rtb,nru,ncd,rtp,l2,l1,ierr); if(ierr.ne.0) then; ierr=15; goto 999; endif
         else !R(c,r) matrix shape
          call calculate_matrix_dimensions(rtb,ncd,nru,rtp,l1,l2,ierr); if(ierr.ne.0) then; ierr=16; goto 999; endif
         endif
!        write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_contract): matrix dimensions (left,right,contr): "&
!         &,i10,1x,i10,1x,i10)') lld,lrd,lcd !debug
         if(l0.ne.lld.or.l1.ne.lcd.or.l2.ne.lrd) then; ierr=17; goto 999; endif
         if(rtrm.eq.'C') then; l2=lrd; else; l2=lcd; endif !leading dimension for the right matrix
 !Multiply two matrices (dtp += ltp * rtp):
         beta=(1d0,0d0); accum=.TRUE.
         if(present(accumulative)) then
          accum=accumulative
          if(.not.accum) then; dtp%scalar_value=(0d0,0d0); beta=(0d0,0d0); endif
         endif
         if(present(alpha)) then; alf=alpha; else; alf=(1d0,0d0); endif
	 start_gemm=thread_wtime() !debug
	 select case(contr_case)
	 case(PARTIAL_CONTRACTION) !destination is an array
	  select case(dtk)
	  case('r4','R4')
#ifdef NO_BLAS
	   call tensor_block_pcontract_dlf(lld,lrd,lcd,ltp%data_real4,rtp%data_real4,dtp%data_real4,ierr,real(alf,4),real(beta,4))
	   if(ierr.ne.0) then; ierr=18; goto 999; endif
#else
	   if(.not.DISABLE_BLAS) then
	    call sgemm(ltrm,rtrm,int(lld,4),int(lrd,4),int(lcd,4),real(alf,4),ltp%data_real4,int(lcd,4),rtp%data_real4,int(l2,4),&
	              &real(beta,4),dtp%data_real4,int(lld,4))
	   else
	    call tensor_block_pcontract_dlf(lld,lrd,lcd,ltp%data_real4,rtp%data_real4,dtp%data_real4,ierr,real(alf,4),real(beta,4))
	    if(ierr.ne.0) then; ierr=19; goto 999; endif
	   endif
#endif
	  case('r8','R8')
#ifdef NO_BLAS
	   call tensor_block_pcontract_dlf(lld,lrd,lcd,ltp%data_real8,rtp%data_real8,dtp%data_real8,ierr,real(alf,8),real(beta,8))
	   if(ierr.ne.0) then; ierr=20; goto 999; endif
#else
	   if(.not.DISABLE_BLAS) then
	    call dgemm(ltrm,rtrm,int(lld,4),int(lrd,4),int(lcd,4),real(alf,8),ltp%data_real8,int(lcd,4),rtp%data_real8,int(l2,4),&
	              &real(beta,8),dtp%data_real8,int(lld,4))
	   else
	    call tensor_block_pcontract_dlf(lld,lrd,lcd,ltp%data_real8,rtp%data_real8,dtp%data_real8,ierr,real(alf,8),real(beta,8))
	    if(ierr.ne.0) then; ierr=21; goto 999; endif
	   endif
#endif
	  case('c4','C4')
#ifdef NO_BLAS
	   call tensor_block_pcontract_dlf(lld,lrd,lcd,ltp%data_cmplx4,rtp%data_cmplx4,dtp%data_cmplx4,ierr,&
                                          &cmplx(alf,kind=4),cmplx(beta,kind=4))
	   if(ierr.ne.0) then; ierr=22; goto 999; endif
#else
	   if(.not.DISABLE_BLAS) then
	    call cgemm(ltrm,rtrm,int(lld,4),int(lrd,4),int(lcd,4),cmplx(alf,kind=4),ltp%data_cmplx4,int(lcd,4),&
                      &rtp%data_cmplx4,int(l2,4),cmplx(beta,kind=4),dtp%data_cmplx4,int(lld,4))
	   else
	    call tensor_block_pcontract_dlf(lld,lrd,lcd,ltp%data_cmplx4,rtp%data_cmplx4,dtp%data_cmplx4,ierr,&
                                           &cmplx(alf,kind=4),cmplx(beta,kind=4))
	    if(ierr.ne.0) then; ierr=23; goto 999; endif
	   endif
#endif
	  case('c8','C8')
#ifdef NO_BLAS
	   call tensor_block_pcontract_dlf(lld,lrd,lcd,ltp%data_cmplx8,rtp%data_cmplx8,dtp%data_cmplx8,ierr,alf,beta)
	   if(ierr.ne.0) then; ierr=24; goto 999; endif
#else
	   if(.not.DISABLE_BLAS) then
	    call zgemm(ltrm,rtrm,int(lld,4),int(lrd,4),int(lcd,4),alf,ltp%data_cmplx8,int(lcd,4),rtp%data_cmplx8,int(l2,4),&
	              &beta,dtp%data_cmplx8,int(lld,4))
	   else
	    call tensor_block_pcontract_dlf(lld,lrd,lcd,ltp%data_cmplx8,rtp%data_cmplx8,dtp%data_cmplx8,ierr,alf,beta)
	    if(ierr.ne.0) then; ierr=25; goto 999; endif
	   endif
#endif
	  end select
	 case(FULL_CONTRACTION) !destination is a scalar variable
	  select case(dtk)
	  case('r4','R4')
	   d_r4=0.0
	   call tensor_block_fcontract_dlf(lcd,ltp%data_real4,rtp%data_real4,d_r4,ierr,real(alf,4),real(beta,4))
	   if(ierr.ne.0) then; ierr=26; goto 999; endif
	   dtp%scalar_value=dtp%scalar_value+cmplx(real(d_r4,8),0d0,kind=8)
	  case('r8','R8')
	   d_r8=0d0
	   call tensor_block_fcontract_dlf(lcd,ltp%data_real8,rtp%data_real8,d_r8,ierr,real(alf,8),real(beta,8))
	   if(ierr.ne.0) then; ierr=27; goto 999; endif
	   dtp%scalar_value=dtp%scalar_value+cmplx(d_r8,0d0,kind=8)
	  case('c4','C4')
	   d_c4=(0.0,0.0)
	   call tensor_block_fcontract_dlf(lcd,ltp%data_cmplx4,rtp%data_cmplx4,d_c4,ierr,cmplx(alf,kind=4),cmplx(beta,kind=4))
	   if(ierr.ne.0) then; ierr=28; goto 999; endif
	   dtp%scalar_value=dtp%scalar_value+cmplx(d_c4,kind=8)
	  case('c8','C8')
	   d_c8=(0d0,0d0)
	   call tensor_block_fcontract_dlf(lcd,ltp%data_cmplx8,rtp%data_cmplx8,d_c8,ierr,alf,beta)
	   if(ierr.ne.0) then; ierr=29; goto 999; endif
	   dtp%scalar_value=dtp%scalar_value+d_c8
	  end select
	 case(ADD_TENSOR)
	  if(ltb.ne.scalar_tensor.and.rtb.eq.scalar_tensor) then
	   if(lconj) then; k=1*2+0; else; k=0; endif !bit 0 -> D; bit 1 -> L
	   if(rconj) then; d_c8=conjg(rtp%scalar_value); else; d_c8=rtp%scalar_value; endif
	   call tensor_block_add(dtp,ltp,ierr,scale_fac=d_c8*alf,arg_conj=k,data_kind=dtk,accumulative=accum)
	   if(ierr.ne.0) then; ierr=30; goto 999; endif
	  elseif(ltb.eq.scalar_tensor.and.rtb.ne.scalar_tensor) then
	   if(rconj) then; k=1*2+0; else; k=0; endif !bit 0 -> D; bit 1 -> L
	   if(lconj) then; d_c8=conjg(ltp%scalar_value); else; d_c8=ltp%scalar_value; endif
	   call tensor_block_add(dtp,rtp,ierr,scale_fac=d_c8*alf,arg_conj=k,data_kind=dtk,accumulative=accum)
	   if(ierr.ne.0) then; ierr=31; goto 999; endif
	  else
	   ierr=32; goto 999
	  endif
	 case(MULTIPLY_SCALARS)
	  if(lconj) then; l_c8=conjg(ltp%scalar_value); else; l_c8=ltp%scalar_value; endif
	  if(rconj) then; r_c8=conjg(rtp%scalar_value); else; r_c8=rtp%scalar_value; endif
	  dtp%scalar_value=dtp%scalar_value+l_c8*r_c8*alf
	 end select
	 finish_gemm=thread_wtime()
 !Transpose the matrix-result back into the output tensor:
	 if(dtransp) then
!	  write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_contract): permutation to be performed for ",i2)') 0 !debug
	  select case(dtb)
	  case(scalar_tensor)
	  case(dimension_led)
	   call tensor_block_copy(dtp,dtens,ierr,transp=do2n); if(ierr.ne.0) then; ierr=33; goto 999; endif
	  case(bricked_dense,bricked_ordered)
	   !`Future
	  case(sparse_list)
	   !`Future
	  case(compressed)
	   !`Future
	  case default
	   ierr=34; goto 999
	  end select
	 endif
	 if(DATA_KIND_SYNC) then
	  call tensor_block_sync(dtens,dtk,ierr); if(ierr.ne.0) then; ierr=35; goto 999; endif
	 endif
 !Destroy temporary tensor blocks:
999	 nullify(ltp); nullify(rtp); nullify(dtp)
	 select case(contr_case)
	 case(PARTIAL_CONTRACTION)
	  if(ltransp) then; call tensor_block_destroy(lta,j); if(j.ne.0) ierr=ierr+100+j; endif
	  if(rtransp) then; call tensor_block_destroy(rta,j); if(j.ne.0) ierr=ierr+200+j; endif
	  if(dtransp) then; call tensor_block_destroy(dta,j); if(j.ne.0) ierr=ierr+500+j; endif
	 case(FULL_CONTRACTION)
	  if(ltransp) then; call tensor_block_destroy(lta,j); if(j.ne.0) ierr=ierr+1000+j; endif
	 case(ADD_TENSOR)
	  if(dtransp) then; call tensor_block_destroy(dta,j); if(j.ne.0) ierr=ierr+2000+j; endif
	 case(MULTIPLY_SCALARS)
	 end select
	 if(LOGGING.gt.0) then
	  write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_contract): Max threads = ",i3,": GEMM time ",F10.4)')&
          &nthr,finish_gemm-start_gemm !debug
	 endif
	else
	 ierr=36
	endif
!	write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_contract): exit error code: ",i5)') ierr !debug
	return

	contains

	 subroutine calculate_matrix_dimensions(tbst,nm,ns,tens,lm,ls,ier)
	 integer, intent(in):: tbst,nm,ns !tensor block storage layout, number of minor dimensions, number of senior dimensions
	 type(tensor_block_t), intent(in):: tens !tensor block
	 integer(LONGINT), intent(out):: lm,ls !minor extent, senior extent (of the matrix)
	 integer, intent(out):: ier
	 integer j0
	 ier=0
	 if(nm.ge.0.and.ns.ge.0.and.nm+ns.eq.tens%tensor_shape%num_dim) then
	  select case(tbst)
	  case(scalar_tensor)
	   lm=1_LONGINT; ls=1_LONGINT
	  case(dimension_led)
	   lm=1_LONGINT; do j0=1,nm; lm=lm*tens%tensor_shape%dim_extent(j0); enddo
	   ls=1_LONGINT; do j0=1,ns; ls=ls*tens%tensor_shape%dim_extent(nm+j0); enddo
	  case(bricked_dense) !`Future
	  case(bricked_ordered) !`Future
	  case(sparse_list) !`Future
	  case(compressed) !`Future
	  case default
	   ier=2
	  end select
	 else
	  ier=1
	 endif
	 return
	 end subroutine calculate_matrix_dimensions

	 subroutine determine_index_permutations !sets {dtransp,ltransp,rtransp},{do2n,lo2n,ro2n},{ncd,nlu,nru}
	 integer jkey(1:max_tensor_rank),jtrn0(0:max_tensor_rank),jtrn1(0:max_tensor_rank),jj,j0,j1
 !Destination operand:
	 if(drank.gt.0) then
	  do2n(0)=+1; j1=0
	  do j0=1,lrank+rrank
	   if(contr_ptrn(j0).gt.0) then
	    j1=j1+1; do2n(j1)=contr_ptrn(j0)
	   endif
	  enddo
	  if(perm_trivial(j1,do2n)) then; dtransp=.FALSE.; else; dtransp=.TRUE.; endif
	 else
	  dtransp=.FALSE.
	 endif
 !Right tensor operand:
	 nru=0; ncd=0 !numbers of the right uncontracted and contracted dimensions
	 if(rrank.gt.0) then
	  ro2n(0)=+1; j1=0
	  do j0=1,rrank; if(contr_ptrn(lrank+j0).lt.0) then; j1=j1+1; ro2n(j0)=j1; endif; enddo; ncd=j1 !contracted dimensions
	  do j0=1,rrank; if(contr_ptrn(lrank+j0).gt.0) then; j1=j1+1; ro2n(j0)=j1; endif; enddo; nru=j1-ncd !uncontracted dimensions
	  if(perm_trivial(j1,ro2n)) then; rtransp=.FALSE.; else; rtransp=.TRUE.; endif
	 else
	  rtransp=.FALSE.
	 endif
 !Left tensor operand:
	 nlu=0 !number of the left uncontracted dimensions
	 if(lrank.gt.0) then
	  lo2n(0)=+1; j1=0
	  do j0=1,lrank; if(contr_ptrn(j0).lt.0) then; j1=j1+1; jtrn1(j1)=j0; jkey(j1)=abs(contr_ptrn(j0)); endif; enddo; ncd=j1 !contracted dimensions
	  jtrn0(0:j1)=(/+1,(jj,jj=1,j1)/); call merge_sort_key_int(j1,jkey,jtrn0)
	  do j0=1,j1; jj=jtrn0(j0); lo2n(jtrn1(jj))=j0; enddo !contracted dimensions of the left operand are aligned to the corresponding dimensions of the right operand
	  do j0=1,lrank; if(contr_ptrn(j0).gt.0) then; j1=j1+1; lo2n(j0)=j1; endif; enddo; nlu=j1-ncd !uncontracted dimensions
	  if(perm_trivial(j1,lo2n)) then; ltransp=.FALSE.; else; ltransp=.TRUE.; endif
	 else
	  ltransp=.FALSE.
	 endif
	 return
	 end subroutine determine_index_permutations

	 subroutine determine_data_kind(dtkd,ier)
	 character(2), intent(out):: dtkd
	 integer, intent(out):: ier
	 ier=0
	 select case(contr_case)
	 case(PARTIAL_CONTRACTION)
	  if(associated(ltens%data_cmplx8).and.associated(rtens%data_cmplx8).and.associated(dtens%data_cmplx8)) then
	   dtkd='c8'
	  else
	   if(associated(ltens%data_cmplx4).and.associated(rtens%data_cmplx4).and.associated(dtens%data_cmplx4)) then
	    dtkd='c4'
	   else
	    if(associated(ltens%data_real8).and.associated(rtens%data_real8).and.associated(dtens%data_real8)) then
	     dtkd='r8'
	    else
	     if(associated(ltens%data_real4).and.associated(rtens%data_real4).and.associated(dtens%data_real4)) then
	      dtkd='r4'
	     else
	      ier=101
	     endif
	    endif
	   endif
	  endif
	 case(FULL_CONTRACTION)
	  if(associated(ltens%data_cmplx8).and.associated(rtens%data_cmplx8)) then
	   dtkd='c8'
	  else
	   if(associated(ltens%data_cmplx4).and.associated(rtens%data_cmplx4)) then
	    dtkd='c4'
	   else
	    if(associated(ltens%data_real8).and.associated(rtens%data_real8)) then
	     dtkd='r8'
	    else
	     if(associated(ltens%data_real4).and.associated(rtens%data_real4)) then
	      dtkd='r4'
	     else
	      ier=102
	     endif
	    endif
	   endif
	  endif
	 case(ADD_TENSOR)
	  if(associated(ltens%data_cmplx8).or.associated(rtens%data_cmplx8)) then
	   dtkd='c8'
	  else
	   if(associated(ltens%data_cmplx4).or.associated(rtens%data_cmplx4)) then
	    dtkd='c4'
	   else
	    if(associated(ltens%data_real8).or.associated(rtens%data_real8)) then
	     dtkd='r8'
	    else
	     if(associated(ltens%data_real4).or.associated(rtens%data_real4)) then
	      dtkd='r4'
	     else
	      ier=103
	     endif
	    endif
	   endif
	  endif
	 case(MULTIPLY_SCALARS)
	  dtkd='c8' !scalars are always complex(8)
	 end select
	 return
	 end subroutine determine_data_kind

	 logical function contr_ptrn_ok(ptrn,lr,rr,dr)
	 integer, intent(in):: ptrn(1:*),lr,rr,dr
	 integer j0,j1,jl,jbus(dr+lr+rr)
	 contr_ptrn_ok=.TRUE.; jl=dr+lr+rr
	 if(jl.gt.0) then
	  jbus(1:jl)=0
	  do j0=1,lr !left tensor-argument
	   j1=ptrn(j0)
	   if(j1.gt.0.and.j1.le.dr) then !uncontracted index
	    jbus(j1)=jbus(j1)+1; jbus(dr+j0)=jbus(dr+j0)+1
	    if(ltens%tensor_shape%dim_extent(j0).ne.dtens%tensor_shape%dim_extent(j1)) then
	     contr_ptrn_ok=.FALSE.; return
	    endif
	   elseif(j1.lt.0.and.abs(j1).le.rr) then !contracted index
	    jbus(dr+lr+abs(j1))=jbus(dr+lr+abs(j1))+1
	    if(ltens%tensor_shape%dim_extent(j0).ne.rtens%tensor_shape%dim_extent(abs(j1))) then
	     contr_ptrn_ok=.FALSE.; return
	    endif
	   else
	    contr_ptrn_ok=.FALSE.; return
	   endif
	  enddo
	  do j0=lr+1,lr+rr !right tensor-argument
	   j1=ptrn(j0)
	   if(j1.gt.0.and.j1.le.dr) then !uncontracted index
	    jbus(j1)=jbus(j1)+1; jbus(dr+j0)=jbus(dr+j0)+1
	    if(rtens%tensor_shape%dim_extent(j0-lr).ne.dtens%tensor_shape%dim_extent(j1)) then
	     contr_ptrn_ok=.FALSE.; return
	    endif
	   elseif(j1.lt.0.and.abs(j1).le.lr) then !contracted index
	    jbus(dr+abs(j1))=jbus(dr+abs(j1))+1
	    if(rtens%tensor_shape%dim_extent(j0-lr).ne.ltens%tensor_shape%dim_extent(abs(j1))) then
	     contr_ptrn_ok=.FALSE.; return
	    endif
	   else
	    contr_ptrn_ok=.FALSE.; return
	   endif
	  enddo
	  do j0=1,jl
	   if(jbus(j0).ne.1) then; contr_ptrn_ok=.FALSE.; return; endif
	  enddo
	 endif
	 return
	 end function contr_ptrn_ok

	 logical function ord_rest_ok(ord,ptrn,lr,rr,dr)!`Finish
	 integer, intent(in):: ord(1:*),ptrn(1:*),lr,rr,dr
	 ord_rest_ok=.TRUE.
	 return
	 end function ord_rest_ok

	end subroutine tensor_block_contract
!----------------------------------------------------------
        complex(8) function tensor_block_scalar_value(tens) !SERIAL
!Returns the .scalar_value component of <tensor_block_t>.
         implicit none
         type(tensor_block_t), intent(in):: tens
         tensor_block_scalar_value=tens%scalar_value
         return
        end function tensor_block_scalar_value
!----------------------------------------------------------------------
	subroutine get_mlndx_addr(intyp,id1,id2,mnii,ia1,ivol,iba,ierr) !SERIAL
!This subroutine creates an addressing array IBA(index_value,index_place) containing addressing increments for multiindices.
!A multiindex is supposed to be in an ascending order for an i_multiindex and descending order for an a_multiindex
!with respect to moving from the minor multiindex positions to senior ones (left ro right): i1<i2<...<in; a1>a2>...>an,
!such that general index positions (minor) precede active index positions (senior).
!INPUT:
! - intyp - type of the multiindex {i/a}: {1/2};
! - id1   - index value upper bound: [0...id1], range=(1+id1);
! - id2   - length of the multiindex: [1..id2];
! - mnii  - maximal number of inactive indices (first <mnii> positions);
! - ia1   - first active index value for an i_multiindex, last active index value for an a_multiindex;
! - ierr - if non-zero, the validity test will be executed at the end;
!OUTPUT:
! - iba(0:id1,1:id2) - addressing array (addressing increments);
! - ivol(1:id2) - number of possible multiindices for each multiindex length less or equal to <id2>;
! - ierr - error code (0:success).
	implicit none
	integer i,j,k,l,m,n,k1,k2,ks,kf,ierr
	integer, intent(in):: intyp,id1,id2,mnii,ia1
	integer(LONGINT), intent(out):: ivol(1:id2),iba(0:id1,1:id2)
	integer nii,ibnd(1:id2,2),im(0:id2+1)
	integer(LONGINT) kv1,kv2,mz,l1
	logical validity_test

	if(ierr.eq.0) then; validity_test=.FALSE.; else; validity_test=.TRUE.; ierr=0; endif
	nii=min(id2,mnii) !actual number of inactive (leading minor) positions
!Test arguments:
	if(id1.lt.0.or.id2.lt.0.or.id2.gt.1+id1) then
	 if(VERBOSE) write(CONS_OUT,'("ERROR(tensor_algebra::get_mlndx_addr): invalid or incompatible multiindex specification: "&
	             &,i1,1x,i6,1x,i2,1x,i2,1x,i6)') intyp,id1,id2,mnii,ia1
	 ierr=1; return
	endif
	if(mnii.lt.0.or.ia1.lt.0.or.ia1.gt.id1.or.(intyp.eq.1.and.id2-nii.gt.id1-ia1+1).or.(intyp.eq.2.and.id2-nii.gt.ia1+1)) then
	 if(VERBOSE) write(CONS_OUT,'("ERROR(tensor_algebra::get_mlndx_addr): invalid active space configuration: "&
	             &,i1,1x,i6,1x,i2,1x,i2,1x,i6)') intyp,id1,id2,mnii,ia1
	 ierr=2; return
	endif
	if(id2.eq.0) return !empty multiindex
!Set the multiindex addressing increments:
	iba(:,:)=0_LONGINT; ivol(:)=0_LONGINT
	if(intyp.eq.1) then
 !i_multiindex, runnning downwards:
  !set index bounds:
	 ibnd(1:id2,1)=(/(j,j=0,nii-1),(max(nii,ia1)+j,j=0,id2-nii-1)/) !lower bounds do not depend on the multiindex length
	 ibnd(1:id2,2)=(/(j,j=id1-id2+1,id1)/) !upper bounds
  !set IBA:
	 kv2=0_LONGINT; do j=id1,ibnd(1,1),-1; iba(j,1)=kv2; kv2=kv2+1_LONGINT; enddo !the most minor position (the most rapidly changing)
	 ivol(1)=kv2; kv2=kv2-1_LONGINT
	 do k=2,id2 !loop over the index positions
	  kv1=0_LONGINT; do j=1,k-1; kv1=kv1+iba(id1-j,k-j); enddo !loop over the previous index positions
	  iba(id1,k)=-kv1 !set the first element at position k
	  do l=id1-1,ibnd(k,1),-1 !loop over the other index values
	   kv1=0_LONGINT; do j=1,k-1; kv1=kv1+iba(l-j,k-j); enddo
	   iba(l,k)=iba(l+1,k)+ivol(k-1)-kv1
	  enddo
	  kv2=kv2+iba(ibnd(k,1),k); ivol(k)=kv2+1_LONGINT
	 enddo
	elseif(intyp.eq.2) then
 !a_multiindex, runnning upwards:
  !set index bounds:
	 ibnd(1:id2,1)=(/(id2-j,j=1,id2)/) !lower bounds
	 ibnd(1:id2,2)=(/(id1-j,j=0,nii-1),(min(id1-nii,ia1)-j,j=0,id2-nii-1)/) !upper bounds do not depend on the multiindex length
  !set IBA:
	 kv2=0_LONGINT; do j=0,ibnd(1,2); iba(j,1)=kv2; kv2=kv2+1_LONGINT; enddo !the most minor position (the most rapidly changing)
	 ivol(1)=kv2; kv2=kv2-1_LONGINT
	 do k=2,id2 !loop over the index positions
	  kv1=0_LONGINT; do j=1,k-1; kv1=kv1+iba(j,k-j); enddo !loop over the previous index positions
	  iba(0,k)=-kv1 !set the first element at position k
	  do l=1,ibnd(k,2) !loop over the other index values
	   kv1=0_LONGINT; do j=1,k-1; kv1=kv1+iba(l+j,k-j); enddo
	   iba(l,k)=iba(l-1,k)+ivol(k-1)-kv1
	  enddo
	  kv2=kv2+iba(ibnd(k,2),k); ivol(k)=kv2+1_LONGINT
	 enddo
	else
	 if(VERBOSE) write(CONS_OUT,'("ERROR(tensor_algebra::get_mlndx_addr): invalid multiindex type requested: ",i2)') intyp
	 ierr=3; return
	endif
!Testing IBA for multiindices of size <id2>:
	if(validity_test) then
	 if(intyp.eq.1) then
 !i_multiindex, runnning downward:
  !set bounds:
	  ibnd(1:id2,1)=(/(j,j=0,nii-1),(max(nii,ia1)+j,j=0,id2-nii-1)/) !lower bounds do not depend on the multiindex length
	  ibnd(1:id2,2)=(/(j,j=id1-id2+1,id1)/) !upper bounds
  !init IM:
	  im(1:id2)=ibnd(1:id2,2); mz=0_LONGINT; do k=1,id2; mz=mz+iba(im(k),k); enddo; l1=0_LONGINT
	  iloop: do
!	   write(CONS_OUT,'("DEBUG(tensor_algebra::get_mlndx_addr): Curent MLNDX: ",i10,5x,128(1x,i4))') l1,im(1:id2) !debug
	   if(mz.ne.l1) then
	    if(VERBOSE) write(CONS_OUT,*)'ERROR(tensor_algebra::get_mlndx_addr): wrong i_MZ: ',l1,mz
	    ierr=4; return
	   endif
	   l1=l1+1_LONGINT
  !loop footer:
	   k=1
	   do while(k.le.id2)
	    mz=mz-iba(im(k),k)
	    if(im(k).gt.ibnd(k,1)) then
	     im(k)=im(k)-1; mz=mz+iba(im(k),k)
	     do k1=k-1,1,-1; im(k1)=min(im(k1+1)-1,ibnd(k1,2)); mz=mz+iba(im(k1),k1); enddo
	     cycle iloop
	    else
	     k=k+1
	    endif
	   enddo
!	   write(CONS_OUT,'("DEBUG(tensor_algebra::get_mlndx_addr): TESTED i-VOLUME: ",i20)') l1 !debug
	   exit iloop
	  enddo iloop
	 else !intyp=2
 !a_multiindex, runnning upward:
  !set bounds:
	  ibnd(1:id2,1)=(/(id2-j,j=1,id2)/) !lower bounds
	  ibnd(1:id2,2)=(/(id1-j,j=0,nii-1),(min(id1-nii,ia1)-j,j=0,id2-nii-1)/) !upper bounds do not depend on the multiindex length
  !init IM:
	  im(1:id2)=ibnd(1:id2,1); mz=0_LONGINT; do k=1,id2; mz=mz+iba(im(k),k); enddo; l1=0_LONGINT
	  aloop: do
!	   write(CONS_OUT,'("DEBUG(tensor_algebra::get_mlndx_addr): Curent MLNDX: ",i10,5x,128(1x,i4))') im(id2:1:-1) !debug
	   if(mz.ne.l1) then
	    if(VERBOSE) write(CONS_OUT,*)'ERROR(tensor_algebra::get_mlndx_addr): wrong a_MZ: ',l1,mz
	    ierr=5; return
	   endif
	   l1=l1+1_LONGINT
  !loop footer:
	   k=1
	   do while(k.le.id2)
	    mz=mz-iba(im(k),k)
	    if(im(k).lt.ibnd(k,2)) then
	     im(k)=im(k)+1; mz=mz+iba(im(k),k)
	     do k1=k-1,1,-1; im(k1)=max(im(k1+1)+1,ibnd(k1,1)); mz=mz+iba(im(k1),k1); enddo
	     cycle aloop
	    else
	     k=k+1
	    endif
	   enddo
!	   write(CONS_OUT,'("DEBUG(tensor_algebra::get_mlndx_addr): TESTED a-VOLUME: ",i20)') l1 !debug
	   exit aloop
	  enddo aloop
	 endif
	endif !validity test
	return
	end subroutine get_mlndx_addr
!-------------------------------------------------------
	integer(LONGINT) function mlndx_value(ml,im,iba) !SERIAL
!This function returns an address associated with the given multiindex.
!Each index is greater or equal to zero. Index position numeration starts from 1.
!INPUT:
! - ml - multiindex length;
! - im(1:ml) - multiindex;
! - iba(0:,1:) - array with addressing increments (generated by <get_mlndx_addr>);
!OUTPUT:
! - mlndx_value - address, -1 if ml<0;
	implicit none
	integer, intent(in):: ml,im(1:ml)
	integer(LONGINT), intent(in):: iba(0:,1:)
	integer i,j,k,l,m,n,k0,k1,k2,k3,ks,kf,ierr
	if(ml.ge.0) then
	 mlndx_value=0_LONGINT; ks=mod(ml,2)
	 do i=1,ml-ks,2; mlndx_value=mlndx_value+iba(im(i),i)+iba(im(i+1),i+1); enddo
	 if(ks.ne.0) mlndx_value=mlndx_value+iba(im(ml),ml)
	else
	 mlndx_value=-1_LONGINT
	endif
	return
	end function mlndx_value
!-------------------------------------------------------------------
	subroutine tensor_shape_rnd(tsss,tsl,ierr,tsize,tdim,spread) !SERIAL
!This subroutine returns a random tensor shape specification string.
!Only simple dense tensor blocks (without any ordering) are concerned here.
!INPUT:
! - tsize - (optional) desirable size of the tensor block (approximate);
! - tdim - (optional) desirable rank of the tensor block (exact);
! - spread - (optional) desirable spread in dimension extents (ratio max_dim_ext/min_dim_ext);
!OUTPUT:
! - tsss(1:tsl) - tensor shape specification string.
	implicit none
	character(*), intent(out):: tsss
	integer, intent(out):: tsl
	integer(LONGINT), intent(in), optional:: tsize
	integer, intent(in), optional:: tdim
	integer, intent(in), optional:: spread
	integer, intent(inout):: ierr
!-------------------------------------------------------
	integer(LONGINT), parameter:: max_blk_size=2**30 !max tensor block size (default)
!-------------------------------------------------------
	integer i,j,m,n,tdm,spr
	integer(LONGINT) tsz
	real(8) val,stretch,dme(1:max_tensor_rank)

	ierr=0; tsl=0
	if(present(tdim)) then
	 if(present(tsize)) then; if(tdim.eq.0.and.tsize.ne.1_LONGINT) then; ierr=1; return; endif; endif
	 if(tdim.ge.0) then; tdm=tdim; else; ierr=2; return; endif
	else
	 call random_number(val); tdm=nint(dble(max_tensor_rank)*val)
	endif
	if(present(tsize)) then
	 if(tsize.gt.0_LONGINT) then; tsz=tsize; if(tdm.eq.0.and.tsize.gt.1_LONGINT) tdm=1; else; ierr=3; return; endif
	else
	 if(tdm.gt.0) then; call random_number(val); tsz=int(dble(max_blk_size)*val,LONGINT)+1_LONGINT; else; tsz=1_LONGINT; endif
	endif
	if(present(spread)) then
	 if(spread.ge.1) then; spr=spread; else; ierr=4; return; endif
	 if(present(tsize)) then; if(tsize.lt.int(spread,LONGINT)) then; ierr=5; return; endif; endif
	 tsz=max(tsz,int(spr,8))
	else
	 spr=0
	endif
	if(tdm.gt.0) then
	 tsss(1:1)='('; tsl=1
	 call random_number(dme(1:tdm))
	 val=dme(1); do i=2,tdm; val=min(val,dme(i)); enddo
	 do i=1,tdm; dme(i)=dme(i)/val; enddo
!	 write(CONS_OUT,*) dme(1:tdm) !debug
	 val=dme(1); do i=2,tdm; val=max(val,dme(i)); enddo
	 if(spr.ge.1) then; stretch=dlog10(dble(spr))/dlog10(val); do i=1,tdm; dme(i)=dme(i)**stretch; enddo; endif
!	 write(CONS_OUT,*) stretch; write(CONS_OUT,*) dme(1:tdm) !debug
	 val=dme(1); do i=2,tdm; val=val*dme(i); enddo
	 stretch=(dble(tsz)/val)**(1d0/dble(tdm))
	 do i=1,tdm
	  val=dme(i)*stretch
	  if(val.ge.2d0) then
	   m=nint(val); if(i.lt.tdm) stretch=stretch*((val/dble(m))**(1d0/dble(tdm-i)))
	  else
	   m=2; if(i.lt.tdm) stretch=stretch*(val**(1d0/dble(tdm-i)))
	  endif
	  if(m.lt.2) then; ierr=6; return; endif
	  call numchar(m,j,tsss(tsl+1:)); tsl=tsl+j+1; tsss(tsl:tsl)=','
	 enddo
	 tsss(tsl:tsl)=')'
	else
	 tsl=len_trim('()'); tsss(1:tsl)='()'
	endif
	return
	end subroutine tensor_shape_rnd
!---------------------------------------------------------------------
        integer function tensor_shape_rank(tsss,ierr,dim_ext,tens_vol) !SERIAL
!This function returns the number of dimensions in a tensor shape specification string (TSSS).
        implicit none
        character(*), intent(in):: tsss                 !in: tensor shape specificaton string
        integer, intent(inout):: ierr                   !out: error code
        integer, intent(inout), optional:: dim_ext(1:*) !out: tensor dimension extents
        integer(8), intent(out), optional:: tens_vol    !out: tensor volume (total number of tensor elements)
        integer i,j,l,dme(1:MAX_TENSOR_RANK)

        ierr=0; tensor_shape_rank=0
        if(tsss(1:1).eq.'(') then
         l=1; do while(l.le.MAX_SHAPE_STR_LEN); if(tsss(l:l).eq.')') exit; l=l+1; enddo
         if(l.le.MAX_SHAPE_STR_LEN) then
          if(l.gt.2) then
           i=1
           do while(tsss(i:i).ne.')')
            i=i+1; j=0
            do while(iachar(tsss(i+j:i+j)).ge.iachar('0').and.iachar(tsss(i+j:i+j)).le.iachar('9')); j=j+1; enddo
            if(j.gt.0) then; tensor_shape_rank=tensor_shape_rank+1; i=i+j; else; ierr=1; return; endif
            dme(tensor_shape_rank)=icharnum(j,tsss(i:i+j-1))
           enddo
          endif
          if(present(dim_ext)) dim_ext(1:tensor_shape_rank)=dme(1:tensor_shape_rank)
          if(present(tens_vol)) then
           tens_vol=1_8; do i=1,tensor_shape_rank; tens_vol=tens_vol*dme(i); enddo
          endif
         else
          ierr=2
         endif
        else
         ierr=3
        endif
        return
        end function tensor_shape_rank
!-----------------------------------------------------------------------------
        subroutine tensor_shape_str_create(trank,dims,tsss,tsl,ierr,divs,grps) !SERIAL
!This subroutine creates a tensor shape specification string from {dims;divs;grps}.
!INPUT:
! - trank - tensor block rank;
! - dims(1:trank) - dimension extents;
! - divs(1:trank) - (optional) dimension extent dividers;
! - grps(1:trank) - (optional) dimension groups;
!OUTPUT:
! - tsss(1:tsl) - tensor shape specification string (see <tensor_block_shape_create> below);
! - ierr - error code (0:success).
        implicit none
        integer, intent(in):: trank
        integer, intent(in):: dims(1:*)
        integer, intent(in), optional:: divs(1:*)
        integer, intent(in), optional:: grps(1:*)
        character(*), intent(inout):: tsss
        integer, intent(out):: tsl
        integer:: ierr
        integer i,j
        logical fdivs,fgrps

        ierr=0; tsl=0
        if(trank.eq.0) then !scalar
         tsss(1:2)='()'; tsl=2
        elseif(trank.gt.0) then !tensor
         fdivs=present(divs); fgrps=present(grps)
         tsss(1:1)='('; tsl=1
         do i=1,trank
          if(dims(i).gt.0) then
           call numchar(dims(i),j,tsss(tsl+1:)); tsl=tsl+j
           if(fdivs) then
            if(divs(i).gt.0.and.divs(i).le.dims(i)) then
             if(divs(i).lt.dims(i)) then !non-trivial
              tsl=tsl+1; tsss(tsl:tsl)='/'
              call numchar(divs(i),j,tsss(tsl+1:)); tsl=tsl+j
             endif
            else
             ierr=1; return
            endif
           endif
           if(fgrps) then
            if(grps(i).ge.0) then
             if(grps(i).gt.0) then !non-trivial
              tsl=tsl+1; tsss(tsl:tsl)='{'
              call numchar(grps(i),j,tsss(tsl+1:)); tsl=tsl+j
              tsl=tsl+1; tsss(tsl:tsl)='}'
             endif
            else
             ierr=2; return
            endif
           endif
           tsl=tsl+1; tsss(tsl:tsl)=','
          else
           ierr=3; return
          endif
         enddo
         if(tsss(tsl:tsl).eq.'(') tsl=tsl+1
         tsss(tsl:tsl)=')'
        else
         ierr=4
        endif
        return
        end subroutine tensor_shape_str_create
!------------------------------------------------------------------------
	subroutine get_contr_pattern(cptrn,contr_ptrn,cpl,ierr,conj_bits) !SERIAL
!This subroutine converts a mnemonic contraction pattern into the digital form.
!INPUT:
! - cptrn - mnemonic contraction pattern (e.g., "D(ia1,ib2)+=L(ib2,k2,c3)*R(c3,ia1,k2)" );
!OUTPUT:
! - contr_ptrn(1:cpl) - digital contraction pattern (see, tensor_block_contract);
! - ierr - error code (0:success);
! - conj_bits - (optional) conjugation bits: bit 0 for D, bit 1 for L, and bit 2 for R;
!NOTES:
! - Index labels may only contain English letters and/or numbers.
!   Indices are separated by commas. Parentheses are mandatory.
! - ASCII is assumed.
	implicit none
	character(*), intent(in):: cptrn
	integer, intent(out):: contr_ptrn(1:*),cpl
	integer, intent(inout):: ierr
	integer, intent(out), optional:: conj_bits
	character(1), parameter:: dn1(0:9)=(/'0','1','2','3','4','5','6','7','8','9'/)
	character(2), parameter:: dn2(0:49)=(/'00','01','02','03','04','05','06','07','08','09',&
	                                     &'10','11','12','13','14','15','16','17','18','19',&
	                                     &'20','21','22','23','24','25','26','27','28','29',&
	                                     &'30','31','32','33','34','35','36','37','38','39',&
	                                     &'40','41','42','43','44','45','46','47','48','49'/)
	integer i,j,k,l,m,n,k0,k1,k2,k3,k4,k5,ks,kf,adims(0:2),tag_len,conj
	character(2048) str !increase the length if needed (I doubt)

	ierr=0; l=len_trim(cptrn); cpl=0; conj=0
	if(l.gt.0) then
!	 write(CONS_OUT,*)'DEBUG(tensor_algebra::get_contr_pattern): '//cptrn(1:l) !debug
!Extract the index labels:
	 tag_len=len('{000}')
	 adims(:)=0; n=-1; m=0; ks=0; i=1
	 aloop: do while(i.le.l)
	  do while(cptrn(i:i).ne.'('); i=i+1; if(i.gt.l) exit aloop; enddo
	  if(i.gt.2) then
	   if(cptrn(i-1:i-1).eq.'+'.and.alphanumeric(cptrn(i-2:i-2))) conj=conj+(2**(n+1)) !tensor complex conjugation mark
	  endif
	  ks=i; i=i+1; n=n+1; k=1 !find '(': beginning of an argument #n
	  if(n.gt.2) then; ierr=1; return; endif
	  str(m+1:m+tag_len)='{'//dn1(n)//dn2(k)//'}'; m=m+tag_len
	  do while(i.le.l)
	   if(cptrn(i:i).eq.',') then
	    j=i-1-abs(ks); if(.not.index_label_ok(cptrn(abs(ks)+1:i-1))) then; ierr=2; return; endif
	    str(m+1:m+j)=cptrn(abs(ks)+1:i-1); m=m+j; ks=-i
	    k=k+1; str(m+1:m+tag_len)='{'//dn1(n)//dn2(k)//'}'; m=m+tag_len
	   elseif(cptrn(i:i).eq.')') then
	    j=i-1-abs(ks)
	    if(j.le.0) then
	     if(ks.lt.0) then; ierr=3; return; endif
	    else
	     if(.not.index_label_ok(cptrn(abs(ks)+1:i-1))) then; ierr=4; return; endif
	     str(m+1:m+j)=cptrn(abs(ks)+1:i-1); m=m+j
	     k=k+1; str(m+1:m+tag_len)='{'//dn1(n)//dn2(k)//'}'; m=m+tag_len
	    endif
	    ks=0; i=i+1; exit
	   endif
	   i=i+1
	  enddo
	  m=m-tag_len; if(ks.ne.0) then; ierr=5; return; endif !no closing parenthesis
	  adims(n)=k-1 !number of indices in argument #k
	 enddo aloop
	 str(m+1:m+1)='{' !special setting
	 if(ks.ne.0) then; ierr=6; return; endif !no closing parenthesis
	 if(n.eq.2) then !contraction (three arguments)
 !Analyze the index labels:
	  cpl=adims(1)+adims(2)
!	  write(CONS_OUT,*)'DEBUG(tensor_algebra::get_contr_pattern): str: '//str(1:m+1) !debug
	  i=0
	  do
	   j=index(str(i+1:m),'}')+i; if(j.gt.i) then; i=j; else; exit; endif
	   j=i+1; do while(j.le.m-tag_len); if(str(j:j).eq.'{') exit; j=j+1; enddo
	   if(str(j:j).ne.'{'.or.j.gt.m-tag_len) then; ierr=7; return; endif
	   k0=index(str(j+tag_len-1:m+1),str(i:j))+(j+tag_len-2)
	   if(k0.gt.j+tag_len-2) then
!	    write(CONS_OUT,*)'DEBUG(tensor_algebra::get_contr_pattern): index match: '//str(i+1:j-1) !debug
	    k5=1; k1=icharnum(k5,str(i-tag_len+2:i-tag_len+2))
	    k5=2; k2=icharnum(k5,str(i-tag_len+3:i-tag_len+4))
	    k5=1; k3=icharnum(k5,str(k0-tag_len+2:k0-tag_len+2))
	    k5=2; k4=icharnum(k5,str(k0-tag_len+3:k0-tag_len+4))
	    if(k1.eq.0.and.k3.eq.1) then !open index
	     contr_ptrn(k4)=k2
	    elseif(k1.eq.0.and.k3.eq.2) then !open index
	     contr_ptrn(adims(1)+k4)=k2
	    elseif(k1.eq.1.and.k3.eq.2) then !free index
	     contr_ptrn(k2)=-k4; contr_ptrn(adims(1)+k4)=-k2
	    else
	     ierr=8; return
	    endif
	    str(i+1:j-1)=' '; do while(str(k0:k0).ne.'{'); str(k0:k0)=' '; k0=k0+1; enddo
	   else
	    ierr=9; return
	   endif
	  enddo
	 elseif(n.eq.1) then !permutation (two arguments)
	  !`Add
	  ierr=10
	 endif
	else !empty string
	 ierr=11
	endif
	if(present(conj_bits)) conj_bits=conj
!	write(CONS_OUT,*)'DEBUG(tensor_algebra::get_contr_pattern): cpl,contr_ptrn: ',cpl,contr_ptrn(1:cpl) !debug
	return

	contains

	 logical function index_label_ok(lb)
	  character(*), intent(in):: lb
	  integer j0,j1,j2
	  j0=len(lb); index_label_ok=.TRUE.
	  if(j0.gt.0) then
	   do j1=1,j0
	    j2=iachar(lb(j1:j1))
	    if(.not.((j2.ge.iachar('a').and.j2.le.iachar('z')).or.&
	            &(j2.ge.iachar('A').and.j2.le.iachar('Z')).or.(j2.ge.iachar('0').and.j2.le.iachar('9')))) then
	     index_label_ok=.FALSE.; return
	    endif
	   enddo
!	   j2=iachar(lb(1:1)); if(j2.ge.iachar('0').and.j2.le.iachar('9')) index_label_ok=.FALSE. !the 1st character cannot be a number
	  else
	   index_label_ok=.FALSE.
	  endif
	  return
	 end function index_label_ok

	end subroutine get_contr_pattern
!-----------------------------------------------------------------------------------------------------
        subroutine get_contr_pattern_sym(rank_left,rank_right,conj_bits,cptrn_dig,cptrn_sym,cpl,ierr)&
        &bind(c,name='get_contr_pattern_sym') !SERIAL
!Converts a digital tensor contraction pattern into a symbolic form.
        implicit none
        integer(C_INT), intent(in):: rank_left                         !in: rank of the left tensor
        integer(C_INT), intent(in):: rank_right                        !in: rank of the right tensor
        integer(C_INT), intent(in):: conj_bits                         !in: argument conjugation bits: {0:D,1:L,2:R}
        integer(C_INT), intent(in):: cptrn_dig(1:rank_left+rank_right) !in: digital contraction pattern
        character(C_CHAR), intent(inout):: cptrn_sym(1:*)              !out: symbolic contraction pattern
        integer(C_INT), intent(out):: cpl                              !out: length of <cptrn_sym>
        integer(C_INT), intent(out):: ierr                             !out: error code
        integer(C_INT):: i,m,nu
        character(C_CHAR):: left_lbl(1:MAX_TENSOR_RANK)

        ierr=0; cpl=0
        if(rank_left.ge.0.and.rank_right.ge.0) then
         if(rank_left+rank_right.gt.0) then
!Count uncontracted indices:
          nu=0; do i=1,rank_left+rank_right; if(cptrn_dig(i).gt.0) nu=nu+1; enddo
!Print the destination tensor:
          if(iand(conj_bits,1_C_INT).eq.0) then !no conjugation
           cptrn_sym(1:len_trim('D('))=(/'D','('/); cpl=cpl+len_trim('D(')
          else !conjugation
           cptrn_sym(1:len_trim('D+('))=(/'D','+','('/); cpl=cpl+len_trim('D+(')
          endif
          m=iachar('a'); do i=1,nu; cptrn_sym(cpl+1:cpl+2)=(/achar(m),','/); cpl=cpl+2; m=m+1; enddo
          if(cptrn_sym(cpl).eq.'(') cpl=cpl+1; cptrn_sym(cpl)=')'
!Print the left tensor:
          if(iand(conj_bits,2_C_INT).eq.0) then !no conjugation
           cptrn_sym(cpl+1:cpl+len_trim('+=L('))=(/'+','=','L','('/); cpl=cpl+len_trim('+=L(')
          else !conjugation
           cptrn_sym(cpl+1:cpl+len_trim('+=L+('))=(/'+','=','L','+','('/); cpl=cpl+len_trim('+=L+(')
          endif
          do i=1,rank_left
           if(cptrn_dig(i).gt.0) then !uncontracted index
            cptrn_sym(cpl+1:cpl+2)=(/achar(iachar('a')+cptrn_dig(i)-1),','/); cpl=cpl+2
           else !contracted index
            left_lbl(i)=achar(m)
            cptrn_sym(cpl+1:cpl+2)=(/achar(m),','/); cpl=cpl+2; m=m+1
           endif
          enddo
          if(cptrn_sym(cpl).eq.'(') cpl=cpl+1; cptrn_sym(cpl)=')'
!Print the right tensor:
          if(iand(conj_bits,4_C_INT).eq.0) then !no conjugation
           cptrn_sym(cpl+1:cpl+len_trim('*R('))=(/'*','R','('/); cpl=cpl+len_trim('*R(')
          else !conjugation
           cptrn_sym(cpl+1:cpl+len_trim('*R+('))=(/'*','R','+','('/); cpl=cpl+len_trim('*R+(')
          endif
          do i=1,rank_right
           if(cptrn_dig(rank_left+i).gt.0) then !uncontracted index
            cptrn_sym(cpl+1:cpl+2)=(/achar(iachar('a')+cptrn_dig(rank_left+i)-1),','/); cpl=cpl+2
           else !contracted index
            cptrn_sym(cpl+1:cpl+2)=(/left_lbl(-cptrn_dig(rank_left+i)),','/); cpl=cpl+2
           endif
          enddo
          if(cptrn_sym(cpl).eq.'(') cpl=cpl+1; cptrn_sym(cpl)=')'; cptrn_sym(cpl+1)=achar(0)
         else
          if(iand(conj_bits,1_C_INT).eq.0) then !no conjugation
           cptrn_sym(cpl+1:cpl+len_trim('D()'))=(/'D','(',')'/); cpl=cpl+len_trim('D()')
          else !conjugation
           cptrn_sym(cpl+1:cpl+len_trim('D+()'))=(/'D','+','(',')'/); cpl=cpl+len_trim('D+()')
          endif
          if(iand(conj_bits,2_C_INT).eq.0) then !no conjugation
           cptrn_sym(cpl+1:cpl+len_trim('+=L()'))=(/'+','=','L','(',')'/); cpl=cpl+len_trim('+=L()')
          else !conjugation
           cptrn_sym(cpl+1:cpl+len_trim('+=L+()'))=(/'+','=','L','+','(',')'/); cpl=cpl+len_trim('+=L+()')
          endif
          if(iand(conj_bits,4_C_INT).eq.0) then !no conjugation
           cptrn_sym(cpl+1:cpl+len_trim('*R()'))=(/'*','R','(',')'/); cpl=cpl+len_trim('*R()')
          else !conjugation
           cptrn_sym(cpl+1:cpl+len_trim('*R+()'))=(/'*','R','+','(',')'/); cpl=cpl+len_trim('*R+()')
          endif
          cptrn_sym(cpl+1)=achar(0)
         endif
        else
         ierr=1
        endif
        return
        end subroutine get_contr_pattern_sym
!------------------------------------------------------------------------------------------------------
	subroutine get_contr_permutations(lrank,rrank,cptrn,conj_bits,dprm,lprm,rprm,ncd,nlu,nru,ierr)&
                                         &bind(c,name='get_contr_permutations') !SERIAL
!This subroutine returns all tensor index permutations necessary for the tensor
!contraction specified by <cptrn> (implemented via a matrix multiplication).
!INPUT:
! - cptrn(1:lrank+rrank) - digital contraction pattern;
! - conj_bits - complex conjugation bits {0:D,1:L,2:R};
!OUTPUT:
! - dprm(0:drank) - signed index permutation for the destination tensor (N2O, numeration starts from 1);
! - lprm(0:lrank) - signed index permutation for the left tensor argument (O2N, numeration starts from 1);
! - rprm(0:rrank) - signed index permutation for the right tensor argument (O2N, numeration starts from 1);
! - ncd - total number of contracted indices;
! - nlu - number of left uncontracted indices;
! - nru - number of right uncontracted indices;
! - ierr - error code (0:success).
	implicit none
!------------------------------------------------
	logical, parameter:: check_pattern=.TRUE.
!------------------------------------------------
	integer(C_INT), intent(in), value:: lrank,rrank
	integer(C_INT), intent(in):: cptrn(1:*)
	integer(C_INT), intent(in), value:: conj_bits
	integer(C_INT), intent(out):: dprm(0:*),lprm(0:*),rprm(0:*),ncd,nlu,nru
	integer(C_INT), intent(inout):: ierr
	integer(C_INT):: i,j,k,drank,jkey(1:lrank+rrank),jtrn0(0:lrank+rrank),jtrn1(0:lrank+rrank)
	logical:: pattern_ok,simple,left_conj,right_conj

	ierr=0
	if(check_pattern) then; pattern_ok=contr_pattern_ok(); else; pattern_ok=.TRUE.; endif
	if(pattern_ok.and.lrank.ge.0.and.rrank.ge.0) then
 !Check conjugation bits:
         left_conj=btest(conj_bits,1)
         right_conj=btest(conj_bits,2)
         if(btest(conj_bits,0)) then
          left_conj=.not.left_conj; right_conj=.not.right_conj
         endif
 !Destination operand:
	 drank=0; dprm(0)=+1
	 do i=1,lrank; if(cptrn(i).gt.0) then; drank=drank+1; dprm(drank)=cptrn(i); endif; enddo
	 nlu=drank; simple=(drank.ge.2) !number of the left uncontracted dimensions
	 do i=lrank+1,lrank+rrank
	  if(cptrn(i).gt.0) then
	   drank=drank+1; dprm(drank)=cptrn(i)
	   if(cptrn(i).le.nlu) simple=.FALSE.
	  endif
	 enddo
	 if(simple) dprm(1:drank)=(/(j,j=1,drank)/)
 !Right tensor operand:
	 rprm(0)=+1; nru=0; ncd=0 !numbers of the right uncontracted and contracted dimensions
	 if(rrank.gt.0) then
	  j=0; do i=1,rrank; if(cptrn(lrank+i).lt.0) then; j=j+1; rprm(i)=j; endif; enddo; ncd=j !contracted dimensions
	  nru=rrank-ncd !uncontracted dimensions
	  if(simple.and.nru.ge.2) then
	   do i=1,rrank; if(cptrn(lrank+i).gt.0) rprm(i)=ncd+(cptrn(lrank+i)-nlu); enddo
	  else
	   do i=1,rrank; if(cptrn(lrank+i).gt.0) then; j=j+1; rprm(i)=j; endif; enddo
	  endif
	 endif
 !Left tensor operand:
	 lprm(0)=+1
	 if(lrank.gt.0) then
	  j=0; do i=1,lrank; if(cptrn(i).lt.0) then; j=j+1; jtrn1(j)=i; jkey(j)=abs(cptrn(i)); endif; enddo
	  jtrn0(0:j)=(/+1,(k,k=1,j)/); if(j.ge.2) call merge_sort_key_int(j,jkey,jtrn0)
	  do i=1,j; k=jtrn0(i); lprm(jtrn1(k))=i; enddo !contracted dimensions of the left operand are aligned to the corresponding dimensions of the right operand
	  if(simple.and.nlu.ge.2) then
	   do i=1,lrank; if(cptrn(i).gt.0) lprm(i)=ncd+cptrn(i); enddo
	  else
	   do i=1,lrank; if(cptrn(i).gt.0) then; j=j+1; lprm(i)=j; endif; enddo
	  endif
	 endif
 !Apply conjugation if needed (swap contracted and uncontracted positions):
         if(left_conj.and..FALSE.) then !left argument is already processed as a transposed matrix
          do i=1,lrank
           if(lprm(i).le.ncd) then
            lprm(i)=lprm(i)+nlu
           else
            lprm(i)=lprm(i)-ncd
           endif
          enddo
         endif
         if(right_conj) then
          do i=1,rrank
           if(rprm(i).le.ncd) then
            rprm(i)=rprm(i)+nru
           else
            rprm(i)=rprm(i)-ncd
           endif
          enddo
         endif
	else !invalid lrank or rrank or cptrn(:)
	 ierr=1
	endif
	return

	contains

	 logical function contr_pattern_ok()
	 integer(C_INT):: j0,j1,jc,jl

	 contr_pattern_ok=.TRUE.; jl=lrank+rrank
	 if(jl.gt.0) then
	  jkey(1:jl)=0; jc=0
	  do j0=1,jl
	   j1=cptrn(j0)
	   if(j1.lt.0) then !contracted index
	    if(j0.le.lrank) then
	     if(abs(j1).gt.rrank) then; contr_pattern_ok=.FALSE.; return; endif
	     if(cptrn(lrank+abs(j1)).ne.-j0) then; contr_pattern_ok=.FALSE.; return; endif
	    else
	     if(abs(j1).gt.lrank) then; contr_pattern_ok=.FALSE.; return; endif
	     if(cptrn(abs(j1)).ne.-(j0-lrank)) then; contr_pattern_ok=.FALSE.; return; endif
	    endif
	   elseif(j1.gt.0.and.j1.le.jl) then !uncontracted index
	    jc=jc+1
	    if(jkey(j1).eq.0) then
	     jkey(j1)=1
	    else
	     contr_pattern_ok=.FALSE.; return
	    endif
	   else
	    contr_pattern_ok=.FALSE.; return
	   endif
	  enddo
	  do j0=1,jc; if(jkey(j0).ne.1) then; contr_pattern_ok=.FALSE.; return; endif; enddo
	 endif
	 return
	 end function contr_pattern_ok

	end subroutine get_contr_permutations
!--------------------------------------------------------------------------------------------------------
        subroutine contr_pattern_rnd(max_tens_arg_rank,max_tens_arg_size,shape0,shape1,shape2,cptrn,ierr) !SERIAL
!This subroutine returns a random tensor contraction pattern.
!INPUT:
! - max_tens_arg_rank - max tensor rank allowed (for all three tensors);
! - max_tens_arg_size - max tensor volume allowed (for all three tensors);
!OUTPUT:
! - shape0 - shape string for the tensor-result: '(..,..,..)';
! - shape1 - shape string for the left tensor argument: '(..,..,..)';
! - shape2 - shape string for the right tensor argument: '(..,..,..)';
! - cptrn(1:lrank+rrank) - contraction pattern.
        implicit none
        integer, intent(in):: max_tens_arg_rank
        integer(8), intent(in):: max_tens_arg_size
        character(*), intent(out):: shape0,shape1,shape2
        integer, intent(out):: cptrn(1:*)
        integer, intent(inout):: ierr
        integer i,j,k,l,m,n
        integer ncd,vcd,s0,s1,s2,drank,lrank,rrank
        integer lprm(0:max_tensor_rank),rprm(0:max_tensor_rank),pos(0:max_tensor_rank)
        real(8) val

        ierr=0
!Tensor ranks:
        call random_number(val); lrank=nint(val*dble(max_tens_arg_rank)); if(lrank.gt.max_tens_arg_rank) lrank=max_tens_arg_rank
        call random_number(val); rrank=nint(val*dble(max_tens_arg_rank)); if(rrank.gt.max_tens_arg_rank) rrank=max_tens_arg_rank
        if(lrank.gt.0.and.rrank.gt.0) then !get the number of contracted indices
         m=min(lrank,rrank); call random_number(val); ncd=nint(val*dble(m)); if(ncd.gt.m) ncd=m
        else
         ncd=0
        endif
        drank=lrank+rrank-ncd*2
        do while(drank.gt.max_tens_arg_rank); ncd=ncd+1; drank=drank-2; enddo
        cptrn(1:lrank+rrank)=0
!Left tensor shape and contracted indices:
        vcd=1 !total volume of contracted dimensions
        if(lrank.gt.0) then
         call random_number(val); n=nint(val*dble(max_tens_arg_size)) !desirable size of the left tensor argument
         call tensor_shape_rnd(shape1,s1,j,int(n,LONGINT),lrank); if(j.ne.0) then; ierr=1; return; endif
         if(ncd.gt.0) then
          l=len('(')+1; pos(0)=1; m=0
          do while(shape1(l:l).ne.')'); if(shape1(l:l).eq.',') then; m=m+1; pos(m)=l; endif; l=l+1; enddo; m=m+1; pos(m)=l
          if(m.eq.lrank) then
           call random_permutation(lrank,lprm) !the first ncd numbers will specify positions of contracted indices
           do i=1,ncd
            j=pos(lprm(i))-(pos(lprm(i)-1)+1); m=icharnum(j,shape1(pos(lprm(i)-1)+1:pos(lprm(i))-1))
            if(j.le.0) then; ierr=2; return; endif
            vcd=vcd*m
           enddo
          else
           ierr=3; return
          endif
         endif
        else
         shape1(1:2)='()'; s1=2
        endif
!Right tensor shape:
        if(rrank.gt.0) then
         if(ncd.lt.rrank) then
          call random_number(val); n=int(val*(dble(max_tens_arg_size)/dble(vcd)))+1
!          write(CONS_OUT,*)'DEBUG(tensor_algebra::contr_pattern_rnd): ',drank,lrank,rrank,ncd,vcd,n !debug
          call tensor_shape_rnd(shape0,s0,j,int(n,LONGINT),rrank-ncd)
          if(j.ne.0) then
           if(VERBOSE) write(CONS_OUT,'("ERROR(tensor_algebra::contr_pattern_rnd): random shape generation failed: ",i11)') j
           ierr=4; return
          endif
         else
          shape0='()'; s0=2
         endif
         if(ncd.gt.0) then
          call random_permutation(rrank,rprm) !the first ncd numbers will specify positions of contracted indices
          call merge_sort_int(ncd,rprm)
!          write(CONS_OUT,'("DEBUG(tensor_algebra::contr_pattern_rnd): R-contracted: ",64(1x,i2))') rprm(1:ncd) !debug
          shape2(1:1)='('; s2=len('('); rprm(0)=0; m=1; l=len('(')+1
          do i=1,rrank
           if(m.gt.ncd) m=0
           if(i.eq.rprm(m)) then !insert a contracted index #m
            j=pos(lprm(m))-(pos(lprm(m)-1)+1) !take the contracted dimension from the left tensor argument
            shape2(s2+1:s2+j+1)=shape1(pos(lprm(m)-1)+1:pos(lprm(m))-1)//','; s2=s2+j+1
            cptrn(lprm(m))=-rprm(m); cptrn(lrank+rprm(m))=-lprm(m)
            m=m+1
           else !insert an uncontracted dimension
            do while(shape0(l:l).ne.','.and.shape0(l:l).ne.')'); s2=s2+1; shape2(s2:s2)=shape0(l:l); l=l+1; enddo
            s2=s2+1; shape2(s2:s2)=','; l=l+1
           endif
          enddo
          shape2(s2:s2)=')'
         else
          s2=s0; shape2(1:s2)=shape0(1:s0)
         endif
        else
         shape2(1:2)='()'; s2=2
        endif
!Destination tensor shape:
        if(drank.gt.0) then
         m=0; l=len('(')+1
         do i=1,lrank
          j=0; do while(shape1(l+j:l+j).ne.','.and.shape1(l+j:l+j).ne.')'); j=j+1; enddo
          if(j.le.0) then; ierr=5; return; endif
          if(cptrn(i).eq.0) then; m=m+1; pos(m)=i; lprm(m)=icharnum(j,shape1(l:l+j-1)); endif
          l=l+j+1
         enddo
         if(m.ne.lrank-ncd) then; ierr=6; return; endif
         l=len('(')+1
         do i=1,rrank
          j=0; do while(shape2(l+j:l+j).ne.','.and.shape2(l+j:l+j).ne.')'); j=j+1; enddo
          if(j.le.0) then; ierr=7; return; endif
          if(cptrn(lrank+i).eq.0) then; m=m+1; pos(m)=lrank+i; lprm(m)=icharnum(j,shape2(l:l+j-1)); endif
          l=l+j+1
         enddo
         if(m.ne.drank) then; ierr=8; return; endif
         call random_permutation(drank,rprm)
         shape0(1:1)='('; s0=len('(')+1
         do i=1,drank
          cptrn(pos(rprm(i)))=i; call numchar(lprm(rprm(i)),j,shape0(s0:)); s0=s0+j; shape0(s0:s0)=','; s0=s0+1
         enddo
         s0=s0-1; shape0(s0:s0)=')'
        else
         shape0(1:2)='()'; s0=2
        endif
        return
        end subroutine contr_pattern_rnd
!----------------------------------------------------------------------------------------------------------
        function coherence_control_var(nargs,coh_str) result(coh_ctrl) bind(c,name='coherence_control_var')
!Given a mnemonic description of the coherence control, returns an integer
!that can be used in tensor operations for specifying it. A negative return
!value indicates an error (invalid arguments).
        implicit none
        integer(C_INT):: coh_ctrl                    !out: coherence control variable (-:error)
        integer(C_INT), intent(in):: nargs           !in: number of arguments (>0)
        character(C_CHAR), intent(in):: coh_str(1:*) !in: coherence letter for each argument (dest arg, 1st rhs arg, 2nd rhs arg, ...): "D","M","T","K"
        integer(C_INT):: i

        coh_ctrl=-1
        if(nargs.gt.0.and.nargs.le.MAX_TENSOR_OPERANDS) then
         coh_ctrl=0
         aloop: do i=1,nargs
          select case(coh_str(i))
          case('d','D')
           coh_ctrl=coh_ctrl*4+0
          case('m','M')
           coh_ctrl=coh_ctrl*4+1
          case('t','T')
           coh_ctrl=coh_ctrl*4+2
          case('k','K')
           coh_ctrl=coh_ctrl*4+3
          case default
           coh_ctrl=-2
           exit aloop
          end select
         enddo aloop
        endif
        return
        end function coherence_control_var
!--------------------------------------------------------------------
	subroutine tensor_block_shape_create_sym(tens,shape_str,ierr) !SERIAL
!This subroutine generates a tensor shape based on the tensor shape specification string (TSSS) <str>.
!Only the syntax of the TSSS is checked, but not the logical consistency (which can be checked by function <tensor_block_shape_ok>)!
!FORMAT of <shape_str>:
!"(E1/D1{G1},E2/D2{G2},...)":
!  Ex is the extent of dimension x (segment);
!  /Dx specifies an optional segment divider for the dimension x (lm_segment_size), 1<=Dx<=Ex (DEFAULT = Ex);
!      Ex MUST be a multiple of Dx (for simply dense tensor blocks Dx=Ex).
!  {Gx} optionally specifies the symmetric group the dimension belongs to, Gx>=0 (default group 0 has no symmetry ordering).
!       Dimensions grouped together (group#>0) will obey a non-descending ordering from left to right.
!Only dimension-led-dense, bricked-dense, and bricked-ordered formats are considered here.
!By default, the 1st dimension is the most minor one while the last is the most senior (Fortran-like).
!If the number of dimensions equals to zero, the %scalar_value field will be used instead of data arrays.
	implicit none
        type(tensor_block_t), intent(inout):: tens
	character(*), intent(in):: shape_str
	integer, intent(inout):: ierr
	integer i,j,k,l,m,n,k0,k1,k2,k3,ks,kf
	character(MAX_SHAPE_STR_LEN) shp
	logical res

	ierr=0; l=len_trim(shape_str)
	if(l.gt.MAX_SHAPE_STR_LEN) then
	 if(VERBOSE) write(CONS_OUT,*)'FATAL(tensor_algebra::tensor_block_shape_create_sym): '//&
	             &'max length of a shape specification string exceeded: ',l
	 ierr=1; return
	endif
	call remove_blanks(l,shape_str,shp)
	if(l.ge.len('()')) then
	 if(shp(1:1).eq.'('.and.shp(l:l).eq.')') then
	  tens%tensor_shape%num_dim=-1
	  if(tensor_block_alloc(tens,'sp',ierr)) then
	   if(ierr.ne.0) then; ierr=2; return; endif
	   deallocate(tens%tensor_shape%dim_extent,STAT=ierr); if(ierr.ne.0) then; ierr=3; return; endif
	   deallocate(tens%tensor_shape%dim_divider,STAT=ierr); if(ierr.ne.0) then; ierr=4; return; endif
	   deallocate(tens%tensor_shape%dim_group,STAT=ierr); if(ierr.ne.0) then; ierr=5; return; endif
	   res=tensor_block_alloc(tens,'sp',ierr,.FALSE.); if(ierr.ne.0) then; ierr=6; return; endif
	  else
	   if(ierr.ne.0) then; ierr=7; return; endif
	   nullify(tens%tensor_shape%dim_extent)
	   nullify(tens%tensor_shape%dim_divider)
	   nullify(tens%tensor_shape%dim_group)
	  endif
 !count the number of dimensions:
	  n=0
	  if(l.gt.len('()')) then
	   do i=1,l
	    if(shp(i:i).eq.',') n=n+1
	   enddo
	   n=n+1
	  else
	   if(shp(1:l).ne.'()') then; ierr=8; return; endif
	  endif
 !read the specifications:
	  if(n.gt.0) then
	   allocate(tens%tensor_shape%dim_extent(1:n),STAT=ierr); if(ierr.ne.0) then; ierr=9; return; endif
	   allocate(tens%tensor_shape%dim_divider(1:n),STAT=ierr); if(ierr.ne.0) then; ierr=10; return; endif
	   allocate(tens%tensor_shape%dim_group(1:n),STAT=ierr); if(ierr.ne.0) then; ierr=11; return; endif
	   res=tensor_block_alloc(tens,'sp',ierr,.TRUE.); if(ierr.ne.0) then; ierr=12; return; endif
	   tens%tensor_shape%dim_divider(1:n)=0
	   tens%tensor_shape%dim_group(1:n)=0 !by default, each dimension belongs to symmetric group 0 (with no symmetry ordering)
	   n=1; i=len('(')+1; ks=i; kf=0
	   do while(i.le.l)
	    select case(shp(i:i))
	    case(',',')','/','{','}')
	     if(i.gt.ks) then
	      k=i-ks; m=icharnum(k,shp(ks:i-1)); if(k.le.0) then; ierr=13; return; endif
	     else
	      if(kf.ne.3) then; ierr=14; return; endif
	     endif
	     select case(shp(i:i))
	     case(',',')')
	      if(kf.eq.0) then
	       tens%tensor_shape%dim_extent(n)=m
	      elseif(kf.eq.1) then
	       tens%tensor_shape%dim_divider(n)=m
	      elseif(kf.eq.2) then
	       ierr=15; return
	      endif
	      if(shp(i:i).eq.',') then; n=n+1; i=i+1; ks=i; kf=0; else; if(i.ne.l) then; ierr=16; return; endif; i=i+1; endif
	     case('/')
	      if(kf.eq.0) then; tens%tensor_shape%dim_extent(n)=m; else; ierr=17; return; endif
	      i=i+1; ks=i; kf=1
	     case('{')
	      if(kf.eq.0) then
	       tens%tensor_shape%dim_extent(n)=m
	      elseif(kf.eq.1) then
	       tens%tensor_shape%dim_divider(n)=m
	      else
	       ierr=18; return
	      endif
	      i=i+1; ks=i; kf=2
	     case('}')
	      if(kf.eq.2) then; tens%tensor_shape%dim_group(n)=m; else; ierr=19; return; endif
	      i=i+1; ks=i; kf=3
	     end select
	    case default
	     i=i+1
	    end select
	   enddo
	   do i=1,n
	    if(tens%tensor_shape%dim_divider(i).eq.0) tens%tensor_shape%dim_divider(i)=tens%tensor_shape%dim_extent(i)
	   enddo
	   tens%tensor_shape%num_dim=n
	  elseif(n.eq.0) then
	   tens%tensor_shape%num_dim=0 !scalar (rank-0 tensor)
	  endif
	 else
	  ierr=20 !invalid shape descriptor string
	 endif
	else
	 ierr=21 !invalid shape descriptor string
	endif
	return
	contains

	 subroutine remove_blanks(sl,str_in,str_out)
	 integer, intent(inout):: sl
	 character(*), intent(in):: str_in
	 character(*), intent(out):: str_out
	 integer j0,j1
	 j1=0
	 do j0=1,sl
	  if(str_in(j0:j0).ne.' '.and.iachar(str_in(j0:j0)).ne.9) then
	   j1=j1+1; str_out(j1:j1)=str_in(j0:j0)
	  endif
	 enddo
	 sl=j1
	 return
	 end subroutine remove_blanks

	end subroutine tensor_block_shape_create_sym
!-------------------------------------------------------------------------
        subroutine tensor_block_shape_create_num(tens,dims,ierr,divs,grps)
!Constructs the tensor shape inside a <tensor_block_t> object.
!If the <tensor_block_t> object is not empty, it will be destroyed.
         implicit none
         type(tensor_block_t), intent(inout):: tens !inout: tensor block
         integer, intent(in):: dims(1:)             !in: dimension extents (length = tensor rank)
         integer, intent(inout):: ierr              !out: error code (0:success)
         integer, intent(in), optional:: divs(1:)   !in: dimension dividers
         integer, intent(in), optional:: grps(1:)   !in: dimension groups
         integer:: n
         logical:: res

         ierr=0
         if(.not.tensor_block_is_empty(tens)) then; call tensor_block_destroy(tens,ierr); endif
         if(ierr.eq.0) then
          n=size(dims)
          if(n.ge.0.and.n.le.MAX_TENSOR_RANK) then
           tens%tensor_shape%num_dim=n
           if(n.gt.0) then
            allocate(tens%tensor_shape%dim_extent(1:n),STAT=ierr); if(ierr.ne.0) then; ierr=1; return; endif
            allocate(tens%tensor_shape%dim_divider(1:n),STAT=ierr); if(ierr.ne.0) then; ierr=2; return; endif
            allocate(tens%tensor_shape%dim_group(1:n),STAT=ierr); if(ierr.ne.0) then; ierr=3; return; endif
            res=tensor_block_alloc(tens,'sp',ierr,.TRUE.); if(ierr.ne.0) then; ierr=4; return; endif
            tens%tensor_shape%dim_extent(1:n)=dims(1:n)
            if(present(divs)) then
             if(size(divs).eq.n) then
              tens%tensor_shape%dim_divider(1:n)=divs(1:n)
             else
              ierr=5; return
             endif
            else
             tens%tensor_shape%dim_divider(1:n)=tens%tensor_shape%dim_extent(1:n)
            endif
            if(present(grps)) then
             if(size(grps).eq.n) then
              tens%tensor_shape%dim_group(1:n)=grps(1:n)
             else
              ierr=6; return
             endif
            else
             tens%tensor_shape%dim_group(1:n)=0
            endif
           endif
          else
           ierr=7
          endif
         else
          ierr=8
         endif
         return
        end subroutine tensor_block_shape_create_num
!---------------------------------------------------
	integer function tensor_block_shape_ok(tens) !SERIAL
!This function checks the logical correctness of a tensor shape generated from a tensor shape specification string (TSSS).
!INPUT:
! - tens - tensor block;
!OUTPUT:
! - tensor_block_shape_ok - error code (0:success);
!NOTES:
! - Ordered (symmetric) indices must have the same divider! Whether or not should they have the same extent is still debatable for me (D.I.L.).
	implicit none
	type(tensor_block_t), intent(inout):: tens
	integer i,j,k,l,m,n,k0,k1,k2,k3,ks,kf,ierr
	integer group_ext(1:max_tensor_rank),group_div(1:max_tensor_rank)
	logical res

	tensor_block_shape_ok=0; ierr=0
	if(tens%tensor_shape%num_dim.eq.0) then !scalar (rank-0) tensor
	 if(tensor_block_alloc(tens,'sp',ierr)) then
	  if(ierr.ne.0) then; tensor_block_shape_ok=-1; return; endif
	  deallocate(tens%tensor_shape%dim_extent,STAT=ierr); if(ierr.ne.0) then; tensor_block_shape_ok=-2; return; endif
	  deallocate(tens%tensor_shape%dim_divider,STAT=ierr); if(ierr.ne.0) then; tensor_block_shape_ok=-3; return; endif
	  deallocate(tens%tensor_shape%dim_group,STAT=ierr); if(ierr.ne.0) then; tensor_block_shape_ok=-4; return; endif
	  res=tensor_block_alloc(tens,'sp',ierr,.FALSE.); if(ierr.ne.0) then; tensor_block_shape_ok=-5; return; endif
	 else
	  if(ierr.ne.0) then; tensor_block_shape_ok=-6; return; endif
	  nullify(tens%tensor_shape%dim_extent)
	  nullify(tens%tensor_shape%dim_divider)
	  nullify(tens%tensor_shape%dim_group)
	 endif
	elseif(tens%tensor_shape%num_dim.gt.0) then !true tensor (rank>0)
	 n=tens%tensor_shape%num_dim
	 if(n.le.max_tensor_rank) then
	  if(.not.associated(tens%tensor_shape%dim_extent)) then; tensor_block_shape_ok=1; return; endif
	  if(.not.associated(tens%tensor_shape%dim_divider)) then; tensor_block_shape_ok=2; return; endif
	  if(.not.associated(tens%tensor_shape%dim_group)) then; tensor_block_shape_ok=3; return; endif
	  if(size(tens%tensor_shape%dim_extent).ne.n) then; tensor_block_shape_ok=4; return; endif
	  if(size(tens%tensor_shape%dim_divider).ne.n) then; tensor_block_shape_ok=5; return; endif
	  if(size(tens%tensor_shape%dim_group).ne.n) then; tensor_block_shape_ok=6; return; endif
	  kf=0
	  do i=1,n
	   if(tens%tensor_shape%dim_extent(i).le.0) then; tensor_block_shape_ok=7; return; endif
	   if(tens%tensor_shape%dim_divider(i).gt.0.and.tens%tensor_shape%dim_divider(i).le.tens%tensor_shape%dim_extent(i)) then
	    kf=1
	    if(mod(tens%tensor_shape%dim_extent(i),tens%tensor_shape%dim_divider(i)).ne.0) then
	     tensor_block_shape_ok=8; return
	    endif
	   elseif(tens%tensor_shape%dim_divider(i).eq.0) then
	    if(kf.ne.0) then; tensor_block_shape_ok=9; return; endif
	   else !negative divider
	    tensor_block_shape_ok=10; return
	   endif
	  enddo
	  if(kf.ne.0) then !dimension_led or bricked storage layout
	   group_div(1:n)=0
	   do i=1,n
	    if(tens%tensor_shape%dim_group(i).lt.0) then
	     tensor_block_shape_ok=11; return
	    elseif(tens%tensor_shape%dim_group(i).gt.0) then !non-trivial symmetric group
	     if(tens%tensor_shape%dim_group(i).le.n) then
	      if(group_div(tens%tensor_shape%dim_group(i)).eq.0)&
	        &group_div(tens%tensor_shape%dim_group(i))=tens%tensor_shape%dim_divider(i)
	      if(tens%tensor_shape%dim_divider(i).ne.group_div(tens%tensor_shape%dim_group(i))) then
	       tensor_block_shape_ok=12; return !divider must be the same for symmetric dimensions
	      endif
	     else
	      tensor_block_shape_ok=13; return
	     endif
	    endif
	   enddo
	  else !alternative storage layout
	   !`Future
	  endif
	 else !max tensor rank exceeded: increase parameter <max_tensor_rank> of this module
	  tensor_block_shape_ok=-100-max_tensor_rank
	 endif
	else !negative tensor rank
	 tensor_block_shape_ok=14
	endif
	return
	end function tensor_block_shape_ok
!------------------------------------------------------------
        logical function tensor_block_alloc(tens,dk,ierr,sts) !SERIAL
!This function queries or sets the allocation status of data pointers in a tensor block.
!INPUT:
! - tens: tensor block;
! - dk: data pointer being queried or set:
!       'sp': tensor shape pointers (%extent, %divider, %group);
!       'r4': %data_real4 pointer;
!       'r8': %data_real8 pointer;
!       'c4': %data_cmplx4 pointer;
!       'c8': %data_cmplx8 pointer;
! - sts: if present, makes the function set the status to <sts> instead of queriyng;
!OUTPUT:
! - tensor_block_alloc:
!    Query: .TRUE. (pointer is allocated), .FALSE. (pointer is not allocated);
!    Set: = <sts> is the status to be set;
! - ierr: error code (0: success).
!NOTES:
! - %ptr_alloc component of tensor_block_t stores the relevant info:
!    Bits 0-2: %dim_extent, %dim_divider%, %dim_group;
!    Bit 4: %data_real4;
!    Bit 5: %data_real8;
!    Bit 6: %data_cmplx4.
!    Bit 7: %data_cmplx8.
        implicit none
        type(tensor_block_t), intent(inout):: tens
        character(2), intent(in):: dk
        logical, intent(in), optional:: sts
        integer, intent(inout):: ierr
        integer i
        logical stex,stdv,stgr

        ierr=0; tensor_block_alloc=.FALSE.
        select case(dk)
        case('sp')
         if(present(sts)) then !set
          if(sts) then
           tens%ptr_alloc=ibset(tens%ptr_alloc,0)
           tens%ptr_alloc=ibset(tens%ptr_alloc,1)
           tens%ptr_alloc=ibset(tens%ptr_alloc,2)
          else
           tens%ptr_alloc=ibclr(tens%ptr_alloc,0)
           tens%ptr_alloc=ibclr(tens%ptr_alloc,1)
           tens%ptr_alloc=ibclr(tens%ptr_alloc,2)
          endif
          tensor_block_alloc=sts
         else !query
          stex=btest(tens%ptr_alloc,0); stdv=btest(tens%ptr_alloc,1); stgr=btest(tens%ptr_alloc,2)
          if(stex.and.stdv.and.stgr) then !all .TRUE.
           tensor_block_alloc=.TRUE.
           if((.not.associated(tens%tensor_shape%dim_extent)).or.&
             &(.not.associated(tens%tensor_shape%dim_divider)).or.&
             &(.not.associated(tens%tensor_shape%dim_group))) then
            if(VERBOSE) write(CONS_OUT,'("ERROR(tensor_algebra::tensor_block_alloc): disassociated shape marked allocated!")')
            ierr=1; return
           endif
          elseif((.not.stex).and.(.not.stdv).and.(.not.stgr)) then !all .FALSE.
           tensor_block_alloc=.FALSE.
          else
           tensor_block_alloc=.FALSE.; ierr=2
          endif
         endif
        case('r4','R4')
         if(present(sts)) then !set
          if(sts) then
           tens%ptr_alloc=ibset(tens%ptr_alloc,4)
          else
           tens%ptr_alloc=ibclr(tens%ptr_alloc,4)
          endif
          tensor_block_alloc=sts
         else !query
          tensor_block_alloc=btest(tens%ptr_alloc,4)
          if(tensor_block_alloc) then
           if(.not.associated(tens%data_real4)) then
            if(VERBOSE) write(CONS_OUT,'("ERROR(tensor_algebra::tensor_block_alloc): disassociated R4 marked allocated!")')
            ierr=3; return
           endif
          endif
         endif
        case('r8','R8')
         if(present(sts)) then !set
          if(sts) then
           tens%ptr_alloc=ibset(tens%ptr_alloc,5)
          else
           tens%ptr_alloc=ibclr(tens%ptr_alloc,5)
          endif
          tensor_block_alloc=sts
         else !query
          tensor_block_alloc=btest(tens%ptr_alloc,5)
          if(tensor_block_alloc) then
           if(.not.associated(tens%data_real8)) then
            if(VERBOSE) write(CONS_OUT,'("ERROR(tensor_algebra::tensor_block_alloc): disassociated R8 marked allocated!")')
            ierr=4; return
           endif
          endif
         endif
        case('c4','C4')
         if(present(sts)) then !set
          if(sts) then
           tens%ptr_alloc=ibset(tens%ptr_alloc,6)
          else
           tens%ptr_alloc=ibclr(tens%ptr_alloc,6)
          endif
          tensor_block_alloc=sts
         else !query
          tensor_block_alloc=btest(tens%ptr_alloc,6)
          if(tensor_block_alloc) then
           if(.not.associated(tens%data_cmplx4)) then
            if(VERBOSE) write(CONS_OUT,'("ERROR(tensor_algebra::tensor_block_alloc): disassociated C4 marked allocated!")')
            ierr=5; return
           endif
          endif
         endif
        case('c8','C8')
         if(present(sts)) then !set
          if(sts) then
           tens%ptr_alloc=ibset(tens%ptr_alloc,7)
          else
           tens%ptr_alloc=ibclr(tens%ptr_alloc,7)
          endif
          tensor_block_alloc=sts
         else !query
          tensor_block_alloc=btest(tens%ptr_alloc,7)
          if(tensor_block_alloc) then
           if(.not.associated(tens%data_cmplx8)) then
            if(VERBOSE) write(CONS_OUT,'("ERROR(tensor_algebra::tensor_block_alloc): disassociated C8 marked allocated!")')
            ierr=6; return
           endif
          endif
         endif
        case default
         ierr=7
        end select
        return
        end function tensor_block_alloc
!--------------------------------------
!PRIVATE FUNCTIONS:
!---------------------------------------------------------------------------------
        function array_alloc_r4(arr_p,extent,base,in_buffer,fallback) result(ierr) !SERIAL
!Allocates an R4 1d pointer array either in regular Host memory or in the
!Host argument buffer (<in_buffer>=.TRUE.). If the Host argument buffer cannot
!accomdate the memory allocation request, either an error will be returned (<fallback>=.FALSE.)
!or the regular Fortran memory allocator will be used (<fallback>=.TRUE.). If neither
!<in_buffer> nor <fallback> is provided, the default value will be inferred from
!the current memory allocation policy.
         implicit none
         integer:: ierr                                !out: error code
         real(4), pointer, contiguous, intent(inout):: arr_p(:) !out: pointer array (must be NULL on entrance)
         integer(LONGINT), intent(in):: extent         !in: desired array extent (volume)
         integer(LONGINT), intent(in), optional:: base !in: array base (first element, defaults to 1)
         logical, intent(in), optional:: in_buffer     !in: if .TRUE., the array will be allocated in the Host argument buffer
         logical, intent(in), optional:: fallback      !in: if .TRUE., an unsuccessful allocation in the Host argument buffer will be mitigated by a regular allocation
         real(4), pointer, contiguous:: tmp(:)
         integer(LONGINT):: bs
         integer(C_SIZE_T):: arr_size
         type(C_PTR):: cptr
         integer(C_INT):: buf_entry
         logical:: in_buf,fallb

         ierr=0
         if(extent.gt.0) then
          if(.not.associated(arr_p)) then
           if(present(base)) then; bs=base; else; bs=1_LONGINT; endif
           if(present(in_buffer)) then
            in_buf=in_buffer
           else
            if(MEM_ALLOC_POLICY.eq.MEM_ALLOC_REGULAR) then; in_buf=.FALSE.; else; in_buf=.TRUE.; endif
           endif
           if(present(fallback)) then; fallb=fallback; else; fallb=MEM_ALLOC_FALLBACK; endif
           if(in_buf) then !in buffer allocation
            arr_size=extent*int(size_of(R4_),LONGINT) !size in bytes
            ierr=get_buf_entry_host(arr_size,cptr,buf_entry)
            if(ierr.eq.0) then
             call c_f_pointer(cptr,tmp,(/extent/))
             arr_p(bs:)=>tmp(1:); tmp=>NULL()
            else
             if(fallb) in_buf=.FALSE.
            endif
           endif
           if(.not.in_buf) allocate(arr_p(bs:bs+extent-1_LONGINT),STAT=ierr) !regular allocation
          else
           ierr=2
          endif
         else
          ierr=1
         endif
         return
        end function array_alloc_r4
!---------------------------------------------------------------------------------
        function array_alloc_r8(arr_p,extent,base,in_buffer,fallback) result(ierr) !SERIAL
!Allocates an R8 1d pointer array either in regular Host memory or in the
!Host argument buffer (<in_buffer>=.TRUE.). If the Host argument buffer cannot
!accomdate the memory allocation request, either an error will be returned (<fallback>=.FALSE.)
!or the regular Fortran memory allocator will be used (<fallback>=.TRUE.). If neither
!<in_buffer> nor <fallback> is provided, the default value will be inferred from
!the current memory allocation policy.
         implicit none
         integer:: ierr                                !out: error code
         real(8), pointer, contiguous, intent(inout):: arr_p(:) !out: pointer array (must be NULL on entrance)
         integer(LONGINT), intent(in):: extent         !in: desired array extent (volume)
         integer(LONGINT), intent(in), optional:: base !in: array base (first element, defaults to 1)
         logical, intent(in), optional:: in_buffer     !in: if .TRUE., the array will be allocated in the Host argument buffer
         logical, intent(in), optional:: fallback      !in: if .TRUE., an unsuccessful allocation in the Host argument buffer will be mitigated by a regular allocation
         real(8), pointer, contiguous:: tmp(:)
         integer(LONGINT):: bs
         integer(C_SIZE_T):: arr_size
         type(C_PTR):: cptr
         integer(C_INT):: buf_entry
         logical:: in_buf,fallb

         ierr=0
         if(extent.gt.0) then
          if(.not.associated(arr_p)) then
           if(present(base)) then; bs=base; else; bs=1_LONGINT; endif
           if(present(in_buffer)) then
            in_buf=in_buffer
           else
            if(MEM_ALLOC_POLICY.eq.MEM_ALLOC_REGULAR) then; in_buf=.FALSE.; else; in_buf=.TRUE.; endif
           endif
           if(present(fallback)) then; fallb=fallback; else; fallb=MEM_ALLOC_FALLBACK; endif
           if(in_buf) then !in buffer allocation
            arr_size=extent*int(size_of(R8_),LONGINT) !size in bytes
            ierr=get_buf_entry_host(arr_size,cptr,buf_entry)
            if(ierr.eq.0) then
             call c_f_pointer(cptr,tmp,(/extent/))
             arr_p(bs:)=>tmp(1:); tmp=>NULL()
            else
             if(fallb) in_buf=.FALSE.
            endif
           endif
           if(.not.in_buf) allocate(arr_p(bs:bs+extent-1_LONGINT),STAT=ierr) !regular allocation
          else
           ierr=2
          endif
         else
          ierr=1
         endif
         return
        end function array_alloc_r8
!---------------------------------------------------------------------------------
        function array_alloc_c4(arr_p,extent,base,in_buffer,fallback) result(ierr) !SERIAL
!Allocates an C4 1d pointer array either in regular Host memory or in the
!Host argument buffer (<in_buffer>=.TRUE.). If the Host argument buffer cannot
!accomdate the memory allocation request, either an error will be returned (<fallback>=.FALSE.)
!or the regular Fortran memory allocator will be used (<fallback>=.TRUE.). If neither
!<in_buffer> nor <fallback> is provided, the default value will be inferred from
!the current memory allocation policy.
         implicit none
         integer:: ierr                                !out: error code
         complex(4), pointer, contiguous, intent(inout):: arr_p(:) !out: pointer array (must be NULL on entrance)
         integer(LONGINT), intent(in):: extent         !in: desired array extent (volume)
         integer(LONGINT), intent(in), optional:: base !in: array base (first element, defaults to 1)
         logical, intent(in), optional:: in_buffer     !in: if .TRUE., the array will be allocated in the Host argument buffer
         logical, intent(in), optional:: fallback      !in: if .TRUE., an unsuccessful allocation in the Host argument buffer will be mitigated by a regular allocation
         complex(4), pointer, contiguous:: tmp(:)
         integer(LONGINT):: bs
         integer(C_SIZE_T):: arr_size
         type(C_PTR):: cptr
         integer(C_INT):: buf_entry
         logical:: in_buf,fallb

         ierr=0
         if(extent.gt.0) then
          if(.not.associated(arr_p)) then
           if(present(base)) then; bs=base; else; bs=1_LONGINT; endif
           if(present(in_buffer)) then
            in_buf=in_buffer
           else
            if(MEM_ALLOC_POLICY.eq.MEM_ALLOC_REGULAR) then; in_buf=.FALSE.; else; in_buf=.TRUE.; endif
           endif
           if(present(fallback)) then; fallb=fallback; else; fallb=MEM_ALLOC_FALLBACK; endif
           if(in_buf) then !in buffer allocation
            arr_size=extent*int(size_of(C4_),LONGINT) !size in bytes
            ierr=get_buf_entry_host(arr_size,cptr,buf_entry)
            if(ierr.eq.0) then
             call c_f_pointer(cptr,tmp,(/extent/))
             arr_p(bs:)=>tmp(1:); tmp=>NULL()
            else
             if(fallb) in_buf=.FALSE.
            endif
           endif
           if(.not.in_buf) allocate(arr_p(bs:bs+extent-1_LONGINT),STAT=ierr) !regular allocation
          else
           ierr=2
          endif
         else
          ierr=1
         endif
         return
        end function array_alloc_c4
!---------------------------------------------------------------------------------
        function array_alloc_c8(arr_p,extent,base,in_buffer,fallback) result(ierr) !SERIAL
!Allocates an C8 1d pointer array either in regular Host memory or in the
!Host argument buffer (<in_buffer>=.TRUE.). If the Host argument buffer cannot
!accomdate the memory allocation request, either an error will be returned (<fallback>=.FALSE.)
!or the regular Fortran memory allocator will be used (<fallback>=.TRUE.). If neither
!<in_buffer> nor <fallback> is provided, the default value will be inferred from
!the current memory allocation policy.
         implicit none
         integer:: ierr                                !out: error code
         complex(8), pointer, contiguous, intent(inout):: arr_p(:) !out: pointer array (must be NULL on entrance)
         integer(LONGINT), intent(in):: extent         !in: desired array extent (volume)
         integer(LONGINT), intent(in), optional:: base !in: array base (first element, defaults to 1)
         logical, intent(in), optional:: in_buffer     !in: if .TRUE., the array will be allocated in the Host argument buffer
         logical, intent(in), optional:: fallback      !in: if .TRUE., an unsuccessful allocation in the Host argument buffer will be mitigated by a regular allocation
         complex(8), pointer, contiguous:: tmp(:)
         integer(LONGINT):: bs
         integer(C_SIZE_T):: arr_size
         type(C_PTR):: cptr
         integer(C_INT):: buf_entry
         logical:: in_buf,fallb

         ierr=0
         if(extent.gt.0) then
          if(.not.associated(arr_p)) then
           if(present(base)) then; bs=base; else; bs=1_LONGINT; endif
           if(present(in_buffer)) then
            in_buf=in_buffer
           else
            if(MEM_ALLOC_POLICY.eq.MEM_ALLOC_REGULAR) then; in_buf=.FALSE.; else; in_buf=.TRUE.; endif
           endif
           if(present(fallback)) then; fallb=fallback; else; fallb=MEM_ALLOC_FALLBACK; endif
           if(in_buf) then !in buffer allocation
            arr_size=extent*int(size_of(C8_),LONGINT) !size in bytes
            ierr=get_buf_entry_host(arr_size,cptr,buf_entry)
            if(ierr.eq.0) then
             call c_f_pointer(cptr,tmp,(/extent/))
             arr_p(bs:)=>tmp(1:); tmp=>NULL()
            else
             if(fallb) in_buf=.FALSE.
            endif
           endif
           if(.not.in_buf) allocate(arr_p(bs:bs+extent-1_LONGINT),STAT=ierr) !regular allocation
          else
           ierr=2
          endif
         else
          ierr=1
         endif
         return
        end function array_alloc_c8
!-------------------------------------------
        subroutine array_free_r4(arr_p,ierr) !SERIAL
!Deallocates an R4 1d pointer array which had been previously allocated
!either via a regular allocate() or in the Host argument buffer.
         implicit none
         real(4), pointer, contiguous, intent(inout):: arr_p(:) !in: previously allocated pointer array
         integer, intent(out), optional:: ierr                  !out: error code
         integer(INTD):: i
         integer:: errc

         if(associated(arr_p)) then
          i=get_buf_entry_from_address(encode_device_id(DEV_HOST,0),c_loc(arr_p))
          if(i.ge.0) then !in buffer
           errc=free_buf_entry_host(i); if(errc.eq.0) nullify(arr_p)
          elseif(i.eq.-1) then !regular allocation
           deallocate(arr_p,STAT=errc)
          else !error
           errc=i
          endif
         else
          errc=-1
         endif
         if(present(ierr)) ierr=errc
         return
        end subroutine array_free_r4
!-------------------------------------------
        subroutine array_free_r8(arr_p,ierr) !SERIAL
!Deallocates an R8 1d pointer array which had been previously allocated
!either via a regular allocate() or in the Host argument buffer.
         implicit none
         real(8), pointer, contiguous, intent(inout):: arr_p(:) !in: previously allocated pointer array
         integer, intent(out), optional:: ierr                  !out: error code
         integer(INTD):: i
         integer:: errc

         if(associated(arr_p)) then
          i=get_buf_entry_from_address(encode_device_id(DEV_HOST,0),c_loc(arr_p))
          if(i.ge.0) then !in buffer
           errc=free_buf_entry_host(i); if(errc.eq.0) nullify(arr_p)
          elseif(i.eq.-1) then !regular allocation
           deallocate(arr_p,STAT=errc)
          else !error
           errc=i
          endif
         else
          errc=-1
         endif
         if(present(ierr)) ierr=errc
         return
        end subroutine array_free_r8
!-------------------------------------------
        subroutine array_free_c4(arr_p,ierr) !SERIAL
!Deallocates an C4 1d pointer array which had been previously allocated
!either via a regular allocate() or in the Host argument buffer.
         implicit none
         complex(4), pointer, contiguous, intent(inout):: arr_p(:) !in: previously allocated pointer array
         integer, intent(out), optional:: ierr                     !out: error code
         integer(INTD):: i
         integer:: errc

         if(associated(arr_p)) then
          i=get_buf_entry_from_address(encode_device_id(DEV_HOST,0),c_loc(arr_p))
          if(i.ge.0) then !in buffer
           errc=free_buf_entry_host(i); if(errc.eq.0) nullify(arr_p)
          elseif(i.eq.-1) then !regular allocation
           deallocate(arr_p,STAT=errc)
          else !error
           errc=i
          endif
         else
          errc=-1
         endif
         if(present(ierr)) ierr=errc
         return
        end subroutine array_free_c4
!-------------------------------------------
        subroutine array_free_c8(arr_p,ierr) !SERIAL
!Deallocates an C8 1d pointer array which had been previously allocated
!either via a regular allocate() or in the Host argument buffer.
         implicit none
         complex(8), pointer, contiguous, intent(inout):: arr_p(:) !in: previously allocated pointer array
         integer, intent(out), optional:: ierr                     !out: error code
         integer(INTD):: i
         integer:: errc

         if(associated(arr_p)) then
          i=get_buf_entry_from_address(encode_device_id(DEV_HOST,0),c_loc(arr_p))
          if(i.ge.0) then !in buffer
           errc=free_buf_entry_host(i); if(errc.eq.0) nullify(arr_p)
          elseif(i.eq.-1) then !regular allocation
           deallocate(arr_p,STAT=errc)
          else !error
           errc=i
          endif
         else
          errc=-1
         endif
         if(present(ierr)) ierr=errc
         return
        end subroutine array_free_c8
!-----------------------------------------------------------------------------------------------
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: tensor_block_slice_dlf_r4
#endif
	subroutine tensor_block_slice_dlf_r4(dim_num,tens,tens_ext,slice,slice_ext,ext_beg,ierr) !PARALLEL
!This subroutine extracts a slice from a tensor block.
!INPUT:
! - dim_num - number of tensor dimensions;
! - tens(0:) - tensor block (array);
! - tens_ext(1:dim_num) - dimension extents for <tens>;
! - slice_ext(1:dim_num) - dimension extents for <slice>;
! - ext_beg(1:dim_num) - beginning dimension offsets for <tens> (numeration starts at 0);
!OUTPUT:
! - slice(0:) - slice (array);
! - ierr - error code (0:success).
!NOTES:
! - No argument validity checks.
	implicit none
!---------------------------------------
	integer, parameter:: real_kind=4
!---------------------------------------
	integer, intent(in):: dim_num,tens_ext(1:dim_num),slice_ext(1:dim_num),ext_beg(1:dim_num)
	real(real_kind), intent(in):: tens(0:*)
	real(real_kind), intent(out):: slice(0:*)
	integer, intent(inout):: ierr
	integer i,j,k,l,m,n,ks,kf,im(1:dim_num)
	integer(LONGINT):: lts,lss,l_in,l_out,lb,le,ll,bases_in(1:dim_num),bases_out(1:dim_num),segs(0:MAX_THREADS) !`Is segs(:) threadsafe?
	real(8) time_beg
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: real_kind
!DIR$ ATTRIBUTES ALIGN:128:: real_kind,im,bases_in,bases_out,segs
#endif

	ierr=0
!	time_beg=thread_wtime() !debug
	if(dim_num.gt.0) then
	 lts=1_LONGINT; do i=1,dim_num; bases_in(i)=lts; lts=lts*tens_ext(i); enddo   !tensor block indexing bases
	 lss=1_LONGINT; do i=1,dim_num; bases_out(i)=lss; lss=lss*slice_ext(i); enddo !slice indexing bases
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,m,n,im,l_in,l_out,lb,le,ll)
#ifndef NO_OMP
	 n=omp_get_thread_num(); m=omp_get_num_threads()
#else
	 n=0; m=1
#endif
!$OMP MASTER
	 segs(0)=0_LONGINT; call divide_segment(lss,int(m,LONGINT),segs(1:),ierr); do i=2,m; segs(i)=segs(i)+segs(i-1); enddo
!$OMP END MASTER
!$OMP BARRIER
!$OMP FLUSH(segs)
	 l_out=segs(n); do i=dim_num,1,-1; im(i)=l_out/bases_out(i); l_out=l_out-im(i)*bases_out(i); enddo
	 l_in=ext_beg(1); do i=2,dim_num; l_in=l_in+(ext_beg(i)+im(i))*bases_in(i); enddo
	 lb=int(im(1),LONGINT); le=int(slice_ext(1)-1,LONGINT); l_out=segs(n)-lb
	 sloop: do while(l_out+lb.lt.segs(n+1))
	  le=min(le,segs(n+1)-1_LONGINT-l_out) !to avoid different threads doing the same work
	  do ll=lb,le; slice(l_out+ll)=tens(l_in+ll); enddo
	  l_out=l_out+le+1_LONGINT; lb=0_LONGINT
	  do i=2,dim_num
	   if(im(i)+1.lt.slice_ext(i)) then
	    im(i)=im(i)+1; l_in=l_in+bases_in(i); exit
	   else
	    l_in=l_in-im(i)*bases_in(i); im(i)=0
	   endif
	  enddo
	 enddo sloop
!$OMP END PARALLEL
	else
	 ierr=1 !zero-rank tensor
	endif
!	write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_slice_dlf_r4): kernel time/error code: ",F10.4,1x,i3)') &
!        thread_wtime(time_beg),ierr !debug
	return
	end subroutine tensor_block_slice_dlf_r4
!-----------------------------------------------------------------------------------------------
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: tensor_block_slice_dlf_r8
#endif
	subroutine tensor_block_slice_dlf_r8(dim_num,tens,tens_ext,slice,slice_ext,ext_beg,ierr) !PARALLEL
!This subroutine extracts a slice from a tensor block.
!INPUT:
! - dim_num - number of tensor dimensions;
! - tens(0:) - tensor block (array);
! - tens_ext(1:dim_num) - dimension extents for <tens>;
! - slice_ext(1:dim_num) - dimension extents for <slice>;
! - ext_beg(1:dim_num) - beginning dimension offsets for <tens> (numeration starts at 0);
!OUTPUT:
! - slice(0:) - slice (array);
! - ierr - error code (0:success).
!NOTES:
! - No argument validity checks.
	implicit none
!---------------------------------------
	integer, parameter:: real_kind=8
!---------------------------------------
	integer, intent(in):: dim_num,tens_ext(1:dim_num),slice_ext(1:dim_num),ext_beg(1:dim_num)
	real(real_kind), intent(in):: tens(0:*)
	real(real_kind), intent(out):: slice(0:*)
	integer, intent(inout):: ierr
	integer i,j,k,l,m,n,ks,kf,im(1:dim_num)
	integer(LONGINT):: lts,lss,l_in,l_out,lb,le,ll,bases_in(1:dim_num),bases_out(1:dim_num),segs(0:MAX_THREADS) !`Is segs(:) threadsafe?
	real(8) time_beg
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: real_kind
!DIR$ ATTRIBUTES ALIGN:128:: real_kind,im,bases_in,bases_out,segs
#endif

	ierr=0
!	time_beg=thread_wtime() !debug
	if(dim_num.gt.0) then
	 lts=1_LONGINT; do i=1,dim_num; bases_in(i)=lts; lts=lts*tens_ext(i); enddo   !tensor block indexing bases
	 lss=1_LONGINT; do i=1,dim_num; bases_out(i)=lss; lss=lss*slice_ext(i); enddo !slice indexing bases
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,m,n,im,l_in,l_out,lb,le,ll)
#ifndef NO_OMP
	 n=omp_get_thread_num(); m=omp_get_num_threads()
#else
	 n=0; m=1
#endif
!$OMP MASTER
	 segs(0)=0_LONGINT; call divide_segment(lss,int(m,LONGINT),segs(1:),ierr); do i=2,m; segs(i)=segs(i)+segs(i-1); enddo
!$OMP END MASTER
!$OMP BARRIER
!$OMP FLUSH(segs)
	 l_out=segs(n); do i=dim_num,1,-1; im(i)=l_out/bases_out(i); l_out=l_out-im(i)*bases_out(i); enddo
	 l_in=ext_beg(1); do i=2,dim_num; l_in=l_in+(ext_beg(i)+im(i))*bases_in(i); enddo
	 lb=int(im(1),LONGINT); le=int(slice_ext(1)-1,LONGINT); l_out=segs(n)-lb
	 sloop: do while(l_out+lb.lt.segs(n+1))
	  le=min(le,segs(n+1)-1_LONGINT-l_out) !to avoid different threads doing the same work
	  do ll=lb,le; slice(l_out+ll)=tens(l_in+ll); enddo
	  l_out=l_out+le+1_LONGINT; lb=0_LONGINT
	  do i=2,dim_num
	   if(im(i)+1.lt.slice_ext(i)) then
	    im(i)=im(i)+1; l_in=l_in+bases_in(i); exit
	   else
	    l_in=l_in-im(i)*bases_in(i); im(i)=0
	   endif
	  enddo
	 enddo sloop
!$OMP END PARALLEL
	else
	 ierr=1 !zero-rank tensor
	endif
!	write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_slice_dlf_r8): kernel time/error code: ",F10.4,1x,i3)') &
!        thread_wtime(time_beg),ierr !debug
	return
	end subroutine tensor_block_slice_dlf_r8
!-----------------------------------------------------------------------------------------------
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: tensor_block_slice_dlf_c4
#endif
	subroutine tensor_block_slice_dlf_c4(dim_num,tens,tens_ext,slice,slice_ext,ext_beg,ierr) !PARALLEL
!This subroutine extracts a slice from a tensor block.
!INPUT:
! - dim_num - number of tensor dimensions;
! - tens(0:) - tensor block (array);
! - tens_ext(1:dim_num) - dimension extents for <tens>;
! - slice_ext(1:dim_num) - dimension extents for <slice>;
! - ext_beg(1:dim_num) - beginning dimension offsets for <tens> (numeration starts at 0);
!OUTPUT:
! - slice(0:) - slice (array);
! - ierr - error code (0:success).
!NOTES:
! - No argument validity checks.
	implicit none
!---------------------------------------
	integer, parameter:: real_kind=4
!---------------------------------------
	integer, intent(in):: dim_num,tens_ext(1:dim_num),slice_ext(1:dim_num),ext_beg(1:dim_num)
	complex(real_kind), intent(in):: tens(0:*)
	complex(real_kind), intent(out):: slice(0:*)
	integer, intent(inout):: ierr
	integer i,j,k,l,m,n,ks,kf,im(1:dim_num)
	integer(LONGINT):: lts,lss,l_in,l_out,lb,le,ll,bases_in(1:dim_num),bases_out(1:dim_num),segs(0:MAX_THREADS) !`Is segs(:) threadsafe?
	real(8) time_beg
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: real_kind
!DIR$ ATTRIBUTES ALIGN:128:: real_kind,im,bases_in,bases_out,segs
#endif

	ierr=0
!	time_beg=thread_wtime() !debug
	if(dim_num.gt.0) then
	 lts=1_LONGINT; do i=1,dim_num; bases_in(i)=lts; lts=lts*tens_ext(i); enddo   !tensor block indexing bases
	 lss=1_LONGINT; do i=1,dim_num; bases_out(i)=lss; lss=lss*slice_ext(i); enddo !slice indexing bases
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,m,n,im,l_in,l_out,lb,le,ll)
#ifndef NO_OMP
	 n=omp_get_thread_num(); m=omp_get_num_threads()
#else
	 n=0; m=1
#endif
!$OMP MASTER
	 segs(0)=0_LONGINT; call divide_segment(lss,int(m,LONGINT),segs(1:),ierr); do i=2,m; segs(i)=segs(i)+segs(i-1); enddo
!$OMP END MASTER
!$OMP BARRIER
!$OMP FLUSH(segs)
	 l_out=segs(n); do i=dim_num,1,-1; im(i)=l_out/bases_out(i); l_out=l_out-im(i)*bases_out(i); enddo
	 l_in=ext_beg(1); do i=2,dim_num; l_in=l_in+(ext_beg(i)+im(i))*bases_in(i); enddo
	 lb=int(im(1),LONGINT); le=int(slice_ext(1)-1,LONGINT); l_out=segs(n)-lb
	 sloop: do while(l_out+lb.lt.segs(n+1))
	  le=min(le,segs(n+1)-1_LONGINT-l_out) !to avoid different threads doing the same work
	  do ll=lb,le; slice(l_out+ll)=tens(l_in+ll); enddo
	  l_out=l_out+le+1_LONGINT; lb=0_LONGINT
	  do i=2,dim_num
	   if(im(i)+1.lt.slice_ext(i)) then
	    im(i)=im(i)+1; l_in=l_in+bases_in(i); exit
	   else
	    l_in=l_in-im(i)*bases_in(i); im(i)=0
	   endif
	  enddo
	 enddo sloop
!$OMP END PARALLEL
	else
	 ierr=1 !zero-rank tensor
	endif
!	write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_slice_dlf_c4): kernel time/error code: ",F10.4,1x,i3)') &
!        thread_wtime(time_beg),ierr !debug
	return
	end subroutine tensor_block_slice_dlf_c4
!-----------------------------------------------------------------------------------------------
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: tensor_block_slice_dlf_c8
#endif
	subroutine tensor_block_slice_dlf_c8(dim_num,tens,tens_ext,slice,slice_ext,ext_beg,ierr) !PARALLEL
!This subroutine extracts a slice from a tensor block.
!INPUT:
! - dim_num - number of tensor dimensions;
! - tens(0:) - tensor block (array);
! - tens_ext(1:dim_num) - dimension extents for <tens>;
! - slice_ext(1:dim_num) - dimension extents for <slice>;
! - ext_beg(1:dim_num) - beginning dimension offsets for <tens> (numeration starts at 0);
!OUTPUT:
! - slice(0:) - slice (array);
! - ierr - error code (0:success).
!NOTES:
! - No argument validity checks.
	implicit none
!---------------------------------------
	integer, parameter:: real_kind=8
!---------------------------------------
	integer, intent(in):: dim_num,tens_ext(1:dim_num),slice_ext(1:dim_num),ext_beg(1:dim_num)
	complex(real_kind), intent(in):: tens(0:*)
	complex(real_kind), intent(out):: slice(0:*)
	integer, intent(inout):: ierr
	integer i,j,k,l,m,n,ks,kf,im(1:dim_num)
	integer(LONGINT):: lts,lss,l_in,l_out,lb,le,ll,bases_in(1:dim_num),bases_out(1:dim_num),segs(0:MAX_THREADS) !`Is segs(:) threadsafe?
	real(8) time_beg
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: real_kind
!DIR$ ATTRIBUTES ALIGN:128:: real_kind,im,bases_in,bases_out,segs
#endif

	ierr=0
!	time_beg=thread_wtime() !debug
	if(dim_num.gt.0) then
	 lts=1_LONGINT; do i=1,dim_num; bases_in(i)=lts; lts=lts*tens_ext(i); enddo   !tensor block indexing bases
	 lss=1_LONGINT; do i=1,dim_num; bases_out(i)=lss; lss=lss*slice_ext(i); enddo !slice indexing bases
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,m,n,im,l_in,l_out,lb,le,ll)
#ifndef NO_OMP
	 n=omp_get_thread_num(); m=omp_get_num_threads()
#else
	 n=0; m=1
#endif
!$OMP MASTER
	 segs(0)=0_LONGINT; call divide_segment(lss,int(m,LONGINT),segs(1:),ierr); do i=2,m; segs(i)=segs(i)+segs(i-1); enddo
!$OMP END MASTER
!$OMP BARRIER
!$OMP FLUSH(segs)
	 l_out=segs(n); do i=dim_num,1,-1; im(i)=l_out/bases_out(i); l_out=l_out-im(i)*bases_out(i); enddo
	 l_in=ext_beg(1); do i=2,dim_num; l_in=l_in+(ext_beg(i)+im(i))*bases_in(i); enddo
	 lb=int(im(1),LONGINT); le=int(slice_ext(1)-1,LONGINT); l_out=segs(n)-lb
	 sloop: do while(l_out+lb.lt.segs(n+1))
	  le=min(le,segs(n+1)-1_LONGINT-l_out) !to avoid different threads doing the same work
	  do ll=lb,le; slice(l_out+ll)=tens(l_in+ll); enddo
	  l_out=l_out+le+1_LONGINT; lb=0_LONGINT
	  do i=2,dim_num
	   if(im(i)+1.lt.slice_ext(i)) then
	    im(i)=im(i)+1; l_in=l_in+bases_in(i); exit
	   else
	    l_in=l_in-im(i)*bases_in(i); im(i)=0
	   endif
	  enddo
	 enddo sloop
!$OMP END PARALLEL
	else
	 ierr=1 !zero-rank tensor
	endif
!	write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_slice_dlf_c8): kernel time/error code: ",F10.4,1x,i3)') &
!        thread_wtime(time_beg),ierr !debug
	return
	end subroutine tensor_block_slice_dlf_c8
!------------------------------------------------------------------------------------------------
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: tensor_block_insert_dlf_r4
#endif
	subroutine tensor_block_insert_dlf_r4(dim_num,tens,tens_ext,slice,slice_ext,ext_beg,ierr) !PARALLEL
!This subroutine inserts a slice into a tensor block.
!INPUT:
! - dim_num - number of tensor dimensions;
! - tens_ext(1:dim_num) - dimension extents for <tens>;
! - slice(0:) - slice (array);
! - slice_ext(1:dim_num) - dimension extents for <slice>;
! - ext_beg(1:dim_num) - beginning dimension offsets for <tens> (numeration starts at 0);
!OUTPUT:
! - tens(0:) - tensor block (array);
! - ierr - error code (0:success).
!NOTES:
! - No argument validity checks.
	implicit none
!---------------------------------------
	integer, parameter:: real_kind=4
!---------------------------------------
	integer, intent(in):: dim_num,tens_ext(1:dim_num),slice_ext(1:dim_num),ext_beg(1:dim_num)
	real(real_kind), intent(in):: slice(0:*)
	real(real_kind), intent(inout):: tens(0:*)
	integer, intent(inout):: ierr
	integer i,j,k,l,m,n,ks,kf,im(1:dim_num)
	integer(LONGINT):: lts,lss,l_in,l_out,lb,le,ll,bases_in(1:dim_num),bases_out(1:dim_num),segs(0:MAX_THREADS) !`Is segs(:) threadsafe?
	real(8) time_beg
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: real_kind
!DIR$ ATTRIBUTES ALIGN:128:: real_kind,im,bases_in,bases_out,segs
#endif

	ierr=0
!	time_beg=thread_wtime() !debug
	if(dim_num.gt.0) then
	 lts=1_LONGINT; do i=1,dim_num; bases_out(i)=lts; lts=lts*tens_ext(i); enddo !tensor block indexing bases
	 lss=1_LONGINT; do i=1,dim_num; bases_in(i)=lss; lss=lss*slice_ext(i); enddo !slice indexing bases
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,m,n,im,l_in,l_out,lb,le,ll)
#ifndef NO_OMP
	 n=omp_get_thread_num(); m=omp_get_num_threads()
#else
	 n=0; m=1
#endif
!$OMP MASTER
	 segs(0)=0_LONGINT; call divide_segment(lss,int(m,LONGINT),segs(1:),ierr); do i=2,m; segs(i)=segs(i)+segs(i-1); enddo
!$OMP END MASTER
!$OMP BARRIER
!$OMP FLUSH(segs)
	 l_in=segs(n); do i=dim_num,1,-1; im(i)=l_in/bases_in(i); l_in=l_in-im(i)*bases_in(i); enddo
	 l_out=ext_beg(1); do i=2,dim_num; l_out=l_out+(ext_beg(i)+im(i))*bases_out(i); enddo
	 lb=int(im(1),LONGINT); le=int(slice_ext(1)-1,LONGINT); l_in=segs(n)-lb
	 sloop: do while(l_in+lb.lt.segs(n+1))
	  le=min(le,segs(n+1)-1_LONGINT-l_in) !to avoid different threads doing the same work
	  do ll=lb,le; tens(l_out+ll)=slice(l_in+ll); enddo
	  l_in=l_in+le+1_LONGINT; lb=0_LONGINT
	  do i=2,dim_num
	   if(im(i)+1.lt.slice_ext(i)) then
	    im(i)=im(i)+1; l_out=l_out+bases_out(i)
	   else
	    l_out=l_out-im(i)*bases_out(i); im(i)=0
	   endif
	  enddo
	 enddo sloop
!$OMP END PARALLEL
	else
	 ierr=1 !zero-rank tensor
	endif
!	write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_insert_dlf_r4): kernel time/error code: ",F10.4,1x,i3)') &
!        thread_wtime(time_beg),ierr !debug
	return
	end subroutine tensor_block_insert_dlf_r4
!------------------------------------------------------------------------------------------------
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: tensor_block_insert_dlf_r8
#endif
	subroutine tensor_block_insert_dlf_r8(dim_num,tens,tens_ext,slice,slice_ext,ext_beg,ierr) !PARALLEL
!This subroutine inserts a slice into a tensor block.
!INPUT:
! - dim_num - number of tensor dimensions;
! - tens_ext(1:dim_num) - dimension extents for <tens>;
! - slice(0:) - slice (array);
! - slice_ext(1:dim_num) - dimension extents for <slice>;
! - ext_beg(1:dim_num) - beginning dimension offsets for <tens> (numeration starts at 0);
!OUTPUT:
! - tens(0:) - tensor block (array);
! - ierr - error code (0:success).
!NOTES:
! - No argument validity checks.
	implicit none
!---------------------------------------
	integer, parameter:: real_kind=8
!---------------------------------------
	integer, intent(in):: dim_num,tens_ext(1:dim_num),slice_ext(1:dim_num),ext_beg(1:dim_num)
	real(real_kind), intent(in):: slice(0:*)
	real(real_kind), intent(inout):: tens(0:*)
	integer, intent(inout):: ierr
	integer i,j,k,l,m,n,ks,kf,im(1:dim_num)
	integer(LONGINT):: lts,lss,l_in,l_out,lb,le,ll,bases_in(1:dim_num),bases_out(1:dim_num),segs(0:MAX_THREADS) !`Is segs(:) threadsafe?
	real(8) time_beg
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: real_kind
!DIR$ ATTRIBUTES ALIGN:128:: real_kind,im,bases_in,bases_out,segs
#endif

	ierr=0
!	time_beg=thread_wtime() !debug
	if(dim_num.gt.0) then
	 lts=1_LONGINT; do i=1,dim_num; bases_out(i)=lts; lts=lts*tens_ext(i); enddo !tensor block indexing bases
	 lss=1_LONGINT; do i=1,dim_num; bases_in(i)=lss; lss=lss*slice_ext(i); enddo !slice indexing bases
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,m,n,im,l_in,l_out,lb,le,ll)
#ifndef NO_OMP
	 n=omp_get_thread_num(); m=omp_get_num_threads()
#else
	 n=0; m=1
#endif
!$OMP MASTER
	 segs(0)=0_LONGINT; call divide_segment(lss,int(m,LONGINT),segs(1:),ierr); do i=2,m; segs(i)=segs(i)+segs(i-1); enddo
!$OMP END MASTER
!$OMP BARRIER
!$OMP FLUSH(segs)
	 l_in=segs(n); do i=dim_num,1,-1; im(i)=l_in/bases_in(i); l_in=l_in-im(i)*bases_in(i); enddo
	 l_out=ext_beg(1); do i=2,dim_num; l_out=l_out+(ext_beg(i)+im(i))*bases_out(i); enddo
	 lb=int(im(1),LONGINT); le=int(slice_ext(1)-1,LONGINT); l_in=segs(n)-lb
	 sloop: do while(l_in+lb.lt.segs(n+1))
	  le=min(le,segs(n+1)-1_LONGINT-l_in) !to avoid different threads doing the same work
	  do ll=lb,le; tens(l_out+ll)=slice(l_in+ll); enddo
	  l_in=l_in+le+1_LONGINT; lb=0_LONGINT
	  do i=2,dim_num
	   if(im(i)+1.lt.slice_ext(i)) then
	    im(i)=im(i)+1; l_out=l_out+bases_out(i)
	   else
	    l_out=l_out-im(i)*bases_out(i); im(i)=0
	   endif
	  enddo
	 enddo sloop
!$OMP END PARALLEL
	else
	 ierr=1 !zero-rank tensor
	endif
!	write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_insert_dlf_r8): kernel time/error code: ",F10.4,1x,i3)') &
!        thread_wtime(time_beg),ierr !debug
	return
	end subroutine tensor_block_insert_dlf_r8
!------------------------------------------------------------------------------------------------
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: tensor_block_insert_dlf_c4
#endif
	subroutine tensor_block_insert_dlf_c4(dim_num,tens,tens_ext,slice,slice_ext,ext_beg,ierr) !PARALLEL
!This subroutine inserts a slice into a tensor block.
!INPUT:
! - dim_num - number of tensor dimensions;
! - tens_ext(1:dim_num) - dimension extents for <tens>;
! - slice(0:) - slice (array);
! - slice_ext(1:dim_num) - dimension extents for <slice>;
! - ext_beg(1:dim_num) - beginning dimension offsets for <tens> (numeration starts at 0);
!OUTPUT:
! - tens(0:) - tensor block (array);
! - ierr - error code (0:success).
!NOTES:
! - No argument validity checks.
	implicit none
!---------------------------------------
	integer, parameter:: real_kind=4
!---------------------------------------
	integer, intent(in):: dim_num,tens_ext(1:dim_num),slice_ext(1:dim_num),ext_beg(1:dim_num)
	complex(real_kind), intent(in):: slice(0:*)
	complex(real_kind), intent(inout):: tens(0:*)
	integer, intent(inout):: ierr
	integer i,j,k,l,m,n,ks,kf,im(1:dim_num)
	integer(LONGINT):: lts,lss,l_in,l_out,lb,le,ll,bases_in(1:dim_num),bases_out(1:dim_num),segs(0:MAX_THREADS) !`Is segs(:) threadsafe?
	real(8) time_beg
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: real_kind
!DIR$ ATTRIBUTES ALIGN:128:: real_kind,im,bases_in,bases_out,segs
#endif

	ierr=0
!	time_beg=thread_wtime() !debug
	if(dim_num.gt.0) then
	 lts=1_LONGINT; do i=1,dim_num; bases_out(i)=lts; lts=lts*tens_ext(i); enddo !tensor block indexing bases
	 lss=1_LONGINT; do i=1,dim_num; bases_in(i)=lss; lss=lss*slice_ext(i); enddo !slice indexing bases
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,m,n,im,l_in,l_out,lb,le,ll)
#ifndef NO_OMP
	 n=omp_get_thread_num(); m=omp_get_num_threads()
#else
	 n=0; m=1
#endif
!$OMP MASTER
	 segs(0)=0_LONGINT; call divide_segment(lss,int(m,LONGINT),segs(1:),ierr); do i=2,m; segs(i)=segs(i)+segs(i-1); enddo
!$OMP END MASTER
!$OMP BARRIER
!$OMP FLUSH(segs)
	 l_in=segs(n); do i=dim_num,1,-1; im(i)=l_in/bases_in(i); l_in=l_in-im(i)*bases_in(i); enddo
	 l_out=ext_beg(1); do i=2,dim_num; l_out=l_out+(ext_beg(i)+im(i))*bases_out(i); enddo
	 lb=int(im(1),LONGINT); le=int(slice_ext(1)-1,LONGINT); l_in=segs(n)-lb
	 sloop: do while(l_in+lb.lt.segs(n+1))
	  le=min(le,segs(n+1)-1_LONGINT-l_in) !to avoid different threads doing the same work
	  do ll=lb,le; tens(l_out+ll)=slice(l_in+ll); enddo
	  l_in=l_in+le+1_LONGINT; lb=0_LONGINT
	  do i=2,dim_num
	   if(im(i)+1.lt.slice_ext(i)) then
	    im(i)=im(i)+1; l_out=l_out+bases_out(i)
	   else
	    l_out=l_out-im(i)*bases_out(i); im(i)=0
	   endif
	  enddo
	 enddo sloop
!$OMP END PARALLEL
	else
	 ierr=1 !zero-rank tensor
	endif
!	write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_insert_dlf_c4): kernel time/error code: ",F10.4,1x,i3)') &
!        thread_wtime(time_beg),ierr !debug
	return
	end subroutine tensor_block_insert_dlf_c4
!------------------------------------------------------------------------------------------------
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: tensor_block_insert_dlf_c8
#endif
	subroutine tensor_block_insert_dlf_c8(dim_num,tens,tens_ext,slice,slice_ext,ext_beg,ierr) !PARALLEL
!This subroutine inserts a slice into a tensor block.
!INPUT:
! - dim_num - number of tensor dimensions;
! - tens_ext(1:dim_num) - dimension extents for <tens>;
! - slice(0:) - slice (array);
! - slice_ext(1:dim_num) - dimension extents for <slice>;
! - ext_beg(1:dim_num) - beginning dimension offsets for <tens> (numeration starts at 0);
!OUTPUT:
! - tens(0:) - tensor block (array);
! - ierr - error code (0:success).
!NOTES:
! - No argument validity checks.
	implicit none
!---------------------------------------
	integer, parameter:: real_kind=8
!---------------------------------------
	integer, intent(in):: dim_num,tens_ext(1:dim_num),slice_ext(1:dim_num),ext_beg(1:dim_num)
	complex(real_kind), intent(in):: slice(0:*)
	complex(real_kind), intent(inout):: tens(0:*)
	integer, intent(inout):: ierr
	integer i,j,k,l,m,n,ks,kf,im(1:dim_num)
	integer(LONGINT):: lts,lss,l_in,l_out,lb,le,ll,bases_in(1:dim_num),bases_out(1:dim_num),segs(0:MAX_THREADS) !`Is segs(:) threadsafe?
	real(8) time_beg
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: real_kind
!DIR$ ATTRIBUTES ALIGN:128:: real_kind,im,bases_in,bases_out,segs
#endif

	ierr=0
!	time_beg=thread_wtime() !debug
	if(dim_num.gt.0) then
	 lts=1_LONGINT; do i=1,dim_num; bases_out(i)=lts; lts=lts*tens_ext(i); enddo !tensor block indexing bases
	 lss=1_LONGINT; do i=1,dim_num; bases_in(i)=lss; lss=lss*slice_ext(i); enddo !slice indexing bases
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,m,n,im,l_in,l_out,lb,le,ll)
#ifndef NO_OMP
	 n=omp_get_thread_num(); m=omp_get_num_threads()
#else
	 n=0; m=1
#endif
!$OMP MASTER
	 segs(0)=0_LONGINT; call divide_segment(lss,int(m,LONGINT),segs(1:),ierr); do i=2,m; segs(i)=segs(i)+segs(i-1); enddo
!$OMP END MASTER
!$OMP BARRIER
!$OMP FLUSH(segs)
	 l_in=segs(n); do i=dim_num,1,-1; im(i)=l_in/bases_in(i); l_in=l_in-im(i)*bases_in(i); enddo
	 l_out=ext_beg(1); do i=2,dim_num; l_out=l_out+(ext_beg(i)+im(i))*bases_out(i); enddo
	 lb=int(im(1),LONGINT); le=int(slice_ext(1)-1,LONGINT); l_in=segs(n)-lb
	 sloop: do while(l_in+lb.lt.segs(n+1))
	  le=min(le,segs(n+1)-1_LONGINT-l_in) !to avoid different threads doing the same work
	  do ll=lb,le; tens(l_out+ll)=slice(l_in+ll); enddo
	  l_in=l_in+le+1_LONGINT; lb=0_LONGINT
	  do i=2,dim_num
	   if(im(i)+1.lt.slice_ext(i)) then
	    im(i)=im(i)+1; l_out=l_out+bases_out(i)
	   else
	    l_out=l_out-im(i)*bases_out(i); im(i)=0
	   endif
	  enddo
	 enddo sloop
!$OMP END PARALLEL
	else
	 ierr=1 !zero-rank tensor
	endif
!	write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_insert_dlf_c8): kernel time/error code: ",F10.4,1x,i3)') &
!        thread_wtime(time_beg),ierr !debug
	return
	end subroutine tensor_block_insert_dlf_c8
!------------------------------------------------------------------------------------------------
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: tensor_block_copy_dlf_r4
#endif
	subroutine tensor_block_copy_dlf_r4(dim_num,dim_extents,dim_transp,tens_in,tens_out,ierr) !PARALLEL
!Given a dense tensor block, this subroutine makes a copy of it, permuting the indices according to the <dim_transp>.
!The algorithm is cache-efficient (Author: Dmitry I. Lyakh (Liakh): quant4me@gmail.com) (C) 2014.
!INPUT:
! - dim_num - number of dimensions (>0);
! - dim_extents(1:dim_num) - dimension extents;
! - dim_transp(0:dim_num) - index permutation (O2N), dim_transp(0) is the sign of the permutation;
! - tens_in(0:) - input tensor data;
!OUTPUT:
! - tens_out(0:) - output (possibly transposed) tensor data;
! - ierr - error code (0:success).
	implicit none
!---------------------------------------
	integer, parameter:: real_kind=4
	logical, parameter:: cache_efficiency=.TRUE.
	integer(LONGINT), parameter:: cache_line_len=64/real_kind     !cache line length (words)
	integer(LONGINT), parameter:: cache_line_min=cache_line_len*2 !lower bound for the input/output minor volume: => L1_cache_line*2
	integer(LONGINT), parameter:: cache_line_lim=cache_line_len*4 !upper bound for the input/output minor volume: <= SQRT(L1_size)
	integer(LONGINT), parameter:: small_tens_size=2**10 !up to this size it is useless to apply cache efficiency (fully fits in L1)
	integer(LONGINT), parameter:: vec_size=2**8 !loop reorganization parameter for direct copy
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: real_kind,cache_efficiency,cache_line_len,cache_line_min,cache_line_lim,small_tens_size,vec_size
!DIR$ ATTRIBUTES ALIGN:128:: real_kind,cache_efficiency,cache_line_len,cache_line_min,cache_line_lim,small_tens_size,vec_size
#endif
!---------------------------------------------------------------------
	integer, intent(in):: dim_num,dim_extents(1:*),dim_transp(0:*)
	real(real_kind), intent(in):: tens_in(0:*)
	real(real_kind), intent(out):: tens_out(0:*)
	integer, intent(inout):: ierr
	integer i,j,k,l,m,n,k1,k2,ks,kf,split_in,split_out
	integer im(1:dim_num),n2o(0:dim_num+1),ipr(1:dim_num+1),dim_beg(1:dim_num),dim_end(1:dim_num)
	integer(LONGINT) bases_in(1:dim_num+1),bases_out(1:dim_num+1),bases_pri(1:dim_num+1),segs(0:MAX_THREADS) !`Is segs(:) threadsafe?
	integer(LONGINT) bs,l0,l1,l2,l3,ll,lb,le,ls,l_in,l_out,seg_in,seg_out,vol_min,vol_ext
	logical trivial
	real(8) time_beg,tm
#ifndef NO_PHI
!DIR$ ATTRIBUTES ALIGN:128:: im,n2o,ipr,dim_beg,dim_end,bases_in,bases_out,bases_pri,segs
#endif
	ierr=0
	time_beg=thread_wtime() !debug
	if(dim_num.lt.0) then; ierr=1; return; elseif(dim_num.eq.0) then; tens_out(0)=tens_in(0); return; endif
!Check the index permutation:
	trivial=.TRUE.; do i=1,dim_num; if(dim_transp(i).ne.i) then; trivial=.FALSE.; exit; endif; enddo
	if(trivial.and.cache_efficiency) then
!Trivial index permutation (no permutation):
 !Compute indexing bases:
	 bs=1_LONGINT; do i=1,dim_num; bases_in(i)=bs; bs=bs*dim_extents(i); enddo
 !Copy input to output:
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(l0,l1)
!$OMP DO SCHEDULE(GUIDED)
	 do l0=0_LONGINT,bs-1_LONGINT-mod(bs,vec_size),vec_size
	  do l1=0_LONGINT,vec_size-1_LONGINT; tens_out(l0+l1)=tens_in(l0+l1); enddo
	 enddo
!$OMP END DO NOWAIT
!$OMP SINGLE
	 do l0=bs-mod(bs,vec_size),bs-1_LONGINT; tens_out(l0)=tens_in(l0); enddo
!$OMP END SINGLE
!$OMP END PARALLEL
	else
!Non-trivial index permutation:
 !Compute indexing bases:
	 do i=1,dim_num; n2o(dim_transp(i))=i; enddo; n2o(dim_num+1)=dim_num+1 !get the N2O
	 bs=1_LONGINT; do i=1,dim_num; bases_in(i)=bs; bs=bs*dim_extents(i); enddo; bases_in(dim_num+1)=bs
	 bs=1_LONGINT; do i=1,dim_num; bases_out(n2o(i))=bs; bs=bs*dim_extents(n2o(i)); enddo; bases_out(dim_num+1)=bs
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
	   k1=k1+1; split_in=k1; seg_in=(cache_line_lim-1_LONGINT)/bases_in(split_in)+1_LONGINT
	   split_out=n2o(k2); seg_out=dim_extents(split_out)
	  elseif(bases_in(k1+1).ge.cache_line_min.and.bases_out(n2o(k2+1)).lt.cache_line_min) then !split the last minor output dim
	   k2=k2+1; split_in=n2o(k2); seg_in=(cache_line_lim-1_LONGINT)/bases_out(split_in)+1_LONGINT
	   split_out=k1; seg_out=dim_extents(split_out)
	  elseif(bases_in(k1+1).lt.cache_line_min.and.bases_out(n2o(k2+1)).lt.cache_line_min) then !split both
	   k1=k1+1; k2=k2+1
	   if(k1.eq.n2o(k2)) then
	    split_in=k1; seg_in=(cache_line_lim-1_LONGINT)/min(bases_in(split_in),bases_out(split_in))+1_LONGINT
	    split_out=k1; seg_out=dim_extents(split_out)
	   else
	    split_in=k1; seg_in=(cache_line_lim-1_LONGINT)/bases_in(split_in)+1_LONGINT
	    split_out=n2o(k2); seg_out=(cache_line_lim-1_LONGINT)/bases_out(split_out)+1_LONGINT
	   endif
	  else !split none
	   split_in=k1; seg_in=dim_extents(split_in)
	   split_out=n2o(k2); seg_out=dim_extents(split_out)
	  endif
	  vol_min=1_LONGINT
	  if(seg_in.lt.dim_extents(split_in)) vol_min=vol_min*seg_in
	  if(seg_out.lt.dim_extents(split_out)) vol_min=vol_min*seg_out
	  if(vol_min.gt.1_LONGINT) then
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
	      seg_in=min(seg_in*l,int(dim_extents(split_in),LONGINT))
	      seg_out=min(seg_out*l,int(dim_extents(split_out),LONGINT))
	     else
	      seg_in=min(seg_in*l,int(dim_extents(split_in),LONGINT))
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
	 vol_ext=1_LONGINT; do j=kf+1,dim_num; vol_ext=vol_ext*dim_extents(ipr(j)); enddo !external volume
!	 write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_copy_dlf_r4): extents:",99(1x,i5))') dim_extents(1:dim_num) !debug
!	 write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_copy_dlf_r4): permutation:",99(1x,i2))') dim_transp(1:dim_num) !debug
!	 write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_copy_dlf_r4): minor ",i3,": priority:",99(1x,i2))') &
!         kf,ipr(1:dim_num) !debug
!        write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_copy_dlf_r4): vol_ext ",i11,": segs:",4(1x,i5))') &
!         vol_ext,split_in,split_out,seg_in,seg_out !debug
 !Transpose:
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,j,m,n,ks,l0,l1,l2,l3,ll,lb,le,ls,l_in,l_out,vol_min,im,dim_beg,dim_end)
#ifndef NO_OMP
	 n=omp_get_thread_num(); m=omp_get_num_threads() !multi-threaded execution
#else
	 n=0; m=1 !serial execution
#endif
!	 if(n.eq.0) write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_copy_dlf_r4): number of threads = ",i5)') m !debug
         if(kf.lt.dim_num) then !external indices present
!$OMP MASTER
	  segs(0)=0_LONGINT; call divide_segment(vol_ext,int(m,LONGINT),segs(1:),i); do j=2,m; segs(j)=segs(j)+segs(j-1); enddo
	  l0=1_LONGINT; do i=kf+1,dim_num; bases_pri(ipr(i))=l0; l0=l0*dim_extents(ipr(i)); enddo !priority bases
!$OMP END MASTER
!$OMP BARRIER
!$OMP FLUSH(segs,bases_pri)
	  dim_beg(1:dim_num)=0; dim_end(1:dim_num)=dim_extents(1:dim_num)-1
          l2=dim_end(split_in); l3=dim_end(split_out); ls=bases_out(1)
	  loop0: do l1=0_LONGINT,l3,seg_out !output dimension
	   dim_beg(split_out)=l1; dim_end(split_out)=min(l1+seg_out-1_LONGINT,l3)
	   do l0=0_LONGINT,l2,seg_in !input dimension
	    dim_beg(split_in)=l0; dim_end(split_in)=min(l0+seg_in-1_LONGINT,l2)
	    ll=segs(n); do i=dim_num,kf+1,-1; j=ipr(i); im(j)=ll/bases_pri(j); ll=ll-im(j)*bases_pri(j); enddo
	    vol_min=1_LONGINT; do i=1,kf; j=ipr(i); vol_min=vol_min*(dim_end(j)-dim_beg(j)+1); im(j)=dim_beg(j); enddo
	    l_in=0_LONGINT; do j=1,dim_num; l_in=l_in+im(j)*bases_in(j); enddo
	    l_out=0_LONGINT; do j=1,dim_num; l_out=l_out+im(j)*bases_out(j); enddo
	    le=dim_end(1)-dim_beg(1); lb=(segs(n+1)-segs(n))*vol_min; ks=0
	    loop1: do while(lb.gt.0_LONGINT)
	     do ll=0_LONGINT,le
	      tens_out(l_out+ll*ls)=tens_in(l_in+ll)
	     enddo
	     lb=lb-(le+1_LONGINT)
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
	    if(lb.ne.0_LONGINT) then
	     if(VERBOSE) write(CONS_OUT,'("ERROR(tensor_algebra::tensor_block_copy_dlf_r4): invalid remainder: ",i11,1x,i4)') lb,n
!!!$OMP ATOMIC WRITE SEQ_CST
!$OMP ATOMIC WRITE
	     ierr=2
	     exit loop0
	    endif
	   enddo !l0
          enddo loop0 !l1
         else !external indices absent
!$OMP MASTER
	  l0=1_LONGINT; do i=kf+1,dim_num; bases_pri(ipr(i))=l0; l0=l0*dim_extents(ipr(i)); enddo !priority bases
!$OMP END MASTER
!$OMP BARRIER
!$OMP FLUSH(bases_pri)
	  dim_beg(1:dim_num)=0; dim_end(1:dim_num)=dim_extents(1:dim_num)-1
          l2=dim_end(split_in); l3=dim_end(split_out); ls=bases_out(1)
!$OMP DO SCHEDULE(DYNAMIC) COLLAPSE(2)
	  do l1=0_LONGINT,l3,seg_out !output dimension
	   do l0=0_LONGINT,l2,seg_in !input dimension
	    dim_beg(split_out)=l1; dim_end(split_out)=min(l1+seg_out-1_LONGINT,l3)
	    dim_beg(split_in)=l0; dim_end(split_in)=min(l0+seg_in-1_LONGINT,l2)
	    vol_min=1_LONGINT; do i=1,kf; j=ipr(i); vol_min=vol_min*(dim_end(j)-dim_beg(j)+1); im(j)=dim_beg(j); enddo
	    l_in=0_LONGINT; do j=1,dim_num; l_in=l_in+im(j)*bases_in(j); enddo
	    l_out=0_LONGINT; do j=1,dim_num; l_out=l_out+im(j)*bases_out(j); enddo
	    le=dim_end(1)-dim_beg(1); lb=vol_min; ks=0
	    loop2: do while(lb.gt.0_LONGINT)
	     do ll=0_LONGINT,le
	      tens_out(l_out+ll*ls)=tens_in(l_in+ll)
	     enddo
	     lb=lb-(le+1_LONGINT)
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
	tm=thread_wtime(time_beg) !debug
	if(LOGGING.gt.0) then
	 write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_copy_dlf_r4): Done: ",F10.4," sec, ",F10.4," GB/s, error ",i3)') &
	 tm,dble(2_LONGINT*bs*real_kind)/(tm*1024d0*1024d0*1024d0),ierr !debug
	endif
	return
	end subroutine tensor_block_copy_dlf_r4
!------------------------------------------------------------------------------------------------
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: tensor_block_copy_dlf_r8
#endif
	subroutine tensor_block_copy_dlf_r8(dim_num,dim_extents,dim_transp,tens_in,tens_out,ierr) !PARALLEL
!Given a dense tensor block, this subroutine makes a copy of it, permuting the indices according to the <dim_transp>.
!The algorithm is cache-efficient (Author: Dmitry I. Lyakh (Liakh): quant4me@gmail.com) (C) 2014.
!INPUT:
! - dim_num - number of dimensions (>0);
! - dim_extents(1:dim_num) - dimension extents;
! - dim_transp(0:dim_num) - index permutation (O2N), dim_transp(0) is the sign of the permutation;
! - tens_in(0:) - input tensor data;
!OUTPUT:
! - tens_out(0:) - output (possibly transposed) tensor data;
! - ierr - error code (0:success).
	implicit none
!---------------------------------------
	integer, parameter:: real_kind=8
	logical, parameter:: cache_efficiency=.TRUE.
	integer(LONGINT), parameter:: cache_line_len=64/real_kind     !cache line length (words)
	integer(LONGINT), parameter:: cache_line_min=cache_line_len*2 !lower bound for the input/output minor volume: => L1_cache_line*2
	integer(LONGINT), parameter:: cache_line_lim=cache_line_len*4 !upper bound for the input/output minor volume: <= SQRT(L1_size)
	integer(LONGINT), parameter:: small_tens_size=2**10 !up to this size it is useless to apply cache efficiency (fully fits in L1)
	integer(LONGINT), parameter:: vec_size=2**8 !loop reorganization parameter for direct copy
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: real_kind,cache_efficiency,cache_line_len,cache_line_min,cache_line_lim,small_tens_size,vec_size
!DIR$ ATTRIBUTES ALIGN:128:: real_kind,cache_efficiency,cache_line_len,cache_line_min,cache_line_lim,small_tens_size,vec_size
#endif
!---------------------------------------------------------------------
	integer, intent(in):: dim_num,dim_extents(1:*),dim_transp(0:*)
	real(real_kind), intent(in):: tens_in(0:*)
	real(real_kind), intent(out):: tens_out(0:*)
	integer, intent(inout):: ierr
	integer i,j,k,l,m,n,k1,k2,ks,kf,split_in,split_out
	integer im(1:dim_num),n2o(0:dim_num+1),ipr(1:dim_num+1),dim_beg(1:dim_num),dim_end(1:dim_num)
	integer(LONGINT) bases_in(1:dim_num+1),bases_out(1:dim_num+1),bases_pri(1:dim_num+1),segs(0:MAX_THREADS) !`Is segs(:) threadsafe?
	integer(LONGINT) bs,l0,l1,l2,l3,ll,lb,le,ls,l_in,l_out,seg_in,seg_out,vol_min,vol_ext
	logical trivial
	real(8) time_beg,tm
#ifndef NO_PHI
!DIR$ ATTRIBUTES ALIGN:128:: im,n2o,ipr,dim_beg,dim_end,bases_in,bases_out,bases_pri,segs
#endif
	ierr=0
	time_beg=thread_wtime() !debug
	if(dim_num.lt.0) then; ierr=1; return; elseif(dim_num.eq.0) then; tens_out(0)=tens_in(0); return; endif
!Check the index permutation:
	trivial=.TRUE.; do i=1,dim_num; if(dim_transp(i).ne.i) then; trivial=.FALSE.; exit; endif; enddo
	if(trivial.and.cache_efficiency) then
!Trivial index permutation (no permutation):
 !Compute indexing bases:
	 bs=1_LONGINT; do i=1,dim_num; bases_in(i)=bs; bs=bs*dim_extents(i); enddo
 !Copy input to output:
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(l0,l1)
!$OMP DO SCHEDULE(GUIDED)
	 do l0=0_LONGINT,bs-1_LONGINT-mod(bs,vec_size),vec_size
	  do l1=0_LONGINT,vec_size-1_LONGINT; tens_out(l0+l1)=tens_in(l0+l1); enddo
	 enddo
!$OMP END DO NOWAIT
!$OMP SINGLE
	 do l0=bs-mod(bs,vec_size),bs-1_LONGINT; tens_out(l0)=tens_in(l0); enddo
!$OMP END SINGLE
!$OMP END PARALLEL
	else
!Non-trivial index permutation:
 !Compute indexing bases:
	 do i=1,dim_num; n2o(dim_transp(i))=i; enddo; n2o(dim_num+1)=dim_num+1 !get the N2O
	 bs=1_LONGINT; do i=1,dim_num; bases_in(i)=bs; bs=bs*dim_extents(i); enddo; bases_in(dim_num+1)=bs
	 bs=1_LONGINT; do i=1,dim_num; bases_out(n2o(i))=bs; bs=bs*dim_extents(n2o(i)); enddo; bases_out(dim_num+1)=bs
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
	   k1=k1+1; split_in=k1; seg_in=(cache_line_lim-1_LONGINT)/bases_in(split_in)+1_LONGINT
	   split_out=n2o(k2); seg_out=dim_extents(split_out)
	  elseif(bases_in(k1+1).ge.cache_line_min.and.bases_out(n2o(k2+1)).lt.cache_line_min) then !split the last minor output dim
	   k2=k2+1; split_in=n2o(k2); seg_in=(cache_line_lim-1_LONGINT)/bases_out(split_in)+1_LONGINT
	   split_out=k1; seg_out=dim_extents(split_out)
	  elseif(bases_in(k1+1).lt.cache_line_min.and.bases_out(n2o(k2+1)).lt.cache_line_min) then !split both
	   k1=k1+1; k2=k2+1
	   if(k1.eq.n2o(k2)) then
	    split_in=k1; seg_in=(cache_line_lim-1_LONGINT)/min(bases_in(split_in),bases_out(split_in))+1_LONGINT
	    split_out=k1; seg_out=dim_extents(split_out)
	   else
	    split_in=k1; seg_in=(cache_line_lim-1_LONGINT)/bases_in(split_in)+1_LONGINT
	    split_out=n2o(k2); seg_out=(cache_line_lim-1_LONGINT)/bases_out(split_out)+1_LONGINT
	   endif
	  else !split none
	   split_in=k1; seg_in=dim_extents(split_in)
	   split_out=n2o(k2); seg_out=dim_extents(split_out)
	  endif
	  vol_min=1_LONGINT
	  if(seg_in.lt.dim_extents(split_in)) vol_min=vol_min*seg_in
	  if(seg_out.lt.dim_extents(split_out)) vol_min=vol_min*seg_out
	  if(vol_min.gt.1_LONGINT) then
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
	      seg_in=min(seg_in*l,int(dim_extents(split_in),LONGINT))
	      seg_out=min(seg_out*l,int(dim_extents(split_out),LONGINT))
	     else
	      seg_in=min(seg_in*l,int(dim_extents(split_in),LONGINT))
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
	 vol_ext=1_LONGINT; do j=kf+1,dim_num; vol_ext=vol_ext*dim_extents(ipr(j)); enddo !external volume
!	 write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_copy_dlf_r8): extents:",99(1x,i5))') dim_extents(1:dim_num) !debug
!	 write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_copy_dlf_r8): permutation:",99(1x,i2))') dim_transp(1:dim_num) !debug
!	 write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_copy_dlf_r8): minor ",i3,": priority:",99(1x,i2))') &
!         kf,ipr(1:dim_num) !debug
!        write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_copy_dlf_r8): vol_ext ",i11,": segs:",4(1x,i5))') &
!         vol_ext,split_in,split_out,seg_in,seg_out !debug
 !Transpose:
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,j,m,n,ks,l0,l1,l2,l3,ll,lb,le,ls,l_in,l_out,vol_min,im,dim_beg,dim_end)
#ifndef NO_OMP
	 n=omp_get_thread_num(); m=omp_get_num_threads() !multi-threaded execution
#else
	 n=0; m=1 !serial execution
#endif
!	 if(n.eq.0) write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_copy_dlf_r8): number of threads = ",i5)') m !debug
	 if(kf.lt.dim_num) then !external indices present
!$OMP MASTER
	  segs(0)=0_LONGINT; call divide_segment(vol_ext,int(m,LONGINT),segs(1:),i); do j=2,m; segs(j)=segs(j)+segs(j-1); enddo
	  l0=1_LONGINT; do i=kf+1,dim_num; bases_pri(ipr(i))=l0; l0=l0*dim_extents(ipr(i)); enddo !priority bases
!$OMP END MASTER
!$OMP BARRIER
!$OMP FLUSH(segs,bases_pri)
	  dim_beg(1:dim_num)=0; dim_end(1:dim_num)=dim_extents(1:dim_num)-1
	  l2=dim_end(split_in); l3=dim_end(split_out); ls=bases_out(1)
	  loop0: do l1=0_LONGINT,l3,seg_out !output dimension
	   dim_beg(split_out)=l1; dim_end(split_out)=min(l1+seg_out-1_LONGINT,l3)
	   do l0=0_LONGINT,l2,seg_in !input dimension
	    dim_beg(split_in)=l0; dim_end(split_in)=min(l0+seg_in-1_LONGINT,l2)
	    ll=segs(n); do i=dim_num,kf+1,-1; j=ipr(i); im(j)=ll/bases_pri(j); ll=ll-im(j)*bases_pri(j); enddo
	    vol_min=1_LONGINT; do i=1,kf; j=ipr(i); vol_min=vol_min*(dim_end(j)-dim_beg(j)+1); im(j)=dim_beg(j); enddo
	    l_in=0_LONGINT; do j=1,dim_num; l_in=l_in+im(j)*bases_in(j); enddo
	    l_out=0_LONGINT; do j=1,dim_num; l_out=l_out+im(j)*bases_out(j); enddo
	    le=dim_end(1)-dim_beg(1); lb=(segs(n+1)-segs(n))*vol_min; ks=0
	    loop1: do while(lb.gt.0_LONGINT)
	     do ll=0_LONGINT,le
	      tens_out(l_out+ll*ls)=tens_in(l_in+ll)
	     enddo
	     lb=lb-(le+1_LONGINT)
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
	    if(lb.ne.0_LONGINT) then
	     if(VERBOSE) write(CONS_OUT,'("ERROR(tensor_algebra::tensor_block_copy_dlf_r8): invalid remainder: ",i11,1x,i4)') lb,n
!!!$OMP ATOMIC WRITE SEQ_CST
!$OMP ATOMIC WRITE
	     ierr=2
	     exit loop0
	    endif
	   enddo !l0
          enddo loop0 !l1
         else !external indices absent
!$OMP MASTER
	  l0=1_LONGINT; do i=kf+1,dim_num; bases_pri(ipr(i))=l0; l0=l0*dim_extents(ipr(i)); enddo !priority bases
!$OMP END MASTER
!$OMP BARRIER
!$OMP FLUSH(bases_pri)
	  dim_beg(1:dim_num)=0; dim_end(1:dim_num)=dim_extents(1:dim_num)-1
          l2=dim_end(split_in); l3=dim_end(split_out); ls=bases_out(1)
!$OMP DO SCHEDULE(DYNAMIC) COLLAPSE(2)
	  do l1=0_LONGINT,l3,seg_out !output dimension
	   do l0=0_LONGINT,l2,seg_in !input dimension
	    dim_beg(split_out)=l1; dim_end(split_out)=min(l1+seg_out-1_LONGINT,l3)
	    dim_beg(split_in)=l0; dim_end(split_in)=min(l0+seg_in-1_LONGINT,l2)
	    vol_min=1_LONGINT; do i=1,kf; j=ipr(i); vol_min=vol_min*(dim_end(j)-dim_beg(j)+1); im(j)=dim_beg(j); enddo
	    l_in=0_LONGINT; do j=1,dim_num; l_in=l_in+im(j)*bases_in(j); enddo
	    l_out=0_LONGINT; do j=1,dim_num; l_out=l_out+im(j)*bases_out(j); enddo
	    le=dim_end(1)-dim_beg(1); lb=vol_min; ks=0
	    loop2: do while(lb.gt.0_LONGINT)
	     do ll=0_LONGINT,le
	      tens_out(l_out+ll*ls)=tens_in(l_in+ll)
	     enddo
	     lb=lb-(le+1_LONGINT)
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
	tm=thread_wtime(time_beg) !debug
	if(LOGGING.gt.0) then
	 write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_copy_dlf_r8): Done: ",F10.4," sec, ",F10.4," GB/s, error ",i3)')&
	 &tm,dble(2_LONGINT*bs*real_kind)/(tm*1024d0*1024d0*1024d0),ierr !debug
	endif
	return
	end subroutine tensor_block_copy_dlf_r8
!-------------------------------------------------------------------------------------------------------
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: tensor_block_copy_dlf_c4
#endif
	subroutine tensor_block_copy_dlf_c4(dim_num,dim_extents,dim_transp,tens_in,tens_out,ierr,conjug) !PARALLEL
!Given a dense tensor block, this subroutine makes a copy of it, permuting the indices according to the <dim_transp>.
!The algorithm is cache-efficient (Author: Dmitry I. Lyakh (Liakh): quant4me@gmail.com) (C) 2014.
!INPUT:
! - dim_num - number of dimensions (>0);
! - dim_extents(1:dim_num) - dimension extents;
! - dim_transp(0:dim_num) - index permutation (O2N), dim_transp(0) is the sign of the permutation;
! - tens_in(0:) - input tensor data;
! - conjug - (optional) complex conjugation flag;
!OUTPUT:
! - tens_out(0:) - output (possibly transposed) tensor data;
! - ierr - error code (0:success).
	implicit none
!---------------------------------------
	integer, parameter:: real_kind=4
	logical, parameter:: cache_efficiency=.TRUE.
	integer(LONGINT), parameter:: cache_line_len=64/(real_kind*2) !cache line length (words)
	integer(LONGINT), parameter:: cache_line_min=cache_line_len*2 !lower bound for the input/output minor volume: => L1_cache_line*2
	integer(LONGINT), parameter:: cache_line_lim=cache_line_len*4 !upper bound for the input/output minor volume: <= SQRT(L1_size)
	integer(LONGINT), parameter:: small_tens_size=2**10 !up to this size it is useless to apply cache efficiency (fully fits in L1)
	integer(LONGINT), parameter:: vec_size=2**8 !loop reorganization parameter for direct copy
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: real_kind,cache_efficiency,cache_line_len,cache_line_min,cache_line_lim,small_tens_size,vec_size
!DIR$ ATTRIBUTES ALIGN:128:: real_kind,cache_efficiency,cache_line_len,cache_line_min,cache_line_lim,small_tens_size,vec_size
#endif
!---------------------------------------------------------------------
	integer, intent(in):: dim_num,dim_extents(1:*),dim_transp(0:*)
	complex(real_kind), intent(in):: tens_in(0:*)
	complex(real_kind), intent(out):: tens_out(0:*)
	integer, intent(inout):: ierr
	logical, intent(in), optional:: conjug
	integer i,j,k,l,m,n,k1,k2,ks,kf,split_in,split_out
	integer im(1:dim_num),n2o(0:dim_num+1),ipr(1:dim_num+1),dim_beg(1:dim_num),dim_end(1:dim_num)
	integer(LONGINT) bases_in(1:dim_num+1),bases_out(1:dim_num+1),bases_pri(1:dim_num+1),segs(0:MAX_THREADS) !`Is segs(:) threadsafe?
	integer(LONGINT) bs,l0,l1,l2,l3,ll,lb,le,ls,l_in,l_out,seg_in,seg_out,vol_min,vol_ext
	logical trivial,conj
	real(8) time_beg,tm
#ifndef NO_PHI
!DIR$ ATTRIBUTES ALIGN:128:: im,n2o,ipr,dim_beg,dim_end,bases_in,bases_out,bases_pri,segs
#endif
	ierr=0
	time_beg=thread_wtime() !debug
	if(present(conjug)) then; conj=conjug; else; conj=.FALSE.; endif !optional complex conjugation
	if(dim_num.lt.0) then
	 ierr=1; return
	elseif(dim_num.eq.0) then
	 if(conj) then; tens_out(0)=conjg(tens_in(0)); else; tens_out(0)=tens_in(0); endif
	 return
	endif
!Check the index permutation:
	trivial=.TRUE.; do i=1,dim_num; if(dim_transp(i).ne.i) then; trivial=.FALSE.; exit; endif; enddo
	if(trivial.and.cache_efficiency) then
!Trivial index permutation (no permutation):
 !Compute indexing bases:
	 bs=1_LONGINT; do i=1,dim_num; bases_in(i)=bs; bs=bs*dim_extents(i); enddo
 !Copy input to output:
	 if(conj) then
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(l0,l1)
!$OMP DO SCHEDULE(GUIDED)
	  do l0=0_LONGINT,bs-1_LONGINT-mod(bs,vec_size),vec_size
	   do l1=0_LONGINT,vec_size-1_LONGINT; tens_out(l0+l1)=conjg(tens_in(l0+l1)); enddo
	  enddo
!$OMP END DO NOWAIT
!$OMP SINGLE
	  do l0=bs-mod(bs,vec_size),bs-1_LONGINT; tens_out(l0)=conjg(tens_in(l0)); enddo
!$OMP END SINGLE
!$OMP END PARALLEL
	 else
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(l0,l1)
!$OMP DO SCHEDULE(GUIDED)
	  do l0=0_LONGINT,bs-1_LONGINT-mod(bs,vec_size),vec_size
	   do l1=0_LONGINT,vec_size-1_LONGINT; tens_out(l0+l1)=tens_in(l0+l1); enddo
	  enddo
!$OMP END DO NOWAIT
!$OMP SINGLE
	  do l0=bs-mod(bs,vec_size),bs-1_LONGINT; tens_out(l0)=tens_in(l0); enddo
!$OMP END SINGLE
!$OMP END PARALLEL
	 endif
	else
!Non-trivial index permutation:
 !Compute indexing bases:
	 do i=1,dim_num; n2o(dim_transp(i))=i; enddo; n2o(dim_num+1)=dim_num+1 !get the N2O
	 bs=1_LONGINT; do i=1,dim_num; bases_in(i)=bs; bs=bs*dim_extents(i); enddo; bases_in(dim_num+1)=bs
	 bs=1_LONGINT; do i=1,dim_num; bases_out(n2o(i))=bs; bs=bs*dim_extents(n2o(i)); enddo; bases_out(dim_num+1)=bs
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
	   k1=k1+1; split_in=k1; seg_in=(cache_line_lim-1_LONGINT)/bases_in(split_in)+1_LONGINT
	   split_out=n2o(k2); seg_out=dim_extents(split_out)
	  elseif(bases_in(k1+1).ge.cache_line_min.and.bases_out(n2o(k2+1)).lt.cache_line_min) then !split the last minor output dim
	   k2=k2+1; split_in=n2o(k2); seg_in=(cache_line_lim-1_LONGINT)/bases_out(split_in)+1_LONGINT
	   split_out=k1; seg_out=dim_extents(split_out)
	  elseif(bases_in(k1+1).lt.cache_line_min.and.bases_out(n2o(k2+1)).lt.cache_line_min) then !split both
	   k1=k1+1; k2=k2+1
	   if(k1.eq.n2o(k2)) then
	    split_in=k1; seg_in=(cache_line_lim-1_LONGINT)/min(bases_in(split_in),bases_out(split_in))+1_LONGINT
	    split_out=k1; seg_out=dim_extents(split_out)
	   else
	    split_in=k1; seg_in=(cache_line_lim-1_LONGINT)/bases_in(split_in)+1_LONGINT
	    split_out=n2o(k2); seg_out=(cache_line_lim-1_LONGINT)/bases_out(split_out)+1_LONGINT
	   endif
	  else !split none
	   split_in=k1; seg_in=dim_extents(split_in)
	   split_out=n2o(k2); seg_out=dim_extents(split_out)
	  endif
	  vol_min=1_LONGINT
	  if(seg_in.lt.dim_extents(split_in)) vol_min=vol_min*seg_in
	  if(seg_out.lt.dim_extents(split_out)) vol_min=vol_min*seg_out
	  if(vol_min.gt.1_LONGINT) then
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
	      seg_in=min(seg_in*l,int(dim_extents(split_in),LONGINT))
	      seg_out=min(seg_out*l,int(dim_extents(split_out),LONGINT))
	     else
	      seg_in=min(seg_in*l,int(dim_extents(split_in),LONGINT))
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
	 vol_ext=1_LONGINT; do j=kf+1,dim_num; vol_ext=vol_ext*dim_extents(ipr(j)); enddo !external volume
!	 write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_copy_dlf_c4): extents:",99(1x,i5))') dim_extents(1:dim_num) !debug
!	 write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_copy_dlf_c4): permutation:",99(1x,i2))') dim_transp(1:dim_num) !debug
!	 write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_copy_dlf_c4): minor ",i3,": priority:",99(1x,i2))') &
!         kf,ipr(1:dim_num) !debug
!        write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_copy_dlf_c4): vol_ext ",i11,": segs:",4(1x,i5))') &
!         vol_ext,split_in,split_out,seg_in,seg_out !debug
 !Transpose:
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,j,m,n,ks,l0,l1,l2,l3,ll,lb,le,ls,l_in,l_out,vol_min,im,dim_beg,dim_end)
#ifndef NO_OMP
	 n=omp_get_thread_num(); m=omp_get_num_threads() !multi-threaded execution
#else
	 n=0; m=1 !serial execution
#endif
!	 if(n.eq.0) write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_copy_dlf_c4): number of threads = ",i5)') m !debug
         if(kf.lt.dim_num) then !external indices present
!$OMP MASTER
	  segs(0)=0_LONGINT; call divide_segment(vol_ext,int(m,LONGINT),segs(1:),i); do j=2,m; segs(j)=segs(j)+segs(j-1); enddo
	  l0=1_LONGINT; do i=kf+1,dim_num; bases_pri(ipr(i))=l0; l0=l0*dim_extents(ipr(i)); enddo !priority bases
!$OMP END MASTER
!$OMP BARRIER
!$OMP FLUSH(segs,bases_pri)
	  dim_beg(1:dim_num)=0; dim_end(1:dim_num)=dim_extents(1:dim_num)-1
	  l2=dim_end(split_in); l3=dim_end(split_out); ls=bases_out(1)
	  loop0: do l1=0_LONGINT,l3,seg_out !output dimension
	   dim_beg(split_out)=l1; dim_end(split_out)=min(l1+seg_out-1_LONGINT,l3)
	   do l0=0_LONGINT,l2,seg_in !input dimension
	    dim_beg(split_in)=l0; dim_end(split_in)=min(l0+seg_in-1_LONGINT,l2)
	    ll=segs(n); do i=dim_num,kf+1,-1; j=ipr(i); im(j)=ll/bases_pri(j); ll=ll-im(j)*bases_pri(j); enddo
	    vol_min=1_LONGINT; do i=1,kf; j=ipr(i); vol_min=vol_min*(dim_end(j)-dim_beg(j)+1); im(j)=dim_beg(j); enddo
	    l_in=0_LONGINT; do j=1,dim_num; l_in=l_in+im(j)*bases_in(j); enddo
	    l_out=0_LONGINT; do j=1,dim_num; l_out=l_out+im(j)*bases_out(j); enddo
	    le=dim_end(1)-dim_beg(1); lb=(segs(n+1)-segs(n))*vol_min; ks=0
	    loop1: do while(lb.gt.0_LONGINT)
	     if(conj) then
	      do ll=0_LONGINT,le
	       tens_out(l_out+ll*ls)=conjg(tens_in(l_in+ll))
	      enddo
	     else
	      do ll=0_LONGINT,le
	       tens_out(l_out+ll*ls)=tens_in(l_in+ll)
	      enddo
	     endif
	     lb=lb-(le+1_LONGINT)
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
	    if(lb.ne.0_LONGINT) then
	     if(VERBOSE) write(CONS_OUT,'("ERROR(tensor_algebra::tensor_block_copy_dlf_c4): invalid remainder: ",i11,1x,i4)') lb,n
!!!$OMP ATOMIC WRITE SEQ_CST
!$OMP ATOMIC WRITE
	     ierr=2
	     exit loop0
	    endif
	   enddo !l0
          enddo loop0 !l1
         else !external indices absent
!$OMP MASTER
	  l0=1_LONGINT; do i=kf+1,dim_num; bases_pri(ipr(i))=l0; l0=l0*dim_extents(ipr(i)); enddo !priority bases
!$OMP END MASTER
!$OMP BARRIER
!$OMP FLUSH(bases_pri)
	  dim_beg(1:dim_num)=0; dim_end(1:dim_num)=dim_extents(1:dim_num)-1
          l2=dim_end(split_in); l3=dim_end(split_out); ls=bases_out(1)
!$OMP DO SCHEDULE(DYNAMIC) COLLAPSE(2)
	  do l1=0_LONGINT,l3,seg_out !output dimension
	   do l0=0_LONGINT,l2,seg_in !input dimension
	    dim_beg(split_out)=l1; dim_end(split_out)=min(l1+seg_out-1_LONGINT,l3)
	    dim_beg(split_in)=l0; dim_end(split_in)=min(l0+seg_in-1_LONGINT,l2)
	    vol_min=1_LONGINT; do i=1,kf; j=ipr(i); vol_min=vol_min*(dim_end(j)-dim_beg(j)+1); im(j)=dim_beg(j); enddo
	    l_in=0_LONGINT; do j=1,dim_num; l_in=l_in+im(j)*bases_in(j); enddo
	    l_out=0_LONGINT; do j=1,dim_num; l_out=l_out+im(j)*bases_out(j); enddo
	    le=dim_end(1)-dim_beg(1); lb=vol_min; ks=0
	    loop2: do while(lb.gt.0_LONGINT)
	     if(conj) then
	      do ll=0_LONGINT,le
	       tens_out(l_out+ll*ls)=conjg(tens_in(l_in+ll))
	      enddo
	     else
	      do ll=0_LONGINT,le
	       tens_out(l_out+ll*ls)=tens_in(l_in+ll)
	      enddo
	     endif
	     lb=lb-(le+1_LONGINT)
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
	tm=thread_wtime(time_beg) !debug
	if(LOGGING.gt.0) then
	 write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_copy_dlf_c4): Done: ",F10.4," sec, ",F10.4," GB/s, error ",i3)') &
	 tm,dble(2_LONGINT*bs*real_kind*2_LONGINT)/(tm*1024d0*1024d0*1024d0),ierr !debug
	endif
	return
	end subroutine tensor_block_copy_dlf_c4
!-------------------------------------------------------------------------------------------------------
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: tensor_block_copy_dlf_c8
#endif
	subroutine tensor_block_copy_dlf_c8(dim_num,dim_extents,dim_transp,tens_in,tens_out,ierr,conjug) !PARALLEL
!Given a dense tensor block, this subroutine makes a copy of it, permuting the indices according to the <dim_transp>.
!The algorithm is cache-efficient (Author: Dmitry I. Lyakh (Liakh): quant4me@gmail.com) (C) 2014.
!INPUT:
! - dim_num - number of dimensions (>0);
! - dim_extents(1:dim_num) - dimension extents;
! - dim_transp(0:dim_num) - index permutation (O2N), dim_transp(0) is the sign of the permutation;
! - tens_in(0:) - input tensor data;
! - conjug - (optional) complex conjugation flag;
!OUTPUT:
! - tens_out(0:) - output (possibly transposed) tensor data;
! - ierr - error code (0:success).
	implicit none
!---------------------------------------
	integer, parameter:: real_kind=8
	logical, parameter:: cache_efficiency=.TRUE.
	integer(LONGINT), parameter:: cache_line_len=64/(real_kind*2) !cache line length (words)
	integer(LONGINT), parameter:: cache_line_min=cache_line_len*2 !lower bound for the input/output minor volume: => L1_cache_line*2
	integer(LONGINT), parameter:: cache_line_lim=cache_line_len*4 !upper bound for the input/output minor volume: <= SQRT(L1_size)
	integer(LONGINT), parameter:: small_tens_size=2**10 !up to this size it is useless to apply cache efficiency (fully fits in L1)
	integer(LONGINT), parameter:: vec_size=2**8 !loop reorganization parameter for direct copy
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: real_kind,cache_efficiency,cache_line_len,cache_line_min,cache_line_lim,small_tens_size,vec_size
!DIR$ ATTRIBUTES ALIGN:128:: real_kind,cache_efficiency,cache_line_len,cache_line_min,cache_line_lim,small_tens_size,vec_size
#endif
!---------------------------------------------------------------------
	integer, intent(in):: dim_num,dim_extents(1:*),dim_transp(0:*)
	complex(real_kind), intent(in):: tens_in(0:*)
	complex(real_kind), intent(out):: tens_out(0:*)
	integer, intent(inout):: ierr
	logical, intent(in), optional:: conjug
	integer i,j,k,l,m,n,k1,k2,ks,kf,split_in,split_out
	integer im(1:dim_num),n2o(0:dim_num+1),ipr(1:dim_num+1),dim_beg(1:dim_num),dim_end(1:dim_num)
	integer(LONGINT) bases_in(1:dim_num+1),bases_out(1:dim_num+1),bases_pri(1:dim_num+1),segs(0:MAX_THREADS) !`Is segs(:) threadsafe?
	integer(LONGINT) bs,l0,l1,l2,l3,ll,lb,le,ls,l_in,l_out,seg_in,seg_out,vol_min,vol_ext
	logical trivial,conj
	real(8) time_beg,tm
#ifndef NO_PHI
!DIR$ ATTRIBUTES ALIGN:128:: im,n2o,ipr,dim_beg,dim_end,bases_in,bases_out,bases_pri,segs
#endif
	ierr=0
	time_beg=thread_wtime() !debug
	if(present(conjug)) then; conj=conjug; else; conj=.FALSE.; endif !optional complex conjugation
	if(dim_num.lt.0) then
	 ierr=1; return
	elseif(dim_num.eq.0) then
	 if(conj) then; tens_out(0)=conjg(tens_in(0)); else; tens_out(0)=tens_in(0); endif
	 return
	endif
!Check the index permutation:
	trivial=.TRUE.; do i=1,dim_num; if(dim_transp(i).ne.i) then; trivial=.FALSE.; exit; endif; enddo
	if(trivial.and.cache_efficiency) then
!Trivial index permutation (no permutation):
 !Compute indexing bases:
	 bs=1_LONGINT; do i=1,dim_num; bases_in(i)=bs; bs=bs*dim_extents(i); enddo
 !Copy input to output:
	 if(conj) then
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(l0,l1)
!$OMP DO SCHEDULE(GUIDED)
	  do l0=0_LONGINT,bs-1_LONGINT-mod(bs,vec_size),vec_size
	   do l1=0_LONGINT,vec_size-1_LONGINT; tens_out(l0+l1)=conjg(tens_in(l0+l1)); enddo
	  enddo
!$OMP END DO NOWAIT
!$OMP SINGLE
	  do l0=bs-mod(bs,vec_size),bs-1_LONGINT; tens_out(l0)=conjg(tens_in(l0)); enddo
!$OMP END SINGLE
!$OMP END PARALLEL
	 else
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(l0,l1)
!$OMP DO SCHEDULE(GUIDED)
	  do l0=0_LONGINT,bs-1_LONGINT-mod(bs,vec_size),vec_size
	   do l1=0_LONGINT,vec_size-1_LONGINT; tens_out(l0+l1)=tens_in(l0+l1); enddo
	  enddo
!$OMP END DO NOWAIT
!$OMP SINGLE
	  do l0=bs-mod(bs,vec_size),bs-1_LONGINT; tens_out(l0)=tens_in(l0); enddo
!$OMP END SINGLE
!$OMP END PARALLEL
	 endif
	else
!Non-trivial index permutation:
 !Compute indexing bases:
	 do i=1,dim_num; n2o(dim_transp(i))=i; enddo; n2o(dim_num+1)=dim_num+1 !get the N2O
	 bs=1_LONGINT; do i=1,dim_num; bases_in(i)=bs; bs=bs*dim_extents(i); enddo; bases_in(dim_num+1)=bs
	 bs=1_LONGINT; do i=1,dim_num; bases_out(n2o(i))=bs; bs=bs*dim_extents(n2o(i)); enddo; bases_out(dim_num+1)=bs
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
	   k1=k1+1; split_in=k1; seg_in=(cache_line_lim-1_LONGINT)/bases_in(split_in)+1_LONGINT
	   split_out=n2o(k2); seg_out=dim_extents(split_out)
	  elseif(bases_in(k1+1).ge.cache_line_min.and.bases_out(n2o(k2+1)).lt.cache_line_min) then !split the last minor output dim
	   k2=k2+1; split_in=n2o(k2); seg_in=(cache_line_lim-1_LONGINT)/bases_out(split_in)+1_LONGINT
	   split_out=k1; seg_out=dim_extents(split_out)
	  elseif(bases_in(k1+1).lt.cache_line_min.and.bases_out(n2o(k2+1)).lt.cache_line_min) then !split both
	   k1=k1+1; k2=k2+1
	   if(k1.eq.n2o(k2)) then
	    split_in=k1; seg_in=(cache_line_lim-1_LONGINT)/min(bases_in(split_in),bases_out(split_in))+1_LONGINT
	    split_out=k1; seg_out=dim_extents(split_out)
	   else
	    split_in=k1; seg_in=(cache_line_lim-1_LONGINT)/bases_in(split_in)+1_LONGINT
	    split_out=n2o(k2); seg_out=(cache_line_lim-1_LONGINT)/bases_out(split_out)+1_LONGINT
	   endif
	  else !split none
	   split_in=k1; seg_in=dim_extents(split_in)
	   split_out=n2o(k2); seg_out=dim_extents(split_out)
	  endif
	  vol_min=1_LONGINT
	  if(seg_in.lt.dim_extents(split_in)) vol_min=vol_min*seg_in
	  if(seg_out.lt.dim_extents(split_out)) vol_min=vol_min*seg_out
	  if(vol_min.gt.1_LONGINT) then
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
	      seg_in=min(seg_in*l,int(dim_extents(split_in),LONGINT))
	      seg_out=min(seg_out*l,int(dim_extents(split_out),LONGINT))
	     else
	      seg_in=min(seg_in*l,int(dim_extents(split_in),LONGINT))
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
	 vol_ext=1_LONGINT; do j=kf+1,dim_num; vol_ext=vol_ext*dim_extents(ipr(j)); enddo !external volume
!	 write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_copy_dlf_c8): extents:",99(1x,i5))') dim_extents(1:dim_num) !debug
!	 write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_copy_dlf_c8): permutation:",99(1x,i2))') dim_transp(1:dim_num) !debug
!	 write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_copy_dlf_c8): minor ",i3,": priority:",99(1x,i2))') &
!         kf,ipr(1:dim_num) !debug
!        write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_copy_dlf_c8): vol_ext ",i11,": segs:",4(1x,i5))') &
!         vol_ext,split_in,split_out,seg_in,seg_out !debug
 !Transpose:
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,j,m,n,ks,l0,l1,l2,l3,ll,lb,le,ls,l_in,l_out,vol_min,im,dim_beg,dim_end)
#ifndef NO_OMP
	 n=omp_get_thread_num(); m=omp_get_num_threads() !multi-threaded execution
#else
	 n=0; m=1 !serial execution
#endif
!	 if(n.eq.0) write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_copy_dlf_c8): number of threads = ",i5)') m !debug
         if(kf.lt.dim_num) then !external indices present
!$OMP MASTER
	  segs(0)=0_LONGINT; call divide_segment(vol_ext,int(m,LONGINT),segs(1:),i); do j=2,m; segs(j)=segs(j)+segs(j-1); enddo
	  l0=1_LONGINT; do i=kf+1,dim_num; bases_pri(ipr(i))=l0; l0=l0*dim_extents(ipr(i)); enddo !priority bases
!$OMP END MASTER
!$OMP BARRIER
!$OMP FLUSH(segs,bases_pri)
	  dim_beg(1:dim_num)=0; dim_end(1:dim_num)=dim_extents(1:dim_num)-1
	  l2=dim_end(split_in); l3=dim_end(split_out); ls=bases_out(1)
	  loop0: do l1=0_LONGINT,l3,seg_out !output dimension
	   dim_beg(split_out)=l1; dim_end(split_out)=min(l1+seg_out-1_LONGINT,l3)
	   do l0=0_LONGINT,l2,seg_in !input dimension
	    dim_beg(split_in)=l0; dim_end(split_in)=min(l0+seg_in-1_LONGINT,l2)
	    ll=segs(n); do i=dim_num,kf+1,-1; j=ipr(i); im(j)=ll/bases_pri(j); ll=ll-im(j)*bases_pri(j); enddo
	    vol_min=1_LONGINT; do i=1,kf; j=ipr(i); vol_min=vol_min*(dim_end(j)-dim_beg(j)+1); im(j)=dim_beg(j); enddo
	    l_in=0_LONGINT; do j=1,dim_num; l_in=l_in+im(j)*bases_in(j); enddo
	    l_out=0_LONGINT; do j=1,dim_num; l_out=l_out+im(j)*bases_out(j); enddo
	    le=dim_end(1)-dim_beg(1); lb=(segs(n+1)-segs(n))*vol_min; ks=0
	    loop1: do while(lb.gt.0_LONGINT)
	     if(conj) then
	      do ll=0_LONGINT,le
	       tens_out(l_out+ll*ls)=conjg(tens_in(l_in+ll))
	      enddo
	     else
	      do ll=0_LONGINT,le
	       tens_out(l_out+ll*ls)=tens_in(l_in+ll)
	      enddo
	     endif
	     lb=lb-(le+1_LONGINT)
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
	    if(lb.ne.0_LONGINT) then
	     if(VERBOSE) write(CONS_OUT,'("ERROR(tensor_algebra::tensor_block_copy_dlf_c8): invalid remainder: ",i11,1x,i4)') lb,n
!!!$OMP ATOMIC WRITE SEQ_CST
!$OMP ATOMIC WRITE
	     ierr=2
	     exit loop0
	    endif
	   enddo !l0
          enddo loop0 !l1
         else !external indices absent
!$OMP MASTER
	  l0=1_LONGINT; do i=kf+1,dim_num; bases_pri(ipr(i))=l0; l0=l0*dim_extents(ipr(i)); enddo !priority bases
!$OMP END MASTER
!$OMP BARRIER
!$OMP FLUSH(bases_pri)
	  dim_beg(1:dim_num)=0; dim_end(1:dim_num)=dim_extents(1:dim_num)-1
          l2=dim_end(split_in); l3=dim_end(split_out); ls=bases_out(1)
!$OMP DO SCHEDULE(DYNAMIC) COLLAPSE(2)
	  do l1=0_LONGINT,l3,seg_out !output dimension
	   do l0=0_LONGINT,l2,seg_in !input dimension
	    dim_beg(split_out)=l1; dim_end(split_out)=min(l1+seg_out-1_LONGINT,l3)
	    dim_beg(split_in)=l0; dim_end(split_in)=min(l0+seg_in-1_LONGINT,l2)
	    vol_min=1_LONGINT; do i=1,kf; j=ipr(i); vol_min=vol_min*(dim_end(j)-dim_beg(j)+1); im(j)=dim_beg(j); enddo
	    l_in=0_LONGINT; do j=1,dim_num; l_in=l_in+im(j)*bases_in(j); enddo
	    l_out=0_LONGINT; do j=1,dim_num; l_out=l_out+im(j)*bases_out(j); enddo
	    le=dim_end(1)-dim_beg(1); lb=vol_min; ks=0
	    loop2: do while(lb.gt.0_LONGINT)
	     if(conj) then
	      do ll=0_LONGINT,le
	       tens_out(l_out+ll*ls)=conjg(tens_in(l_in+ll))
	      enddo
	     else
	      do ll=0_LONGINT,le
	       tens_out(l_out+ll*ls)=tens_in(l_in+ll)
	      enddo
	     endif
	     lb=lb-(le+1_LONGINT)
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
	tm=thread_wtime(time_beg) !debug
	if(LOGGING.gt.0) then
	 write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_copy_dlf_c8): Done: ",F10.4," sec, ",F10.4," GB/s, error ",i3)') &
	 tm,dble(2_LONGINT*bs*real_kind*2_LONGINT)/(tm*1024d0*1024d0*1024d0),ierr !debug
	endif
	return
	end subroutine tensor_block_copy_dlf_c8
!--------------------------------------------------------------------------------------------------------
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: tensor_block_copy_scatter_dlf_r4
#endif
	subroutine tensor_block_copy_scatter_dlf_r4(dim_num,dim_extents,dim_transp,tens_in,tens_out,ierr) !PARALLEL
!Given a dense tensor block, this subroutine makes a copy of it, permuting the indices according to the <dim_transp>.
!INPUT:
! - dim_num - number of dimensions (>0);
! - dim_extents(1:dim_num) - dimension extents;
! - dim_transp(0:dim_num) - index permutation (O2N);
! - tens_in(0:) - input tensor data;
!OUTPUT:
! - tens_out(0:) - output (possibly transposed) tensor data;
! - ierr - error code (0:success).
	implicit none
!---------------------------------------
	integer, parameter:: real_kind=4
!---------------------------------------
	integer, intent(in):: dim_num,dim_extents(1:*)
	integer, intent(in):: dim_transp(0:*)
	real(real_kind), intent(in):: tens_in(0:*)
	real(real_kind), intent(out):: tens_out(0:*)
	integer, intent(inout):: ierr
	integer i,k,n2o(dim_num)
	integer(LONGINT) j,l,m,n,base_in(dim_num),base_out(dim_num)
	logical trivial
	real(8) time_beg
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: real_kind
!DIR$ ATTRIBUTES ALIGN:128:: real_kind,n2o,base_in,base_out
#endif

	ierr=0
!	time_beg=thread_wtime() !debug
	if(dim_num.eq.0) then !scalar tensor
	 tens_out(0)=tens_in(0)
	elseif(dim_num.gt.0) then
	 trivial=.TRUE.; do i=1,dim_num; if(dim_transp(i).ne.i) then; trivial=.FALSE.; exit; endif; enddo
	 n=dim_extents(1); do i=2,dim_num; n=n*dim_extents(i); enddo
	 if(trivial) then !trivial permutation
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i) SCHEDULE(GUIDED)
	  do l=0_LONGINT,n-1_LONGINT; tens_out(l)=tens_in(l); enddo
!$OMP END PARALLEL DO
	 else !non-trivial permutation
	  do i=1,dim_num; n2o(dim_transp(i))=i; enddo
	  j=1_LONGINT; do i=1,dim_num; base_in(i)=j; j=j*dim_extents(i); enddo
	  j=1_LONGINT; do i=1,dim_num; k=n2o(i); base_out(k)=j; j=j*dim_extents(k); enddo
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,j,k,l) SCHEDULE(GUIDED)
	  do m=0_LONGINT,n-1_LONGINT
	   l=0_LONGINT; j=m; do k=dim_num,1,-1; l=l+(j/base_in(k))*base_out(k); j=mod(j,base_in(k)); enddo
	   tens_out(l)=tens_in(m)
	  enddo
!$OMP END PARALLEL DO
	 endif
	else
	 ierr=1
	endif
!	write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_copy_scatter_dlf_r4): kernel time/error code = ",F10.4,1x,i3)') &
!        thread_wtime(time_beg),ierr !debug
	return
	end subroutine tensor_block_copy_scatter_dlf_r4
!--------------------------------------------------------------------------------------------------------
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: tensor_block_copy_scatter_dlf_r8
#endif
	subroutine tensor_block_copy_scatter_dlf_r8(dim_num,dim_extents,dim_transp,tens_in,tens_out,ierr) !PARALLEL
!Given a dense tensor block, this subroutine makes a copy of it, permuting the indices according to the <dim_transp>.
!INPUT:
! - dim_num - number of dimensions (>0);
! - dim_extents(1:dim_num) - dimension extents;
! - dim_transp(0:dim_num) - index permutation (O2N);
! - tens_in(0:) - input tensor data;
!OUTPUT:
! - tens_out(0:) - output (possibly transposed) tensor data;
! - ierr - error code (0:success).
	implicit none
!---------------------------------------
	integer, parameter:: real_kind=8
!---------------------------------------
	integer, intent(in):: dim_num,dim_extents(1:*)
	integer, intent(in):: dim_transp(0:*)
	real(real_kind), intent(in):: tens_in(0:*)
	real(real_kind), intent(out):: tens_out(0:*)
	integer, intent(inout):: ierr
	integer i,k,n2o(dim_num)
	integer(LONGINT) j,l,m,n,base_in(dim_num),base_out(dim_num)
	logical trivial
	real(8) time_beg,tm
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: real_kind
!DIR$ ATTRIBUTES ALIGN:128:: real_kind,n2o,base_in,base_out
#endif

	ierr=0
!	time_beg=thread_wtime() !debug
	if(dim_num.eq.0) then !scalar tensor
	 tens_out(0)=tens_in(0)
	elseif(dim_num.gt.0) then
	 trivial=.TRUE.; do i=1,dim_num; if(dim_transp(i).ne.i) then; trivial=.FALSE.; exit; endif; enddo
	 n=dim_extents(1); do i=2,dim_num; n=n*dim_extents(i); enddo
	 if(trivial) then !trivial permutation
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i) SCHEDULE(GUIDED)
	  do l=0_LONGINT,n-1_LONGINT; tens_out(l)=tens_in(l); enddo
!$OMP END PARALLEL DO
	 else !non-trivial permutation
	  do i=1,dim_num; n2o(dim_transp(i))=i; enddo
	  j=1_LONGINT; do i=1,dim_num; base_in(i)=j; j=j*dim_extents(i); enddo
	  j=1_LONGINT; do i=1,dim_num; k=n2o(i); base_out(k)=j; j=j*dim_extents(k); enddo
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,j,k,l) SCHEDULE(GUIDED)
	  do m=0_LONGINT,n-1_LONGINT
	   l=0_LONGINT; j=m; do k=dim_num,1,-1; l=l+(j/base_in(k))*base_out(k); j=mod(j,base_in(k)); enddo
	   tens_out(l)=tens_in(m)
	  enddo
!$OMP END PARALLEL DO
	 endif
	else
	 ierr=1
	endif
!       tm=thread_wtime(time_beg)
!        write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_copy_scatter_dlf_r8): Done: ",F10.4," sec, ",F10.4," GB/s, error ",'&
!        &//'i3)') tm,dble(2_LONGINT*n*real_kind)/(tm*1024d0*1024d0*1024d0),ierr !debug
	return
	end subroutine tensor_block_copy_scatter_dlf_r8
!---------------------------------------------------------------------------------------------------------------
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: tensor_block_copy_scatter_dlf_c4
#endif
	subroutine tensor_block_copy_scatter_dlf_c4(dim_num,dim_extents,dim_transp,tens_in,tens_out,ierr,conjug) !PARALLEL
!Given a dense tensor block, this subroutine makes a copy of it, permuting the indices according to the <dim_transp>.
!INPUT:
! - dim_num - number of dimensions (>0);
! - dim_extents(1:dim_num) - dimension extents;
! - dim_transp(0:dim_num) - index permutation (O2N);
! - tens_in(0:) - input tensor data;
! - conjug - (optional) complex conjugation flag;
!OUTPUT:
! - tens_out(0:) - output (possibly transposed) tensor data;
! - ierr - error code (0:success).
	implicit none
!---------------------------------------
	integer, parameter:: real_kind=4
!---------------------------------------
	integer, intent(in):: dim_num,dim_extents(1:*)
	integer, intent(in):: dim_transp(0:*)
	complex(real_kind), intent(in):: tens_in(0:*)
	complex(real_kind), intent(out):: tens_out(0:*)
	integer, intent(inout):: ierr
	logical, intent(in), optional:: conjug
	integer i,k,n2o(dim_num)
	integer(LONGINT) j,l,m,n,base_in(dim_num),base_out(dim_num)
	logical trivial,conj
	real(8) time_beg
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: real_kind
!DIR$ ATTRIBUTES ALIGN:128:: real_kind,n2o,base_in,base_out
#endif

	ierr=0
!	time_beg=thread_wtime() !debug
	if(present(conjug)) then; conj=conjug; else; conj=.FALSE.; endif !optional complex conjugation
	if(dim_num.eq.0) then !scalar tensor
	 if(conj) then; tens_out(0)=conjg(tens_in(0)); else; tens_out(0)=tens_in(0); endif
	elseif(dim_num.gt.0) then
	 trivial=.TRUE.; do i=1,dim_num; if(dim_transp(i).ne.i) then; trivial=.FALSE.; exit; endif; enddo
	 n=dim_extents(1); do i=2,dim_num; n=n*dim_extents(i); enddo
	 if(trivial) then !trivial permutation
	  if(conj) then
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i) SCHEDULE(GUIDED)
	   do l=0_LONGINT,n-1_LONGINT; tens_out(l)=conjg(tens_in(l)); enddo
!$OMP END PARALLEL DO
	  else
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i) SCHEDULE(GUIDED)
	   do l=0_LONGINT,n-1_LONGINT; tens_out(l)=tens_in(l); enddo
!$OMP END PARALLEL DO
	  endif
	 else !non-trivial permutation
	  do i=1,dim_num; n2o(dim_transp(i))=i; enddo
	  j=1_LONGINT; do i=1,dim_num; base_in(i)=j; j=j*dim_extents(i); enddo
	  j=1_LONGINT; do i=1,dim_num; k=n2o(i); base_out(k)=j; j=j*dim_extents(k); enddo
	  if(conj) then
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,j,k,l) SCHEDULE(GUIDED)
	   do m=0_LONGINT,n-1_LONGINT
	    l=0_LONGINT; j=m; do k=dim_num,1,-1; l=l+(j/base_in(k))*base_out(k); j=mod(j,base_in(k)); enddo
	    tens_out(l)=conjg(tens_in(m))
	   enddo
!$OMP END PARALLEL DO
	  else
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,j,k,l) SCHEDULE(GUIDED)
	   do m=0_LONGINT,n-1_LONGINT
	    l=0_LONGINT; j=m; do k=dim_num,1,-1; l=l+(j/base_in(k))*base_out(k); j=mod(j,base_in(k)); enddo
	    tens_out(l)=tens_in(m)
	   enddo
!$OMP END PARALLEL DO
	  endif
	 endif
	else
	 ierr=1
	endif
!	write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_copy_scatter_dlf_c4): kernel time/error code = ",F10.4,1x,i3)') &
!        thread_wtime(time_beg),ierr !debug
	return
	end subroutine tensor_block_copy_scatter_dlf_c4
!---------------------------------------------------------------------------------------------------------------
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: tensor_block_copy_scatter_dlf_c8
#endif
	subroutine tensor_block_copy_scatter_dlf_c8(dim_num,dim_extents,dim_transp,tens_in,tens_out,ierr,conjug) !PARALLEL
!Given a dense tensor block, this subroutine makes a copy of it, permuting the indices according to the <dim_transp>.
!INPUT:
! - dim_num - number of dimensions (>0);
! - dim_extents(1:dim_num) - dimension extents;
! - dim_transp(0:dim_num) - index permutation (O2N);
! - tens_in(0:) - input tensor data;
! - conjug - (optional) complex conjugation flag;
!OUTPUT:
! - tens_out(0:) - output (possibly transposed) tensor data;
! - ierr - error code (0:success).
	implicit none
!---------------------------------------
	integer, parameter:: real_kind=8
!---------------------------------------
	integer, intent(in):: dim_num,dim_extents(1:*)
	integer, intent(in):: dim_transp(0:*)
	complex(real_kind), intent(in):: tens_in(0:*)
	complex(real_kind), intent(out):: tens_out(0:*)
	integer, intent(inout):: ierr
	logical, intent(in), optional:: conjug
	integer i,k,n2o(dim_num)
	integer(LONGINT) j,l,m,n,base_in(dim_num),base_out(dim_num)
	logical trivial,conj
	real(8) time_beg
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: real_kind
!DIR$ ATTRIBUTES ALIGN:128:: real_kind,n2o,base_in,base_out
#endif

	ierr=0
!	time_beg=thread_wtime() !debug
	if(present(conjug)) then; conj=conjug; else; conj=.FALSE.; endif !optional complex conjugation
	if(dim_num.eq.0) then !scalar tensor
	 if(conj) then; tens_out(0)=conjg(tens_in(0)); else; tens_out(0)=tens_in(0); endif
	elseif(dim_num.gt.0) then
	 trivial=.TRUE.; do i=1,dim_num; if(dim_transp(i).ne.i) then; trivial=.FALSE.; exit; endif; enddo
	 n=dim_extents(1); do i=2,dim_num; n=n*dim_extents(i); enddo
	 if(trivial) then !trivial permutation
	  if(conj) then
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i) SCHEDULE(GUIDED)
	   do l=0_LONGINT,n-1_LONGINT; tens_out(l)=conjg(tens_in(l)); enddo
!$OMP END PARALLEL DO
	  else
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i) SCHEDULE(GUIDED)
	   do l=0_LONGINT,n-1_LONGINT; tens_out(l)=tens_in(l); enddo
!$OMP END PARALLEL DO
	  endif
	 else !non-trivial permutation
	  do i=1,dim_num; n2o(dim_transp(i))=i; enddo
	  j=1_LONGINT; do i=1,dim_num; base_in(i)=j; j=j*dim_extents(i); enddo
	  j=1_LONGINT; do i=1,dim_num; k=n2o(i); base_out(k)=j; j=j*dim_extents(k); enddo
	  if(conj) then
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,j,k,l) SCHEDULE(GUIDED)
	   do m=0_LONGINT,n-1_LONGINT
	    l=0_LONGINT; j=m; do k=dim_num,1,-1; l=l+(j/base_in(k))*base_out(k); j=mod(j,base_in(k)); enddo
	    tens_out(l)=conjg(tens_in(m))
	   enddo
!$OMP END PARALLEL DO
	  else
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,j,k,l) SCHEDULE(GUIDED)
	   do m=0_LONGINT,n-1_LONGINT
	    l=0_LONGINT; j=m; do k=dim_num,1,-1; l=l+(j/base_in(k))*base_out(k); j=mod(j,base_in(k)); enddo
	    tens_out(l)=tens_in(m)
	   enddo
!$OMP END PARALLEL DO
	  endif
	 endif
	else
	 ierr=1
	endif
!	write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_copy_scatter_dlf_c8): kernel time/error code = ",F10.4,1x,i3)') &
!        thread_wtime(time_beg),ierr !debug
	return
	end subroutine tensor_block_copy_scatter_dlf_c8
!-------------------------------------------------------------------------------------
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: tensor_block_fcontract_dlf_r4
#endif
	subroutine tensor_block_fcontract_dlf_r4(dc,ltens,rtens,dtens,ierr,alpha,beta) !PARALLEL
!This subroutine fully reduces two vectors derived from the corresponding tensors by index permutations:
!dtens+=ltens(0:dc-1)*rtens(0:dc-1)*alpha, where dtens is a scalar.
	implicit none
!---------------------------------------
	integer, parameter:: real_kind=4 !real data kind
!---------------------------------------
	integer(LONGINT), intent(in):: dc
	real(real_kind), intent(in):: ltens(0:*),rtens(0:*) !true tensors
	real(real_kind), intent(inout):: dtens !scalar
	integer, intent(inout):: ierr !error code
	real(real_kind), intent(in), optional:: alpha !BLAS alpha
	real(real_kind), intent(in), optional:: beta  !BLAS beta (defaults to 1)
	integer i,j,k,l,m,n
	real(real_kind) val,alf,bet
	integer(LONGINT) l0
	real(8) time_beg,tm
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: real_kind
!DIR$ ATTRIBUTES ALIGN:128:: real_kind
#endif

	ierr=0
!	time_beg=thread_wtime() !debug
!	write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_fcontract_dlf_r4): dc: ",i9)') dc !debug
	if(present(alpha)) then; alf=alpha; else; alf=1.0; endif
	if(present(beta)) then; bet=beta; else; bet=1.0; endif
	if(dc.gt.0_LONGINT) then
	 val=0.0
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED) REDUCTION(+:val)
	 do l0=0_LONGINT,dc-1_LONGINT
          val=val+ltens(l0)*rtens(l0)
         enddo
!$OMP END PARALLEL DO
	 dtens=dtens*bet+val*alf
	else
	 ierr=1
	endif
!       tm=thread_wtime(time_beg) !debug
!	write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_fcontract_dlf_r4): time/speed/error = ",2(F10.4,1x),i3)') &
!        tm,2d0*dble(dc)/(tm*1024d0*1024d0*1024d0),ierr !debug
	return
	end subroutine tensor_block_fcontract_dlf_r4
!-------------------------------------------------------------------------------------
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: tensor_block_fcontract_dlf_r8
#endif
	subroutine tensor_block_fcontract_dlf_r8(dc,ltens,rtens,dtens,ierr,alpha,beta) !PARALLEL
!This subroutine fully reduces two vectors derived from the corresponding tensors by index permutations:
!dtens+=ltens(0:dc-1)*rtens(0:dc-1)*alpha, where dtens is a scalar.
	implicit none
!---------------------------------------
	integer, parameter:: real_kind=8 !real data kind
!---------------------------------------
	integer(LONGINT), intent(in):: dc
	real(real_kind), intent(in):: ltens(0:*),rtens(0:*) !true tensors
	real(real_kind), intent(inout):: dtens !scalar
	integer, intent(inout):: ierr !error code
	real(real_kind), intent(in), optional:: alpha !BLAS alpha
	real(real_kind), intent(in), optional:: beta  !BLAS beta (defaults to 1)
	integer i,j,k,l,m,n
	real(real_kind) val,alf,bet
	integer(LONGINT) l0
	real(8) time_beg,tm
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: real_kind
!DIR$ ATTRIBUTES ALIGN:128:: real_kind
#endif

	ierr=0
!	time_beg=thread_wtime() !debug
!	write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_fcontract_dlf_r8): dc: ",i9)') dc !debug
	if(present(alpha)) then; alf=alpha; else; alf=1d0; endif
	if(present(beta)) then; bet=beta; else; bet=1d0; endif
	if(dc.gt.0_LONGINT) then
	 val=0d0
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED) REDUCTION(+:val)
	 do l0=0_LONGINT,dc-1_LONGINT
          val=val+ltens(l0)*rtens(l0)
         enddo
!$OMP END PARALLEL DO
	 dtens=dtens*bet+val*alf
	else
	 ierr=1
	endif
!       tm=thread_wtime(time_beg) !debug
!	write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_fcontract_dlf_r8): time/speed/error = ",2(F10.4,1x),i3)') &
!        tm,2d0*dble(dc)/(tm*1024d0*1024d0*1024d0),ierr !debug
	return
	end subroutine tensor_block_fcontract_dlf_r8
!-------------------------------------------------------------------------------------
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: tensor_block_fcontract_dlf_c4
#endif
	subroutine tensor_block_fcontract_dlf_c4(dc,ltens,rtens,dtens,ierr,alpha,beta) !PARALLEL
!This subroutine fully reduces two vectors derived from the corresponding tensors by index permutations:
!dtens+=ltens(0:dc-1)*rtens(0:dc-1)*alpha, where dtens is a scalar.
	implicit none
!---------------------------------------
	integer, parameter:: real_kind=4 !real data kind
!---------------------------------------
	integer(LONGINT), intent(in):: dc
	complex(real_kind), intent(in):: ltens(0:*),rtens(0:*) !true tensors
	complex(real_kind), intent(inout):: dtens !scalar
	integer, intent(inout):: ierr !error code
	complex(real_kind), intent(in), optional:: alpha !BLAS alpha
	complex(real_kind), intent(in), optional:: beta  !BLAS beta (defaults to 1)
	integer i,j,k,l,m,n
	complex(real_kind) val,alf,bet
	integer(LONGINT) l0
	real(8) time_beg,tm
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: real_kind
!DIR$ ATTRIBUTES ALIGN:128:: real_kind
#endif

	ierr=0
!	time_beg=thread_wtime() !debug
!	write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_fcontract_dlf_c4): dc: ",i9)') dc !debug
	if(present(alpha)) then; alf=alpha; else; alf=(1.0,0.0); endif
	if(present(beta)) then; bet=beta; else; bet=(1.0,0.0); endif
	if(dc.gt.0_LONGINT) then
	 val=(0.0,0.0)
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED) REDUCTION(+:val)
	 do l0=0_LONGINT,dc-1_LONGINT
          val=val+ltens(l0)*rtens(l0)
         enddo
!$OMP END PARALLEL DO
	 dtens=dtens*bet+val*alf
	else
	 ierr=1
	endif
!       tm=thread_wtime(time_beg) !debug
!	write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_fcontract_dlf_c4): time/speed/error = ",2(F10.4,1x),i3)') &
!        tm,2d0*dble(dc)/(tm*1024d0*1024d0*1024d0),ierr !debug
	return
	end subroutine tensor_block_fcontract_dlf_c4
!-------------------------------------------------------------------------------------
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: tensor_block_fcontract_dlf_c8
#endif
	subroutine tensor_block_fcontract_dlf_c8(dc,ltens,rtens,dtens,ierr,alpha,beta) !PARALLEL
!This subroutine fully reduces two vectors derived from the corresponding tensors by index permutations:
!dtens+=ltens(0:dc-1)*rtens(0:dc-1)*alpha, where dtens is a scalar.
	implicit none
!---------------------------------------
	integer, parameter:: real_kind=8 !real data kind
!---------------------------------------
	integer(LONGINT), intent(in):: dc
	complex(real_kind), intent(in):: ltens(0:*),rtens(0:*) !true tensors
	complex(real_kind), intent(inout):: dtens !scalar
	integer, intent(inout):: ierr !error code
	complex(real_kind), intent(in), optional:: alpha !BLAS alpha
	complex(real_kind), intent(in), optional:: beta  !BLAS beta (defaults to 1)
	integer i,j,k,l,m,n
	complex(real_kind) val,alf,bet
	integer(LONGINT) l0
	real(8) time_beg,tm
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: real_kind
!DIR$ ATTRIBUTES ALIGN:128:: real_kind
#endif

	ierr=0
!	time_beg=thread_wtime() !debug
!	write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_fcontract_dlf_c8): dc: ",i9)') dc !debug
	if(present(alpha)) then; alf=alpha; else; alf=(1d0,0d0); endif
	if(present(beta)) then; bet=beta; else; bet=(1d0,0d0); endif
	if(dc.gt.0_LONGINT) then
	 val=(0d0,0d0)
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED) REDUCTION(+:val)
	 do l0=0_LONGINT,dc-1_LONGINT
          val=val+ltens(l0)*rtens(l0)
         enddo
!$OMP END PARALLEL DO
	 dtens=dtens*bet+val*alf
	else
	 ierr=1
	endif
!       tm=thread_wtime(time_beg) !debug
!	write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_fcontract_dlf_c8): time/speed/error = ",2(F10.4,1x),i3)') &
!        tm,2d0*dble(dc)/(tm*1024d0*1024d0*1024d0),ierr !debug
	return
	end subroutine tensor_block_fcontract_dlf_c8
!-------------------------------------------------------------------------------------------
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: tensor_block_pcontract_dlf_r4
#endif
	subroutine tensor_block_pcontract_dlf_r4(dl,dr,dc,ltens,rtens,dtens,ierr,alpha,beta) !PARALLEL
!This subroutine multiplies two matrices derived from the corresponding tensors by index permutations:
!dtens(0:dl-1,0:dr-1)+=ltens(0:dc-1,0:dl-1)*rtens(0:dc-1,0:dr-1)*alpha
!The result is a matrix as well (cannot be a scalar, see tensor_block_fcontract).
	implicit none
!---------------------------------------
	integer, parameter:: real_kind=4                   !real data kind
	integer(LONGINT), parameter:: red_mat_size=32      !the size of the local reduction matrix
	integer(LONGINT), parameter:: arg_cache_size=2**15 !cache-size dependent parameter (increase it as there are more cores per node)
	integer, parameter:: min_distr_seg_size=128  !min segment size of an omp distributed dimension
	integer, parameter:: cdim_stretch=2          !makes the segmentation of the contracted dimension coarser
	integer, parameter:: core_slope=16           !regulates the slope of the segment size of the distributed dimension w.r.t. the number of cores
	integer, parameter:: ker1=0                  !kernel 1 scheme #
	integer, parameter:: ker2=0                  !kernel 2 scheme #
	integer, parameter:: ker3=0                  !kernel 3 scheme #
!----------------------------------
	logical, parameter:: no_case1=.FALSE.
	logical, parameter:: no_case2=.FALSE.
	logical, parameter:: no_case3=.FALSE.
	logical, parameter:: no_case4=.FALSE.
!----------------------------------------------
	integer(LONGINT), intent(in):: dl,dr,dc !matrix dimensions
	real(real_kind), intent(in):: ltens(0:*),rtens(0:*) !input arguments
	real(real_kind), intent(inout):: dtens(0:*) !output argument
	integer, intent(inout):: ierr !error code
	real(real_kind), intent(in), optional:: alpha !BLAS alpha
	real(real_kind), intent(in), optional:: beta  !BLAS beta (defaults to 1)
	integer i,j,k,l,m,n,nthr
	integer(LONGINT) ll,lr,ld,l0,l1,l2,b0,b1,b2,e0r,e0,e1,e2,ls,lf,cl,cr,cc,chunk
	real(real_kind) vec(0:7),redm(0:red_mat_size-1,0:red_mat_size-1),val,alf
	real(8) time_beg,tm
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: real_kind,red_mat_size,arg_cache_size,min_distr_seg_size,cdim_stretch,core_slope,ker1,ker2,ker3
!DIR$ ATTRIBUTES OFFLOAD:mic:: no_case1,no_case2,no_case3,no_case4
!DIR$ ATTRIBUTES ALIGN:128:: real_kind,red_mat_size,arg_cache_size,min_distr_seg_size,cdim_stretch,core_slope,ker1,ker2,ker3
!DIR$ ATTRIBUTES ALIGN:128:: no_case1,no_case2,no_case3,no_case4,vec,redm
#endif

	ierr=0
!	time_beg=thread_wtime() !debug
	if(present(alpha)) then; alf=alpha; else; alf=1.0; endif
	if(present(beta)) then !rescale output tensor if requested
	 if(beta.ne.1.0) then
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED)
	  do l0=0_LONGINT,dr*dl-1_LONGINT
	   dtens(l0)=dtens(l0)*beta
	  enddo
!$OMP END PARALLEL DO
	 endif
	endif
	if(dl.gt.0_LONGINT.and.dr.gt.0_LONGINT.and.dc.gt.0_LONGINT) then
#ifndef NO_OMP
	 nthr=omp_get_max_threads()
#else
	 nthr=1
#endif
	 if(dr.ge.core_slope*nthr.and.(.not.no_case1)) then !the right dimension is large enough to be distributed
!	  write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_r4): kernel case/scheme: 1/",i1)') ker1 !debug
	  select case(ker1)
	  case(0)
!SCHEME 0:
	   cr=min(dr,int(max(core_slope*nthr,min_distr_seg_size),LONGINT))
	   cc=min(dc,max((arg_cache_size*int(cdim_stretch,LONGINT))/cr,1_LONGINT))
	   cl=min(dl,min(max(arg_cache_size/cc,1_LONGINT),max(arg_cache_size/cr,1_LONGINT)))
!	   write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_r4): cl,cr,cc,dl,dr,dc:",6(1x,i9))') &
!           cl,cr,cc,dl,dr,dc !debug
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(b0,b1,b2,e0r,e0,e1,e2,l0,l1,l2,ll,lr,ld,vec)
	   do b0=0_LONGINT,dc-1_LONGINT,cc
	    e0=min(b0+cc-1_LONGINT,dc-1_LONGINT)
	    e0r=mod(e0-b0+1_LONGINT,8_LONGINT)
	    do b1=0_LONGINT,dl-1_LONGINT,cl
	     e1=min(b1+cl-1_LONGINT,dl-1_LONGINT)
	     do b2=0_LONGINT,dr-1_LONGINT,cr
	      e2=min(b2+cr-1_LONGINT,dr-1_LONGINT)
!$OMP DO SCHEDULE(GUIDED)
	      do l2=b2,e2
	       ld=l2*dl
	       do l1=b1,e1
	        ll=l1*dc+b0; lr=l2*dc+b0; vec(:)=0E0_real_kind
	        do l0=b0,e0-e0r,8_LONGINT
	         vec(0)=vec(0)+ltens(ll)*rtens(lr)*alf
	         vec(1)=vec(1)+ltens(ll+1_LONGINT)*rtens(lr+1_LONGINT)*alf
	         vec(2)=vec(2)+ltens(ll+2_LONGINT)*rtens(lr+2_LONGINT)*alf
	         vec(3)=vec(3)+ltens(ll+3_LONGINT)*rtens(lr+3_LONGINT)*alf
	         vec(4)=vec(4)+ltens(ll+4_LONGINT)*rtens(lr+4_LONGINT)*alf
	         vec(5)=vec(5)+ltens(ll+5_LONGINT)*rtens(lr+5_LONGINT)*alf
	         vec(6)=vec(6)+ltens(ll+6_LONGINT)*rtens(lr+6_LONGINT)*alf
	         vec(7)=vec(7)+ltens(ll+7_LONGINT)*rtens(lr+7_LONGINT)*alf
	         ll=ll+8_LONGINT; lr=lr+8_LONGINT
	        enddo
	        do l0=0_LONGINT,e0r-1_LONGINT
	         vec(l0)=vec(l0)+ltens(ll+l0)*rtens(lr+l0)*alf
	        enddo
	        vec(0)=vec(0)+vec(4)
	        vec(1)=vec(1)+vec(5)
	        vec(2)=vec(2)+vec(6)
	        vec(3)=vec(3)+vec(7)
	        dtens(ld+l1)=dtens(ld+l1)+vec(0)+vec(1)+vec(2)+vec(3)
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
	   chunk=max(arg_cache_size/dc,1_LONGINT)
!          write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_r4): chunk,dl,dr,dc:",4(1x,i9))') chunk,dl,dr,dc !debug
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(l0,l1,l2,ll,lr,ld,ls,lf,val)
	   do ls=0_LONGINT,dl-1_LONGINT,chunk
	    lf=min(ls+chunk-1_LONGINT,dl-1_LONGINT)
!!$OMP DO SCHEDULE(DYNAMIC,chunk)
!$OMP DO SCHEDULE(GUIDED)
	    do l2=0_LONGINT,dr-1_LONGINT
	     lr=l2*dc; ld=l2*dl
	     do l1=ls,lf
	      ll=l1*dc
	      val=dtens(ld+l1)
	      do l0=0_LONGINT,dc-1_LONGINT; val=val+ltens(ll+l0)*rtens(lr+l0)*alf; enddo
	      dtens(ld+l1)=val
	     enddo
	    enddo
!$OMP END DO NOWAIT
	   enddo
!$OMP END PARALLEL
	  case(2)
!SCHEME 2:
	   chunk=max(arg_cache_size/dc,1_LONGINT)
	   if(mod(dl,chunk).ne.0) then; ls=dl/chunk+1_LONGINT; else; ls=dl/chunk; endif
	   if(mod(dr,chunk).ne.0) then; lf=dr/chunk+1_LONGINT; else; lf=dr/chunk; endif
!	   write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_r4): chunk,ls,lf,dl,dr,dc:",6(1x,i9))') &
!           chunk,ls,lf,dl,dr,dc !debug
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(b0,b1,b2,l0,l1,l2,ll,lr,ld,val) SCHEDULE(GUIDED)
	   do b0=0_LONGINT,lf*ls-1_LONGINT
	    b2=b0/ls*chunk; b1=mod(b0,ls)*chunk
	    do l2=b2,min(b2+chunk-1_LONGINT,dr-1_LONGINT)
	     lr=l2*dc; ld=l2*dl
	     do l1=b1,min(b1+chunk-1_LONGINT,dl-1_LONGINT)
	      ll=l1*dc
	      val=dtens(ld+l1)
	      do l0=0_LONGINT,dc-1_LONGINT; val=val+ltens(ll+l0)*rtens(lr+l0)*alf; enddo
	      dtens(ld+l1)=val
	     enddo
	    enddo
	   enddo
!$OMP END PARALLEL DO
	  case(3)
!SCHEME 3:
!	   write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_r4): dl,dr,dc:",3(1x,i9))') dl,dr,dc !debug
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0,l1,l2,ll,lr,ld,val) SCHEDULE(DYNAMIC)
	   do l2=0_LONGINT,dr-1_LONGINT
	    lr=l2*dc; ld=l2*dl
	    do l1=0_LONGINT,dl-1_LONGINT
	     ll=l1*dc
	     val=dtens(ld+l1)
	     do l0=0_LONGINT,dc-1_LONGINT; val=val+ltens(ll+l0)*rtens(lr+l0)*alf; enddo
	     dtens(ld+l1)=val
	    enddo
	   enddo
!$OMP END PARALLEL DO
	  case default
	   ierr=1
	  end select
	 else !dr is small
	  if(dl.ge.core_slope*nthr.and.(.not.no_case2)) then !the left dimension is large enough to be distributed
!	   write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_r4): kernel case/scheme: 2/",i1)') ker2 !debug
	   select case(ker2)
	   case(0)
!SCHEME 0:
!            write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_r4): dl,dr,dc:",3(1x,i9))') dl,dr,dc !debug
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(l0,l1,l2,b2,ll,lr,ld,e0r,vec)
            e0r=mod(dc,8_LONGINT)
	    do l2=0_LONGINT,dr-1_LONGINT
	     b2=l2*dc; ld=l2*dl
!$OMP DO SCHEDULE(GUIDED)
	     do l1=0_LONGINT,dl-1_LONGINT
	      ll=l1*dc; lr=b2; vec(:)=0E0_real_kind
	      do l0=0_LONGINT,dc-1_LONGINT-e0r,8_LONGINT
	       vec(0)=vec(0)+ltens(ll)*rtens(lr)*alf
	       vec(1)=vec(1)+ltens(ll+1_LONGINT)*rtens(lr+1_LONGINT)*alf
	       vec(2)=vec(2)+ltens(ll+2_LONGINT)*rtens(lr+2_LONGINT)*alf
	       vec(3)=vec(3)+ltens(ll+3_LONGINT)*rtens(lr+3_LONGINT)*alf
	       vec(4)=vec(4)+ltens(ll+4_LONGINT)*rtens(lr+4_LONGINT)*alf
	       vec(5)=vec(5)+ltens(ll+5_LONGINT)*rtens(lr+5_LONGINT)*alf
	       vec(6)=vec(6)+ltens(ll+6_LONGINT)*rtens(lr+6_LONGINT)*alf
	       vec(7)=vec(7)+ltens(ll+7_LONGINT)*rtens(lr+7_LONGINT)*alf
	       ll=ll+8_LONGINT; lr=lr+8_LONGINT
	      enddo
	      do l0=0_LONGINT,e0r-1_LONGINT
	       vec(l0)=vec(l0)+ltens(ll+l0)*rtens(lr+l0)*alf
	      enddo
	      vec(0)=vec(0)+vec(4)
	      vec(1)=vec(1)+vec(5)
	      vec(2)=vec(2)+vec(6)
	      vec(3)=vec(3)+vec(7)
	      dtens(ld+l1)=dtens(ld+l1)+vec(0)+vec(1)+vec(2)+vec(3)
	     enddo
!$OMP END DO NOWAIT
	    enddo
!$OMP END PARALLEL
	   case(1)
!SCHEME 1:
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0,l1,l2,ll,lr,val) SCHEDULE(GUIDED) COLLAPSE(2)
	    do l2=0_LONGINT,dr-1_LONGINT
	     do l1=0_LONGINT,dl-1_LONGINT
	      ll=l1*dc; lr=l2*dc
	      val=dtens(l2*dl+l1)
	      do l0=0_LONGINT,dc-1_LONGINT; val=val+ltens(ll+l0)*rtens(lr+l0)*alf; enddo
	      dtens(l2*dl+l1)=val
	     enddo
	    enddo
!$OMP END PARALLEL DO
	   case default
	    ierr=2
	   end select
	  else !dr & dl are both small
	   if(dc.ge.core_slope*nthr.and.(.not.no_case3)) then !the contraction dimension is large enough to be distributed
!	    write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_r4): kernel case/scheme: 3/",i1)') ker3 !debug
	    select case(ker3)
	    case(0)
!SCHEME 0:
!            write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_r4): dl,dr,dc:",3(1x,i9))') dl,dr,dc !debug
             redm(:,:)=0E0_real_kind
             do b2=0_LONGINT,dr-1_LONGINT,red_mat_size
              e2=min(red_mat_size-1_LONGINT,dr-1_LONGINT-b2)
              do b1=0_LONGINT,dl-1_LONGINT,red_mat_size
               e1=min(red_mat_size-1_LONGINT,dl-1_LONGINT-b1)
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(l0,l1,l2,ll,lr)
	       do l2=0_LONGINT,e2
	        lr=(b2+l2)*dc
	        do l1=0_LONGINT,e1
	         ll=(b1+l1)*dc
!$OMP MASTER
	         val=0E0_real_kind
!$OMP END MASTER
!$OMP BARRIER
!$OMP DO SCHEDULE(GUIDED) REDUCTION(+:val)
	         do l0=0_LONGINT,dc-1_LONGINT; val=val+ltens(ll+l0)*rtens(lr+l0)*alf; enddo
!$OMP END DO
!$OMP MASTER
		 redm(l1,l2)=val
!$OMP END MASTER
	        enddo
	       enddo
!$OMP END PARALLEL
	       do l2=0_LONGINT,e2
	        ld=(b2+l2)*dl
	        do l1=0_LONGINT,e1
	         dtens(ld+b1+l1)=dtens(ld+b1+l1)+redm(l1,l2)
	        enddo
	       enddo
	      enddo
	     enddo
	    case default
	     ierr=3
	    end select
	   else !dr & dl & dc are all small
	    if(dr*dl.ge.core_slope*nthr.and.(.not.no_case4)) then !the destination matrix is large enough to be distributed
!	     write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_r4): kernel case: 4")') !debug
!	     write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_r4): dl,dr,dc:",3(1x,i9))') dl,dr,dc !debug
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0,l1,l2,ll,lr,val) SCHEDULE(GUIDED) COLLAPSE(2)
	     do l2=0_LONGINT,dr-1_LONGINT
	      do l1=0_LONGINT,dl-1_LONGINT
	       ll=l1*dc; lr=l2*dc
	       val=dtens(l2*dl+l1)
	       do l0=0_LONGINT,dc-1_LONGINT; val=val+ltens(ll+l0)*rtens(lr+l0)*alf; enddo
	       dtens(l2*dl+l1)=val
	      enddo
	     enddo
!$OMP END PARALLEL DO
	    else !all matrices are very small (serial execution)
!	     write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_r4): kernel case: 5 (serial)")') !debug
!	     write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_r4): dl,dr,dc:",3(1x,i9))') dl,dr,dc !debug
	     do l2=0_LONGINT,dr-1_LONGINT
	      lr=l2*dc; ld=l2*dl
	      do l1=0_LONGINT,dl-1_LONGINT
	       ll=l1*dc
	       val=dtens(ld+l1)
	       do l0=0_LONGINT,dc-1_LONGINT; val=val+ltens(ll+l0)*rtens(lr+l0)*alf; enddo
	       dtens(ld+l1)=val
	      enddo
	     enddo
	    endif
	   endif
	  endif
	 endif
	else
	 ierr=4
	endif
!	tm=thread_wtime(time_beg) !debug
!	write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_r4): time/speed/error = ",2(F10.4,1x),i3)') &
!        tm,2d0*dble(dr*dl*dc)/(tm*1024d0*1024d0*1024d0),ierr !debug
	return
	end subroutine tensor_block_pcontract_dlf_r4
!-------------------------------------------------------------------------------------------
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: tensor_block_pcontract_dlf_r8
#endif
	subroutine tensor_block_pcontract_dlf_r8(dl,dr,dc,ltens,rtens,dtens,ierr,alpha,beta) !PARALLEL
!This subroutine multiplies two matrices derived from the corresponding tensors by index permutations:
!dtens(0:dl-1,0:dr-1)+=ltens(0:dc-1,0:dl-1)*rtens(0:dc-1,0:dr-1)*alpha
!The result is a matrix as well (cannot be a scalar, see tensor_block_fcontract).
	implicit none
!---------------------------------------
	integer, parameter:: real_kind=8                   !real data kind
	integer(LONGINT), parameter:: red_mat_size=32      !the size of the local reduction matrix
	integer(LONGINT), parameter:: arg_cache_size=2**15 !cache-size dependent parameter (increase it as there are more cores per node)
	integer, parameter:: min_distr_seg_size=128  !min segment size of an omp distributed dimension
	integer, parameter:: cdim_stretch=2          !makes the segmentation of the contracted dimension coarser
	integer, parameter:: core_slope=16           !regulates the slope of the segment size of the distributed dimension w.r.t. the number of cores
	integer, parameter:: ker1=0                  !kernel 1 scheme #
	integer, parameter:: ker2=0                  !kernel 2 scheme #
	integer, parameter:: ker3=0                  !kernel 3 scheme #
!----------------------------------
	logical, parameter:: no_case1=.FALSE.
	logical, parameter:: no_case2=.FALSE.
	logical, parameter:: no_case3=.FALSE.
	logical, parameter:: no_case4=.FALSE.
!----------------------------------------------
	integer(LONGINT), intent(in):: dl,dr,dc !matrix dimensions
	real(real_kind), intent(in):: ltens(0:*),rtens(0:*) !input arguments
	real(real_kind), intent(inout):: dtens(0:*) !output argument
	integer, intent(inout):: ierr !error code
	real(real_kind), intent(in), optional:: alpha !BLAS alpha
	real(real_kind), intent(in), optional:: beta  !BLAS beta (defaults to 1)
	integer i,j,k,l,m,n,nthr
	integer(LONGINT) ll,lr,ld,l0,l1,l2,b0,b1,b2,e0r,e0,e1,e2,ls,lf,cl,cr,cc,chunk
	real(real_kind) vec(0:7),redm(0:red_mat_size-1,0:red_mat_size-1),val,alf !`thread private (redm)?
	real(8) time_beg,tm
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: real_kind,red_mat_size,arg_cache_size,min_distr_seg_size,cdim_stretch,core_slope,ker1,ker2,ker3
!DIR$ ATTRIBUTES OFFLOAD:mic:: no_case1,no_case2,no_case3,no_case4
!DIR$ ATTRIBUTES ALIGN:128:: real_kind,red_mat_size,arg_cache_size,min_distr_seg_size,cdim_stretch,core_slope,ker1,ker2,ker3
!DIR$ ATTRIBUTES ALIGN:128:: no_case1,no_case2,no_case3,no_case4,vec,redm
#endif

	ierr=0
!	time_beg=thread_wtime() !debug
	if(present(alpha)) then; alf=alpha; else; alf=1d0; endif
	if(present(beta)) then !rescale output tensor if requested
	 if(beta.ne.1d0) then
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED)
	  do l0=0_LONGINT,dr*dl-1_LONGINT
	   dtens(l0)=dtens(l0)*beta
	  enddo
!$OMP END PARALLEL DO
	 endif
	endif
	if(dl.gt.0_LONGINT.and.dr.gt.0_LONGINT.and.dc.gt.0_LONGINT) then
#ifndef NO_OMP
	 nthr=omp_get_max_threads()
#else
	 nthr=1
#endif
	 if(dr.ge.core_slope*nthr.and.(.not.no_case1)) then !the right dimension is large enough to be distributed
!	  write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_r8): kernel case/scheme: 1/",i1)') ker1 !debug
	  select case(ker1)
	  case(0)
!SCHEME 0:
	   cr=min(dr,int(max(core_slope*nthr,min_distr_seg_size),LONGINT))
	   cc=min(dc,max((arg_cache_size*int(cdim_stretch,LONGINT))/cr,1_LONGINT))
	   cl=min(dl,min(max(arg_cache_size/cc,1_LONGINT),max(arg_cache_size/cr,1_LONGINT)))
!	   write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_r8): cl,cr,cc,dl,dr,dc:",6(1x,i9))') &
!           cl,cr,cc,dl,dr,dc !debug
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(b0,b1,b2,e0r,e0,e1,e2,l0,l1,l2,ll,lr,ld,vec)
	   do b0=0_LONGINT,dc-1_LONGINT,cc
	    e0=min(b0+cc-1_LONGINT,dc-1_LONGINT)
	    e0r=mod(e0-b0+1_LONGINT,8_LONGINT)
	    do b1=0_LONGINT,dl-1_LONGINT,cl
	     e1=min(b1+cl-1_LONGINT,dl-1_LONGINT)
	     do b2=0_LONGINT,dr-1_LONGINT,cr
	      e2=min(b2+cr-1_LONGINT,dr-1_LONGINT)
!$OMP DO SCHEDULE(GUIDED)
	      do l2=b2,e2
	       ld=l2*dl
	       do l1=b1,e1
	        ll=l1*dc+b0; lr=l2*dc+b0; vec(:)=0E0_real_kind
	        do l0=b0,e0-e0r,8_LONGINT
	         vec(0)=vec(0)+ltens(ll)*rtens(lr)*alf
	         vec(1)=vec(1)+ltens(ll+1_LONGINT)*rtens(lr+1_LONGINT)*alf
	         vec(2)=vec(2)+ltens(ll+2_LONGINT)*rtens(lr+2_LONGINT)*alf
	         vec(3)=vec(3)+ltens(ll+3_LONGINT)*rtens(lr+3_LONGINT)*alf
	         vec(4)=vec(4)+ltens(ll+4_LONGINT)*rtens(lr+4_LONGINT)*alf
	         vec(5)=vec(5)+ltens(ll+5_LONGINT)*rtens(lr+5_LONGINT)*alf
	         vec(6)=vec(6)+ltens(ll+6_LONGINT)*rtens(lr+6_LONGINT)*alf
	         vec(7)=vec(7)+ltens(ll+7_LONGINT)*rtens(lr+7_LONGINT)*alf
	         ll=ll+8_LONGINT; lr=lr+8_LONGINT
	        enddo
	        do l0=0_LONGINT,e0r-1_LONGINT
	         vec(l0)=vec(l0)+ltens(ll+l0)*rtens(lr+l0)*alf
	        enddo
	        vec(0)=vec(0)+vec(4)
	        vec(1)=vec(1)+vec(5)
	        vec(2)=vec(2)+vec(6)
	        vec(3)=vec(3)+vec(7)
	        dtens(ld+l1)=dtens(ld+l1)+vec(0)+vec(1)+vec(2)+vec(3)
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
	   chunk=max(arg_cache_size/dc,1_LONGINT)
!          write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_r8): chunk,dl,dr,dc:",4(1x,i9))') chunk,dl,dr,dc !debug
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(l0,l1,l2,ll,lr,ld,ls,lf,val)
	   do ls=0_LONGINT,dl-1_LONGINT,chunk
	    lf=min(ls+chunk-1_LONGINT,dl-1_LONGINT)
!!$OMP DO SCHEDULE(DYNAMIC,chunk)
!$OMP DO SCHEDULE(GUIDED)
	    do l2=0_LONGINT,dr-1_LONGINT
	     lr=l2*dc; ld=l2*dl
	     do l1=ls,lf
	      ll=l1*dc
	      val=dtens(ld+l1)
	      do l0=0_LONGINT,dc-1_LONGINT; val=val+ltens(ll+l0)*rtens(lr+l0)*alf; enddo
	      dtens(ld+l1)=val
	     enddo
	    enddo
!$OMP END DO NOWAIT
	   enddo
!$OMP END PARALLEL
	  case(2)
!SCHEME 2:
	   chunk=max(arg_cache_size/dc,1_LONGINT)
	   if(mod(dl,chunk).ne.0) then; ls=dl/chunk+1_LONGINT; else; ls=dl/chunk; endif
	   if(mod(dr,chunk).ne.0) then; lf=dr/chunk+1_LONGINT; else; lf=dr/chunk; endif
!	   write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_r8): chunk,ls,lf,dl,dr,dc:",6(1x,i9))') &
!           chunk,ls,lf,dl,dr,dc !debug
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(b0,b1,b2,l0,l1,l2,ll,lr,ld,val) SCHEDULE(GUIDED)
	   do b0=0_LONGINT,lf*ls-1_LONGINT
	    b2=b0/ls*chunk; b1=mod(b0,ls)*chunk
	    do l2=b2,min(b2+chunk-1_LONGINT,dr-1_LONGINT)
	     lr=l2*dc; ld=l2*dl
	     do l1=b1,min(b1+chunk-1_LONGINT,dl-1_LONGINT)
	      ll=l1*dc
	      val=dtens(ld+l1)
	      do l0=0_LONGINT,dc-1_LONGINT; val=val+ltens(ll+l0)*rtens(lr+l0)*alf; enddo
	      dtens(ld+l1)=val
	     enddo
	    enddo
	   enddo
!$OMP END PARALLEL DO
	  case(3)
!SCHEME 3:
!	   write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_r8): dl,dr,dc:",3(1x,i9))') dl,dr,dc !debug
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0,l1,l2,ll,lr,ld,val) SCHEDULE(DYNAMIC)
	   do l2=0_LONGINT,dr-1_LONGINT
	    lr=l2*dc; ld=l2*dl
	    do l1=0_LONGINT,dl-1_LONGINT
	     ll=l1*dc
	     val=dtens(ld+l1)
	     do l0=0_LONGINT,dc-1_LONGINT; val=val+ltens(ll+l0)*rtens(lr+l0)*alf; enddo
	     dtens(ld+l1)=val
	    enddo
	   enddo
!$OMP END PARALLEL DO
	  case default
	   ierr=1
	  end select
	 else !dr is small
	  if(dl.ge.core_slope*nthr.and.(.not.no_case2)) then !the left dimension is large enough to be distributed
!	   write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_r8): kernel case/scheme: 2/",i1)') ker2 !debug
	   select case(ker2)
	   case(0)
!SCHEME 0:
!           write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_r8): dl,dr,dc:",3(1x,i9))') dl,dr,dc !debug
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(l0,l1,l2,b2,ll,lr,ld,e0r,vec)
            e0r=mod(dc,8_LONGINT)
	    do l2=0_LONGINT,dr-1_LONGINT
	     b2=l2*dc; ld=l2*dl
!$OMP DO SCHEDULE(GUIDED)
	     do l1=0_LONGINT,dl-1_LONGINT
	      ll=l1*dc; lr=b2; vec(:)=0E0_real_kind
	      do l0=0_LONGINT,dc-1_LONGINT-e0r,8_LONGINT
	       vec(0)=vec(0)+ltens(ll)*rtens(lr)*alf
	       vec(1)=vec(1)+ltens(ll+1_LONGINT)*rtens(lr+1_LONGINT)*alf
	       vec(2)=vec(2)+ltens(ll+2_LONGINT)*rtens(lr+2_LONGINT)*alf
	       vec(3)=vec(3)+ltens(ll+3_LONGINT)*rtens(lr+3_LONGINT)*alf
	       vec(4)=vec(4)+ltens(ll+4_LONGINT)*rtens(lr+4_LONGINT)*alf
	       vec(5)=vec(5)+ltens(ll+5_LONGINT)*rtens(lr+5_LONGINT)*alf
	       vec(6)=vec(6)+ltens(ll+6_LONGINT)*rtens(lr+6_LONGINT)*alf
	       vec(7)=vec(7)+ltens(ll+7_LONGINT)*rtens(lr+7_LONGINT)*alf
	       ll=ll+8_LONGINT; lr=lr+8_LONGINT
	      enddo
	      do l0=0_LONGINT,e0r-1_LONGINT
	       vec(l0)=vec(l0)+ltens(ll+l0)*rtens(lr+l0)*alf
	      enddo
	      vec(0)=vec(0)+vec(4)
	      vec(1)=vec(1)+vec(5)
	      vec(2)=vec(2)+vec(6)
	      vec(3)=vec(3)+vec(7)
	      dtens(ld+l1)=dtens(ld+l1)+vec(0)+vec(1)+vec(2)+vec(3)
	     enddo
!$OMP END DO NOWAIT
	    enddo
!$OMP END PARALLEL
	   case(1)
!SCHEME 1:
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0,l1,l2,ll,lr,val) SCHEDULE(GUIDED) COLLAPSE(2)
	    do l2=0_LONGINT,dr-1_LONGINT
	     do l1=0_LONGINT,dl-1_LONGINT
	      ll=l1*dc; lr=l2*dc
	      val=dtens(l2*dl+l1)
	      do l0=0_LONGINT,dc-1_LONGINT; val=val+ltens(ll+l0)*rtens(lr+l0)*alf; enddo
	      dtens(l2*dl+l1)=val
	     enddo
	    enddo
!$OMP END PARALLEL DO
	   case default
	    ierr=2
	   end select
	  else !dr & dl are both small
	   if(dc.ge.core_slope*nthr.and.(.not.no_case3)) then !the contraction dimension is large enough to be distributed
!	    write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_r8): kernel case/scheme: 3/",i1)') ker3 !debug
	    select case(ker3)
	    case(0)
!SCHEME 0:
!            write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_r8): dl,dr,dc:",3(1x,i9))') dl,dr,dc !debug
             redm(:,:)=0E0_real_kind
             do b2=0_LONGINT,dr-1_LONGINT,red_mat_size
              e2=min(red_mat_size-1_LONGINT,dr-1_LONGINT-b2)
              do b1=0_LONGINT,dl-1_LONGINT,red_mat_size
               e1=min(red_mat_size-1_LONGINT,dl-1_LONGINT-b1)
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(l0,l1,l2,ll,lr)
	       do l2=0_LONGINT,e2
	        lr=(b2+l2)*dc
	        do l1=0_LONGINT,e1
	         ll=(b1+l1)*dc
!$OMP MASTER
	         val=0E0_real_kind
!$OMP END MASTER
!$OMP BARRIER
!$OMP DO SCHEDULE(GUIDED) REDUCTION(+:val)
	         do l0=0_LONGINT,dc-1_LONGINT; val=val+ltens(ll+l0)*rtens(lr+l0)*alf; enddo
!$OMP END DO
!$OMP MASTER
		 redm(l1,l2)=val
!$OMP END MASTER
	        enddo
	       enddo
!$OMP END PARALLEL
	       do l2=0_LONGINT,e2
	        ld=(b2+l2)*dl
	        do l1=0_LONGINT,e1
	         dtens(ld+b1+l1)=dtens(ld+b1+l1)+redm(l1,l2)
	        enddo
	       enddo
	      enddo
	     enddo
	    case default
	     ierr=3
	    end select
	   else !dr & dl & dc are all small
	    if(dr*dl.ge.core_slope*nthr.and.(.not.no_case4)) then !the destination matrix is large enough to be distributed
!	     write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_r8): kernel case: 4")') !debug
!	     write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_r8): dl,dr,dc:",3(1x,i9))') dl,dr,dc !debug
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0,l1,l2,ll,lr,val) SCHEDULE(GUIDED) COLLAPSE(2)
	     do l2=0_LONGINT,dr-1_LONGINT
	      do l1=0_LONGINT,dl-1_LONGINT
	       ll=l1*dc; lr=l2*dc
	       val=dtens(l2*dl+l1)
	       do l0=0_LONGINT,dc-1_LONGINT; val=val+ltens(ll+l0)*rtens(lr+l0)*alf; enddo
	       dtens(l2*dl+l1)=val
	      enddo
	     enddo
!$OMP END PARALLEL DO
	    else !all matrices are very small (serial execution)
!	     write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_r8): kernel case: 5 (serial)")') !debug
!	     write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_r8): dl,dr,dc:",3(1x,i9))') dl,dr,dc !debug
	     do l2=0_LONGINT,dr-1_LONGINT
	      lr=l2*dc; ld=l2*dl
	      do l1=0_LONGINT,dl-1_LONGINT
	       ll=l1*dc
	       val=dtens(ld+l1)
	       do l0=0_LONGINT,dc-1_LONGINT; val=val+ltens(ll+l0)*rtens(lr+l0)*alf; enddo
	       dtens(ld+l1)=val
	      enddo
	     enddo
	    endif
	   endif
	  endif
	 endif
	else
	 ierr=4
	endif
!	tm=thread_wtime(time_beg) !debug
!	write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_r8): time/speed/error = ",2(F10.4,1x),i3)') &
!        tm,2d0*dble(dr*dl*dc)/(tm*1024d0*1024d0*1024d0),ierr !debug
	return
	end subroutine tensor_block_pcontract_dlf_r8
!-------------------------------------------------------------------------------------------
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: tensor_block_pcontract_dlf_c4
#endif
	subroutine tensor_block_pcontract_dlf_c4(dl,dr,dc,ltens,rtens,dtens,ierr,alpha,beta) !PARALLEL
!This subroutine multiplies two matrices derived from the corresponding tensors by index permutations:
!dtens(0:dl-1,0:dr-1)+=ltens(0:dc-1,0:dl-1)*rtens(0:dc-1,0:dr-1)*alpha
!The result is a matrix as well (cannot be a scalar, see tensor_block_fcontract).
	implicit none
!---------------------------------------
	integer, parameter:: real_kind=4                   !real data kind
	integer(LONGINT), parameter:: red_mat_size=32      !the size of the local reduction matrix
	integer(LONGINT), parameter:: arg_cache_size=2**15 !cache-size dependent parameter (increase it as there are more cores per node)
	integer, parameter:: min_distr_seg_size=128  !min segment size of an omp distributed dimension
	integer, parameter:: cdim_stretch=2          !makes the segmentation of the contracted dimension coarser
	integer, parameter:: core_slope=16           !regulates the slope of the segment size of the distributed dimension w.r.t. the number of cores
	integer, parameter:: ker1=0                  !kernel 1 scheme #
	integer, parameter:: ker2=0                  !kernel 2 scheme #
	integer, parameter:: ker3=0                  !kernel 3 scheme #
!----------------------------------
	logical, parameter:: no_case1=.FALSE.
	logical, parameter:: no_case2=.FALSE.
	logical, parameter:: no_case3=.FALSE.
	logical, parameter:: no_case4=.FALSE.
!----------------------------------------------
	integer(LONGINT), intent(in):: dl,dr,dc !matrix dimensions
	complex(real_kind), intent(in):: ltens(0:*),rtens(0:*) !input arguments
	complex(real_kind), intent(inout):: dtens(0:*) !output argument
	integer, intent(inout):: ierr !error code
	complex(real_kind), intent(in), optional:: alpha !BLAS alpha
	complex(real_kind), intent(in), optional:: beta  !BLAS beta (defaults to 1)
	integer i,j,k,l,m,n,nthr
	integer(LONGINT) ll,lr,ld,l0,l1,l2,b0,b1,b2,e0r,e0,e1,e2,ls,lf,cl,cr,cc,chunk
	complex(real_kind) vec(0:7),redm(0:red_mat_size-1,0:red_mat_size-1),val,alf !`thread private (redm)?
	real(8) time_beg,tm
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: real_kind,red_mat_size,arg_cache_size,min_distr_seg_size,cdim_stretch,core_slope,ker1,ker2,ker3
!DIR$ ATTRIBUTES OFFLOAD:mic:: no_case1,no_case2,no_case3,no_case4
!DIR$ ATTRIBUTES ALIGN:128:: real_kind,red_mat_size,arg_cache_size,min_distr_seg_size,cdim_stretch,core_slope,ker1,ker2,ker3
!DIR$ ATTRIBUTES ALIGN:128:: no_case1,no_case2,no_case3,no_case4,vec,redm
#endif

	ierr=0
!	time_beg=thread_wtime() !debug
	if(present(alpha)) then; alf=alpha; else; alf=(1d0,0d0); endif
	if(present(beta)) then !rescale output tensor if requested
	 if(beta.ne.(1.0,0.0)) then
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED)
	  do l0=0_LONGINT,dr*dl-1_LONGINT
	   dtens(l0)=dtens(l0)*beta
	  enddo
!$OMP END PARALLEL DO
	 endif
	endif
	if(dl.gt.0_LONGINT.and.dr.gt.0_LONGINT.and.dc.gt.0_LONGINT) then
#ifndef NO_OMP
	 nthr=omp_get_max_threads()
#else
	 nthr=1
#endif
	 if(dr.ge.core_slope*nthr.and.(.not.no_case1)) then !the right dimension is large enough to be distributed
!	  write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_c4): kernel case/scheme: 1/",i1)') ker1 !debug
	  select case(ker1)
	  case(0)
!SCHEME 0:
	   cr=min(dr,int(max(core_slope*nthr,min_distr_seg_size),LONGINT))
	   cc=min(dc,max((arg_cache_size*int(cdim_stretch,LONGINT))/cr,1_LONGINT))
	   cl=min(dl,min(max(arg_cache_size/cc,1_LONGINT),max(arg_cache_size/cr,1_LONGINT)))
!	   write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_c4): cl,cr,cc,dl,dr,dc:",6(1x,i9))') &
!           cl,cr,cc,dl,dr,dc !debug
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(b0,b1,b2,e0r,e0,e1,e2,l0,l1,l2,ll,lr,ld,vec)
	   do b0=0_LONGINT,dc-1_LONGINT,cc
	    e0=min(b0+cc-1_LONGINT,dc-1_LONGINT)
	    e0r=mod(e0-b0+1_LONGINT,8_LONGINT)
	    do b1=0_LONGINT,dl-1_LONGINT,cl
	     e1=min(b1+cl-1_LONGINT,dl-1_LONGINT)
	     do b2=0_LONGINT,dr-1_LONGINT,cr
	      e2=min(b2+cr-1_LONGINT,dr-1_LONGINT)
!$OMP DO SCHEDULE(GUIDED)
	      do l2=b2,e2
	       ld=l2*dl
	       do l1=b1,e1
	        ll=l1*dc+b0; lr=l2*dc+b0; vec(:)=cmplx(0E0_real_kind,0E0_real_kind,kind=real_kind)
	        do l0=b0,e0-e0r,8_LONGINT
	         vec(0)=vec(0)+ltens(ll)*rtens(lr)*alf
	         vec(1)=vec(1)+ltens(ll+1_LONGINT)*rtens(lr+1_LONGINT)*alf
	         vec(2)=vec(2)+ltens(ll+2_LONGINT)*rtens(lr+2_LONGINT)*alf
	         vec(3)=vec(3)+ltens(ll+3_LONGINT)*rtens(lr+3_LONGINT)*alf
	         vec(4)=vec(4)+ltens(ll+4_LONGINT)*rtens(lr+4_LONGINT)*alf
	         vec(5)=vec(5)+ltens(ll+5_LONGINT)*rtens(lr+5_LONGINT)*alf
	         vec(6)=vec(6)+ltens(ll+6_LONGINT)*rtens(lr+6_LONGINT)*alf
	         vec(7)=vec(7)+ltens(ll+7_LONGINT)*rtens(lr+7_LONGINT)*alf
	         ll=ll+8_LONGINT; lr=lr+8_LONGINT
	        enddo
	        do l0=0_LONGINT,e0r-1_LONGINT
	         vec(l0)=vec(l0)+ltens(ll+l0)*rtens(lr+l0)*alf
	        enddo
	        vec(0)=vec(0)+vec(4)
	        vec(1)=vec(1)+vec(5)
	        vec(2)=vec(2)+vec(6)
	        vec(3)=vec(3)+vec(7)
	        dtens(ld+l1)=dtens(ld+l1)+vec(0)+vec(1)+vec(2)+vec(3)
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
	   chunk=max(arg_cache_size/dc,1_LONGINT)
!          write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_c4): chunk,dl,dr,dc:",4(1x,i9))') chunk,dl,dr,dc !debug
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(l0,l1,l2,ll,lr,ld,ls,lf,val)
	   do ls=0_LONGINT,dl-1_LONGINT,chunk
	    lf=min(ls+chunk-1_LONGINT,dl-1_LONGINT)
!!$OMP DO SCHEDULE(DYNAMIC,chunk)
!$OMP DO SCHEDULE(GUIDED)
	    do l2=0_LONGINT,dr-1_LONGINT
	     lr=l2*dc; ld=l2*dl
	     do l1=ls,lf
	      ll=l1*dc
	      val=dtens(ld+l1)
	      do l0=0_LONGINT,dc-1_LONGINT; val=val+ltens(ll+l0)*rtens(lr+l0)*alf; enddo
	      dtens(ld+l1)=val
	     enddo
	    enddo
!$OMP END DO NOWAIT
	   enddo
!$OMP END PARALLEL
	  case(2)
!SCHEME 2:
	   chunk=max(arg_cache_size/dc,1_LONGINT)
	   if(mod(dl,chunk).ne.0) then; ls=dl/chunk+1_LONGINT; else; ls=dl/chunk; endif
	   if(mod(dr,chunk).ne.0) then; lf=dr/chunk+1_LONGINT; else; lf=dr/chunk; endif
!	   write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_c4): chunk,ls,lf,dl,dr,dc:",6(1x,i9))') &
!           chunk,ls,lf,dl,dr,dc !debug
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(b0,b1,b2,l0,l1,l2,ll,lr,ld,val) SCHEDULE(GUIDED)
	   do b0=0_LONGINT,lf*ls-1_LONGINT
	    b2=b0/ls*chunk; b1=mod(b0,ls)*chunk
	    do l2=b2,min(b2+chunk-1_LONGINT,dr-1_LONGINT)
	     lr=l2*dc; ld=l2*dl
	     do l1=b1,min(b1+chunk-1_LONGINT,dl-1_LONGINT)
	      ll=l1*dc
	      val=dtens(ld+l1)
	      do l0=0_LONGINT,dc-1_LONGINT; val=val+ltens(ll+l0)*rtens(lr+l0)*alf; enddo
	      dtens(ld+l1)=val
	     enddo
	    enddo
	   enddo
!$OMP END PARALLEL DO
	  case(3)
!SCHEME 3:
!	   write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_c4): dl,dr,dc:",3(1x,i9))') dl,dr,dc !debug
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0,l1,l2,ll,lr,ld,val) SCHEDULE(DYNAMIC)
	   do l2=0_LONGINT,dr-1_LONGINT
	    lr=l2*dc; ld=l2*dl
	    do l1=0_LONGINT,dl-1_LONGINT
	     ll=l1*dc
	     val=dtens(ld+l1)
	     do l0=0_LONGINT,dc-1_LONGINT; val=val+ltens(ll+l0)*rtens(lr+l0)*alf; enddo
	     dtens(ld+l1)=val
	    enddo
	   enddo
!$OMP END PARALLEL DO
	  case default
	   ierr=1
	  end select
	 else !dr is small
	  if(dl.ge.core_slope*nthr.and.(.not.no_case2)) then !the left dimension is large enough to be distributed
!	   write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_c4): kernel case/scheme: 2/",i1)') ker2 !debug
	   select case(ker2)
	   case(0)
!SCHEME 0:
!           write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_c4): dl,dr,dc:",3(1x,i9))') dl,dr,dc !debug
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(l0,l1,l2,b2,ll,lr,ld,e0r,vec)
            e0r=mod(dc,8_LONGINT)
	    do l2=0_LONGINT,dr-1_LONGINT
	     b2=l2*dc; ld=l2*dl
!$OMP DO SCHEDULE(GUIDED)
	     do l1=0_LONGINT,dl-1_LONGINT
	      ll=l1*dc; lr=b2; vec(:)=cmplx(0E0_real_kind,0E0_real_kind,kind=real_kind)
	      do l0=0_LONGINT,dc-1_LONGINT-e0r,8_LONGINT
	       vec(0)=vec(0)+ltens(ll)*rtens(lr)*alf
	       vec(1)=vec(1)+ltens(ll+1_LONGINT)*rtens(lr+1_LONGINT)*alf
	       vec(2)=vec(2)+ltens(ll+2_LONGINT)*rtens(lr+2_LONGINT)*alf
	       vec(3)=vec(3)+ltens(ll+3_LONGINT)*rtens(lr+3_LONGINT)*alf
	       vec(4)=vec(4)+ltens(ll+4_LONGINT)*rtens(lr+4_LONGINT)*alf
	       vec(5)=vec(5)+ltens(ll+5_LONGINT)*rtens(lr+5_LONGINT)*alf
	       vec(6)=vec(6)+ltens(ll+6_LONGINT)*rtens(lr+6_LONGINT)*alf
	       vec(7)=vec(7)+ltens(ll+7_LONGINT)*rtens(lr+7_LONGINT)*alf
	       ll=ll+8_LONGINT; lr=lr+8_LONGINT
	      enddo
	      do l0=0_LONGINT,e0r-1_LONGINT
	       vec(l0)=vec(l0)+ltens(ll+l0)*rtens(lr+l0)*alf
	      enddo
	      vec(0)=vec(0)+vec(4)
	      vec(1)=vec(1)+vec(5)
	      vec(2)=vec(2)+vec(6)
	      vec(3)=vec(3)+vec(7)
	      dtens(ld+l1)=dtens(ld+l1)+vec(0)+vec(1)+vec(2)+vec(3)
	     enddo
!$OMP END DO NOWAIT
	    enddo
!$OMP END PARALLEL
	   case(1)
!SCHEME 1:
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0,l1,l2,ll,lr,val) SCHEDULE(GUIDED) COLLAPSE(2)
	    do l2=0_LONGINT,dr-1_LONGINT
	     do l1=0_LONGINT,dl-1_LONGINT
	      ll=l1*dc; lr=l2*dc
	      val=dtens(l2*dl+l1)
	      do l0=0_LONGINT,dc-1_LONGINT; val=val+ltens(ll+l0)*rtens(lr+l0)*alf; enddo
	      dtens(l2*dl+l1)=val
	     enddo
	    enddo
!$OMP END PARALLEL DO
	   case default
	    ierr=2
	   end select
	  else !dr & dl are both small
	   if(dc.ge.core_slope*nthr.and.(.not.no_case3)) then !the contraction dimension is large enough to be distributed
!	    write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_c4): kernel case/scheme: 3/",i1)') ker3 !debug
	    select case(ker3)
	    case(0)
!SCHEME 0:
!            write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_c4): dl,dr,dc:",3(1x,i9))') dl,dr,dc !debug
             redm(:,:)=cmplx(0E0_real_kind,0E0_real_kind,kind=real_kind)
             do b2=0_LONGINT,dr-1_LONGINT,red_mat_size
              e2=min(red_mat_size-1_LONGINT,dr-1_LONGINT-b2)
              do b1=0_LONGINT,dl-1_LONGINT,red_mat_size
               e1=min(red_mat_size-1_LONGINT,dl-1_LONGINT-b1)
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(l0,l1,l2,ll,lr)
	       do l2=0_LONGINT,e2
	        lr=(b2+l2)*dc
	        do l1=0_LONGINT,e1
	         ll=(b1+l1)*dc
!$OMP MASTER
	         val=cmplx(0E0_real_kind,0E0_real_kind,kind=real_kind)
!$OMP END MASTER
!$OMP BARRIER
!$OMP DO SCHEDULE(GUIDED) REDUCTION(+:val)
	         do l0=0_LONGINT,dc-1_LONGINT; val=val+ltens(ll+l0)*rtens(lr+l0)*alf; enddo
!$OMP END DO
!$OMP MASTER
		 redm(l1,l2)=val
!$OMP END MASTER
	        enddo
	       enddo
!$OMP END PARALLEL
	       do l2=0_LONGINT,e2
	        ld=(b2+l2)*dl
	        do l1=0_LONGINT,e1
	         dtens(ld+b1+l1)=dtens(ld+b1+l1)+redm(l1,l2)
	        enddo
	       enddo
	      enddo
	     enddo
	    case default
	     ierr=3
	    end select
	   else !dr & dl & dc are all small
	    if(dr*dl.ge.core_slope*nthr.and.(.not.no_case4)) then !the destination matrix is large enough to be distributed
!	     write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_c4): kernel case: 4")') !debug
!	     write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_c4): dl,dr,dc:",3(1x,i9))') dl,dr,dc !debug
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0,l1,l2,ll,lr,val) SCHEDULE(GUIDED) COLLAPSE(2)
	     do l2=0_LONGINT,dr-1_LONGINT
	      do l1=0_LONGINT,dl-1_LONGINT
	       ll=l1*dc; lr=l2*dc
	       val=dtens(l2*dl+l1)
	       do l0=0_LONGINT,dc-1_LONGINT; val=val+ltens(ll+l0)*rtens(lr+l0)*alf; enddo
	       dtens(l2*dl+l1)=val
	      enddo
	     enddo
!$OMP END PARALLEL DO
	    else !all matrices are very small (serial execution)
!	     write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_c4): kernel case: 5 (serial)")') !debug
!	     write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_c4): dl,dr,dc:",3(1x,i9))') dl,dr,dc !debug
	     do l2=0_LONGINT,dr-1_LONGINT
	      lr=l2*dc; ld=l2*dl
	      do l1=0_LONGINT,dl-1_LONGINT
	       ll=l1*dc
	       val=dtens(ld+l1)
	       do l0=0_LONGINT,dc-1_LONGINT; val=val+ltens(ll+l0)*rtens(lr+l0)*alf; enddo
	       dtens(ld+l1)=val
	      enddo
	     enddo
	    endif
	   endif
	  endif
	 endif
	else
	 ierr=4
	endif
!	tm=thread_wtime(time_beg) !debug
!	write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_c4): time/speed/error = ",2(F10.4,1x),i3)') &
!        tm,8d0*dble(dr*dl*dc)/(tm*1024d0*1024d0*1024d0),ierr !debug
	return
	end subroutine tensor_block_pcontract_dlf_c4
!-------------------------------------------------------------------------------------------
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: tensor_block_pcontract_dlf_c8
#endif
	subroutine tensor_block_pcontract_dlf_c8(dl,dr,dc,ltens,rtens,dtens,ierr,alpha,beta) !PARALLEL
!This subroutine multiplies two matrices derived from the corresponding tensors by index permutations:
!dtens(0:dl-1,0:dr-1)+=ltens(0:dc-1,0:dl-1)*rtens(0:dc-1,0:dr-1)*alpha
!The result is a matrix as well (cannot be a scalar, see tensor_block_fcontract).
	implicit none
!---------------------------------------
	integer, parameter:: real_kind=8                   !real data kind
	integer(LONGINT), parameter:: red_mat_size=32      !the size of the local reduction matrix
	integer(LONGINT), parameter:: arg_cache_size=2**15 !cache-size dependent parameter (increase it as there are more cores per node)
	integer, parameter:: min_distr_seg_size=128  !min segment size of an omp distributed dimension
	integer, parameter:: cdim_stretch=2          !makes the segmentation of the contracted dimension coarser
	integer, parameter:: core_slope=16           !regulates the slope of the segment size of the distributed dimension w.r.t. the number of cores
	integer, parameter:: ker1=0                  !kernel 1 scheme #
	integer, parameter:: ker2=0                  !kernel 2 scheme #
	integer, parameter:: ker3=0                  !kernel 3 scheme #
!----------------------------------
	logical, parameter:: no_case1=.FALSE.
	logical, parameter:: no_case2=.FALSE.
	logical, parameter:: no_case3=.FALSE.
	logical, parameter:: no_case4=.FALSE.
!----------------------------------------------
	integer(LONGINT), intent(in):: dl,dr,dc !matrix dimensions
	complex(real_kind), intent(in):: ltens(0:*),rtens(0:*) !input arguments
	complex(real_kind), intent(inout):: dtens(0:*) !output argument
	integer, intent(inout):: ierr !error code
	complex(real_kind), intent(in), optional:: alpha !BLAS alpha
	complex(real_kind), intent(in), optional:: beta  !BLAS beta (defaults to 1)
	integer i,j,k,l,m,n,nthr
	integer(LONGINT) ll,lr,ld,l0,l1,l2,b0,b1,b2,e0r,e0,e1,e2,ls,lf,cl,cr,cc,chunk
	complex(real_kind) vec(0:7),redm(0:red_mat_size-1,0:red_mat_size-1),val,alf !`thread private (redm)?
	real(8) time_beg,tm
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: real_kind,red_mat_size,arg_cache_size,min_distr_seg_size,cdim_stretch,core_slope,ker1,ker2,ker3
!DIR$ ATTRIBUTES OFFLOAD:mic:: no_case1,no_case2,no_case3,no_case4
!DIR$ ATTRIBUTES ALIGN:128:: real_kind,red_mat_size,arg_cache_size,min_distr_seg_size,cdim_stretch,core_slope,ker1,ker2,ker3
!DIR$ ATTRIBUTES ALIGN:128:: no_case1,no_case2,no_case3,no_case4,vec,redm
#endif

	ierr=0
!	time_beg=thread_wtime() !debug
	if(present(alpha)) then; alf=alpha; else; alf=(1d0,0d0); endif
	if(present(beta)) then !rescale output tensor if requested
	 if(beta.ne.(1d0,0d0)) then
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0) SCHEDULE(GUIDED)
	  do l0=0_LONGINT,dr*dl-1_LONGINT
	   dtens(l0)=dtens(l0)*beta
	  enddo
!$OMP END PARALLEL DO
	 endif
	endif
	if(dl.gt.0_LONGINT.and.dr.gt.0_LONGINT.and.dc.gt.0_LONGINT) then
#ifndef NO_OMP
	 nthr=omp_get_max_threads()
#else
	 nthr=1
#endif
	 if(dr.ge.core_slope*nthr.and.(.not.no_case1)) then !the right dimension is large enough to be distributed
!	  write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_c8): kernel case/scheme: 1/",i1)') ker1 !debug
	  select case(ker1)
	  case(0)
!SCHEME 0:
	   cr=min(dr,int(max(core_slope*nthr,min_distr_seg_size),LONGINT))
	   cc=min(dc,max((arg_cache_size*int(cdim_stretch,LONGINT))/cr,1_LONGINT))
	   cl=min(dl,min(max(arg_cache_size/cc,1_LONGINT),max(arg_cache_size/cr,1_LONGINT)))
!	   write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_c8): cl,cr,cc,dl,dr,dc:",6(1x,i9))') &
!           cl,cr,cc,dl,dr,dc !debug
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(b0,b1,b2,e0r,e0,e1,e2,l0,l1,l2,ll,lr,ld,vec)
	   do b0=0_LONGINT,dc-1_LONGINT,cc
	    e0=min(b0+cc-1_LONGINT,dc-1_LONGINT)
	    e0r=mod(e0-b0+1_LONGINT,8_LONGINT)
	    do b1=0_LONGINT,dl-1_LONGINT,cl
	     e1=min(b1+cl-1_LONGINT,dl-1_LONGINT)
	     do b2=0_LONGINT,dr-1_LONGINT,cr
	      e2=min(b2+cr-1_LONGINT,dr-1_LONGINT)
!$OMP DO SCHEDULE(GUIDED)
	      do l2=b2,e2
	       ld=l2*dl
	       do l1=b1,e1
	        ll=l1*dc+b0; lr=l2*dc+b0; vec(:)=cmplx(0E0_real_kind,0E0_real_kind,kind=real_kind)
	        do l0=b0,e0-e0r,8_LONGINT
	         vec(0)=vec(0)+ltens(ll)*rtens(lr)*alf
	         vec(1)=vec(1)+ltens(ll+1_LONGINT)*rtens(lr+1_LONGINT)*alf
	         vec(2)=vec(2)+ltens(ll+2_LONGINT)*rtens(lr+2_LONGINT)*alf
	         vec(3)=vec(3)+ltens(ll+3_LONGINT)*rtens(lr+3_LONGINT)*alf
	         vec(4)=vec(4)+ltens(ll+4_LONGINT)*rtens(lr+4_LONGINT)*alf
	         vec(5)=vec(5)+ltens(ll+5_LONGINT)*rtens(lr+5_LONGINT)*alf
	         vec(6)=vec(6)+ltens(ll+6_LONGINT)*rtens(lr+6_LONGINT)*alf
	         vec(7)=vec(7)+ltens(ll+7_LONGINT)*rtens(lr+7_LONGINT)*alf
	         ll=ll+8_LONGINT; lr=lr+8_LONGINT
	        enddo
	        do l0=0_LONGINT,e0r-1_LONGINT
	         vec(l0)=vec(l0)+ltens(ll+l0)*rtens(lr+l0)*alf
	        enddo
	        vec(0)=vec(0)+vec(4)
	        vec(1)=vec(1)+vec(5)
	        vec(2)=vec(2)+vec(6)
	        vec(3)=vec(3)+vec(7)
	        dtens(ld+l1)=dtens(ld+l1)+vec(0)+vec(1)+vec(2)+vec(3)
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
	   chunk=max(arg_cache_size/dc,1_LONGINT)
!          write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_c8): chunk,dl,dr,dc:",4(1x,i9))') chunk,dl,dr,dc !debug
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(l0,l1,l2,ll,lr,ld,ls,lf,val)
	   do ls=0_LONGINT,dl-1_LONGINT,chunk
	    lf=min(ls+chunk-1_LONGINT,dl-1_LONGINT)
!!$OMP DO SCHEDULE(DYNAMIC,chunk)
!$OMP DO SCHEDULE(GUIDED)
	    do l2=0_LONGINT,dr-1_LONGINT
	     lr=l2*dc; ld=l2*dl
	     do l1=ls,lf
	      ll=l1*dc
	      val=dtens(ld+l1)
	      do l0=0_LONGINT,dc-1_LONGINT; val=val+ltens(ll+l0)*rtens(lr+l0)*alf; enddo
	      dtens(ld+l1)=val
	     enddo
	    enddo
!$OMP END DO NOWAIT
	   enddo
!$OMP END PARALLEL
	  case(2)
!SCHEME 2:
	   chunk=max(arg_cache_size/dc,1_LONGINT)
	   if(mod(dl,chunk).ne.0) then; ls=dl/chunk+1_LONGINT; else; ls=dl/chunk; endif
	   if(mod(dr,chunk).ne.0) then; lf=dr/chunk+1_LONGINT; else; lf=dr/chunk; endif
!	   write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_c8): chunk,ls,lf,dl,dr,dc:",6(1x,i9))') &
!           chunk,ls,lf,dl,dr,dc !debug
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(b0,b1,b2,l0,l1,l2,ll,lr,ld,val) SCHEDULE(GUIDED)
	   do b0=0_LONGINT,lf*ls-1_LONGINT
	    b2=b0/ls*chunk; b1=mod(b0,ls)*chunk
	    do l2=b2,min(b2+chunk-1_LONGINT,dr-1_LONGINT)
	     lr=l2*dc; ld=l2*dl
	     do l1=b1,min(b1+chunk-1_LONGINT,dl-1_LONGINT)
	      ll=l1*dc
	      val=dtens(ld+l1)
	      do l0=0_LONGINT,dc-1_LONGINT; val=val+ltens(ll+l0)*rtens(lr+l0)*alf; enddo
	      dtens(ld+l1)=val
	     enddo
	    enddo
	   enddo
!$OMP END PARALLEL DO
	  case(3)
!SCHEME 3:
!	   write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_c8): dl,dr,dc:",3(1x,i9))') dl,dr,dc !debug
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0,l1,l2,ll,lr,ld,val) SCHEDULE(DYNAMIC)
	   do l2=0_LONGINT,dr-1_LONGINT
	    lr=l2*dc; ld=l2*dl
	    do l1=0_LONGINT,dl-1_LONGINT
	     ll=l1*dc
	     val=dtens(ld+l1)
	     do l0=0_LONGINT,dc-1_LONGINT; val=val+ltens(ll+l0)*rtens(lr+l0)*alf; enddo
	     dtens(ld+l1)=val
	    enddo
	   enddo
!$OMP END PARALLEL DO
	  case default
	   ierr=1
	  end select
	 else !dr is small
	  if(dl.ge.core_slope*nthr.and.(.not.no_case2)) then !the left dimension is large enough to be distributed
!	   write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_c8): kernel case/scheme: 2/",i1)') ker2 !debug
	   select case(ker2)
	   case(0)
!SCHEME 0:
!           write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_c8): dl,dr,dc:",3(1x,i9))') dl,dr,dc !debug
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(l0,l1,l2,b2,ll,lr,ld,e0r,vec)
            e0r=mod(dc,8_LONGINT)
	    do l2=0_LONGINT,dr-1_LONGINT
	     b2=l2*dc; ld=l2*dl
!$OMP DO SCHEDULE(GUIDED)
	     do l1=0_LONGINT,dl-1_LONGINT
	      ll=l1*dc; lr=b2; vec(:)=cmplx(0E0_real_kind,0E0_real_kind,kind=real_kind)
	      do l0=0_LONGINT,dc-1_LONGINT-e0r,8_LONGINT
	       vec(0)=vec(0)+ltens(ll)*rtens(lr)*alf
	       vec(1)=vec(1)+ltens(ll+1_LONGINT)*rtens(lr+1_LONGINT)*alf
	       vec(2)=vec(2)+ltens(ll+2_LONGINT)*rtens(lr+2_LONGINT)*alf
	       vec(3)=vec(3)+ltens(ll+3_LONGINT)*rtens(lr+3_LONGINT)*alf
	       vec(4)=vec(4)+ltens(ll+4_LONGINT)*rtens(lr+4_LONGINT)*alf
	       vec(5)=vec(5)+ltens(ll+5_LONGINT)*rtens(lr+5_LONGINT)*alf
	       vec(6)=vec(6)+ltens(ll+6_LONGINT)*rtens(lr+6_LONGINT)*alf
	       vec(7)=vec(7)+ltens(ll+7_LONGINT)*rtens(lr+7_LONGINT)*alf
	       ll=ll+8_LONGINT; lr=lr+8_LONGINT
	      enddo
	      do l0=0_LONGINT,e0r-1_LONGINT
	       vec(l0)=vec(l0)+ltens(ll+l0)*rtens(lr+l0)*alf
	      enddo
	      vec(0)=vec(0)+vec(4)
	      vec(1)=vec(1)+vec(5)
	      vec(2)=vec(2)+vec(6)
	      vec(3)=vec(3)+vec(7)
	      dtens(ld+l1)=dtens(ld+l1)+vec(0)+vec(1)+vec(2)+vec(3)
	     enddo
!$OMP END DO NOWAIT
	    enddo
!$OMP END PARALLEL
	   case(1)
!SCHEME 1:
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0,l1,l2,ll,lr,val) SCHEDULE(GUIDED) COLLAPSE(2)
	    do l2=0_LONGINT,dr-1_LONGINT
	     do l1=0_LONGINT,dl-1_LONGINT
	      ll=l1*dc; lr=l2*dc
	      val=dtens(l2*dl+l1)
	      do l0=0_LONGINT,dc-1_LONGINT; val=val+ltens(ll+l0)*rtens(lr+l0)*alf; enddo
	      dtens(l2*dl+l1)=val
	     enddo
	    enddo
!$OMP END PARALLEL DO
	   case default
	    ierr=2
	   end select
	  else !dr & dl are both small
	   if(dc.ge.core_slope*nthr.and.(.not.no_case3)) then !the contraction dimension is large enough to be distributed
!	    write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_c8): kernel case/scheme: 3/",i1)') ker3 !debug
	    select case(ker3)
	    case(0)
!SCHEME 0:
!            write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_c8): dl,dr,dc:",3(1x,i9))') dl,dr,dc !debug
             redm(:,:)=cmplx(0E0_real_kind,0E0_real_kind,kind=real_kind)
             do b2=0_LONGINT,dr-1_LONGINT,red_mat_size
              e2=min(red_mat_size-1_LONGINT,dr-1_LONGINT-b2)
              do b1=0_LONGINT,dl-1_LONGINT,red_mat_size
               e1=min(red_mat_size-1_LONGINT,dl-1_LONGINT-b1)
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(l0,l1,l2,ll,lr)
	       do l2=0_LONGINT,e2
	        lr=(b2+l2)*dc
	        do l1=0_LONGINT,e1
	         ll=(b1+l1)*dc
!$OMP MASTER
	         val=cmplx(0E0_real_kind,0E0_real_kind,kind=real_kind)
!$OMP END MASTER
!$OMP BARRIER
!$OMP DO SCHEDULE(GUIDED) REDUCTION(+:val)
	         do l0=0_LONGINT,dc-1_LONGINT; val=val+ltens(ll+l0)*rtens(lr+l0)*alf; enddo
!$OMP END DO
!$OMP MASTER
		 redm(l1,l2)=val
!$OMP END MASTER
	        enddo
	       enddo
!$OMP END PARALLEL
	       do l2=0_LONGINT,e2
	        ld=(b2+l2)*dl
	        do l1=0_LONGINT,e1
	         dtens(ld+b1+l1)=dtens(ld+b1+l1)+redm(l1,l2)
	        enddo
	       enddo
	      enddo
	     enddo
	    case default
	     ierr=3
	    end select
	   else !dr & dl & dc are all small
	    if(dr*dl.ge.core_slope*nthr.and.(.not.no_case4)) then !the destination matrix is large enough to be distributed
!	     write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_c8): kernel case: 4")') !debug
!	     write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_c8): dl,dr,dc:",3(1x,i9))') dl,dr,dc !debug
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(l0,l1,l2,ll,lr,val) SCHEDULE(GUIDED) COLLAPSE(2)
	     do l2=0_LONGINT,dr-1_LONGINT
	      do l1=0_LONGINT,dl-1_LONGINT
	       ll=l1*dc; lr=l2*dc
	       val=dtens(l2*dl+l1)
	       do l0=0_LONGINT,dc-1_LONGINT; val=val+ltens(ll+l0)*rtens(lr+l0)*alf; enddo
	       dtens(l2*dl+l1)=val
	      enddo
	     enddo
!$OMP END PARALLEL DO
	    else !all matrices are very small (serial execution)
!	     write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_c8): kernel case: 5 (serial)")') !debug
!	     write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_c8): dl,dr,dc:",3(1x,i9))') dl,dr,dc !debug
	     do l2=0_LONGINT,dr-1_LONGINT
	      lr=l2*dc; ld=l2*dl
	      do l1=0_LONGINT,dl-1_LONGINT
	       ll=l1*dc
	       val=dtens(ld+l1)
	       do l0=0_LONGINT,dc-1_LONGINT; val=val+ltens(ll+l0)*rtens(lr+l0)*alf; enddo
	       dtens(ld+l1)=val
	      enddo
	     enddo
	    endif
	   endif
	  endif
	 endif
	else
	 ierr=4
	endif
!	tm=thread_wtime(time_beg) !debug
!	write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_pcontract_dlf_c8): time/speed/error = ",2(F10.4,1x),i3)') &
!        tm,8d0*dble(dr*dl*dc)/(tm*1024d0*1024d0*1024d0),ierr !debug
	return
	end subroutine tensor_block_pcontract_dlf_c8
!------------------------------------------------------------------------------------------------------
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: tensor_block_ftrace_dlf_r4
#endif
	subroutine tensor_block_ftrace_dlf_r4(contr_ptrn,ord_rest,tens_in,rank_in,dims_in,val_out,ierr) !PARALLEL
!This subroutine takes a full trace in a tensor block and accumulates it into a scalar.
!A full trace consists of one or more pairwise index contractions such that no single index is left uncontracted.
!Consequently, only even rank tensor blocks can be passed here.
!INPUT:
! - contr_ptrn(1:rank_in) - index contraction pattern;
! - ord_rest(1:rank_in) - index ordering restrictions (for contracted indices);
! - tens_in - input tensor block;
! - rank_in - rank of <tens_in>;
! - dims_in(1:rank_in) - dimension extents of <tens_in>;
! - val_out - initialized! scalar;
!OUTPUT:
! - val_out - modified scalar (the trace has been accumulated in);
! - ierr - error code (0:success).
!NOTES:
! - The algorithm used here is not cache-efficient, and I doubt there is any (D.I.L.).
! - No thorough argument checks.
!`Enable index ordering restrictions.
	implicit none
!---------------------------------------
	integer, parameter:: real_kind=4
!---------------------------------------
	integer, intent(in):: rank_in,contr_ptrn(1:rank_in),ord_rest(1:rank_in),dims_in(1:rank_in)
	real(real_kind), intent(in):: tens_in(0:*)
	real(real_kind), intent(inout):: val_out
	integer, intent(inout):: ierr
	integer i,j,k,l,m,n,ks,kf,im(1:rank_in),ic(1:rank_in)
	integer(LONGINT) bases_in(1:rank_in),bases_tr(1:rank_in),segs(0:MAX_THREADS),ls,lc,l_in,l0 !`Is segs(:) threadsafe?
	real(real_kind) val_tr
	real(8) time_beg
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: real_kind
!DIR$ ATTRIBUTES ALIGN:128:: real_kind,im,ic,bases_in,bases_tr,segs
#endif

	ierr=0
!	time_beg=thread_wtime() !debug
	if(rank_in.gt.0.and.mod(rank_in,2).eq.0) then !even rank since index contractions are pairwise
!Set index links:
	 do i=1,rank_in
	  j=contr_ptrn(i)
	  if(j.lt.0) then !contracted index
	   if(-j.gt.rank_in) then; ierr=1; return; endif
	   if(contr_ptrn(-j).ne.-i) then; ierr=2; return; endif
	   if(dims_in(-j).ne.dims_in(i)) then; ierr=3; return; endif
	   if(-j.gt.i) then; ic(i)=-j; elseif(-j.lt.i) then; ic(i)=0; else; ierr=4; return; endif
	  else
	   ierr=5; return
	  endif
	 enddo
!Compute indexing bases:
	 ls=1_LONGINT; do i=1,rank_in; bases_in(i)=ls; ls=ls*dims_in(i); enddo !total size of the tensor block
	 lc=1_LONGINT
	 do i=1,rank_in
	  if(ic(i).gt.0) then; bases_tr(i)=lc; lc=lc*dims_in(i); else; bases_tr(i)=1_LONGINT; endif !size of the trace range
	 enddo
!Trace:
	 if(lc.gt.1_LONGINT) then !tracing over multiple elements
	  val_tr=0.
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,j,m,n,kf,l0,l_in,im) REDUCTION(+:val_tr)
#ifndef NO_OMP
	  n=omp_get_thread_num(); m=omp_get_num_threads()
#else
	  n=0; m=1
#endif
!$OMP MASTER
	  segs(0)=0_LONGINT; call divide_segment(lc,int(m,LONGINT),segs(1:),ierr); do i=2,m; segs(i)=segs(i)+segs(i-1); enddo
!$OMP END MASTER
!$OMP BARRIER
!$OMP FLUSH(segs)
	  l0=segs(n)
	  do i=rank_in,1,-1 !init multiindex for each thread
	   if(ic(i).gt.0) then; im(i)=l0/bases_tr(i); l0=mod(l0,bases_tr(i)); im(ic(i))=im(i); endif
	  enddo
	  kf=0; l_in=im(1); do i=2,rank_in; l_in=l_in+im(i)*bases_in(i); enddo !start offset for each thread
	  tloop: do l0=segs(n),segs(n+1)-1_LONGINT
	   val_tr=val_tr+tens_in(l_in)
	   do i=1,rank_in
	    j=ic(i)
	    if(j.gt.0) then
	     if(im(i)+1.lt.dims_in(i)) then
	      im(i)=im(i)+1; im(j)=im(j)+1; l_in=l_in+bases_in(i)+bases_in(j)
	      kf=kf+1; exit
	     else
	      l_in=l_in-im(i)*bases_in(i)-im(j)*bases_in(j); im(i)=0; im(j)=0
	     endif
	    endif
	   enddo
	   kf=kf-1; if(kf.lt.0) exit tloop
	  enddo tloop
!$OMP END PARALLEL
	  val_out=val_out+val_tr
	 elseif(lc.eq.1_LONGINT) then !tracing over only one element
	  if(ls.ne.1_LONGINT) then; ierr=6; return; endif
	  val_out=val_out+tens_in(0)
	 else
	  ierr=7 !negative or zero trace range
	 endif
	else
	 ierr=8 !negative or zero tensor rank
	endif
!	write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_ftrace_dlf_r4): kernel time/error code: ",F10.4,1x,i3)') &
!        thread_wtime(time_beg),ierr !debug
	return
	end subroutine tensor_block_ftrace_dlf_r4
!------------------------------------------------------------------------------------------------------
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: tensor_block_ftrace_dlf_r8
#endif
	subroutine tensor_block_ftrace_dlf_r8(contr_ptrn,ord_rest,tens_in,rank_in,dims_in,val_out,ierr) !PARALLEL
!This subroutine takes a full trace in a tensor block and accumulates it into a scalar.
!A full trace consists of one or more pairwise index contractions such that no single index is left uncontracted.
!Consequently, only even rank tensor blocks can be passed here.
!INPUT:
! - contr_ptrn(1:rank_in) - index contraction pattern;
! - ord_rest(1:rank_in) - index ordering restrictions (for contracted indices);
! - tens_in - input tensor block;
! - rank_in - rank of <tens_in>;
! - dims_in(1:rank_in) - dimension extents of <tens_in>;
! - val_out - initialized! scalar;
!OUTPUT:
! - val_out - modified scalar (the trace has been accumulated in);
! - ierr - error code (0:success).
!NOTES:
! - The algorithm used here is not cache-efficient, and I doubt there is any (D.I.L.).
! - No thorough argument checks.
!`Enable index ordering restrictions.
	implicit none
!---------------------------------------
	integer, parameter:: real_kind=8
!---------------------------------------
	integer, intent(in):: rank_in,contr_ptrn(1:rank_in),ord_rest(1:rank_in),dims_in(1:rank_in)
	real(real_kind), intent(in):: tens_in(0:*)
	real(real_kind), intent(inout):: val_out
	integer, intent(inout):: ierr
	integer i,j,k,l,m,n,ks,kf,im(1:rank_in),ic(1:rank_in)
	integer(LONGINT) bases_in(1:rank_in),bases_tr(1:rank_in),segs(0:MAX_THREADS),ls,lc,l_in,l0  !`Is segs(:) threadsafe?
	real(real_kind) val_tr
	real(8) time_beg
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: real_kind
!DIR$ ATTRIBUTES ALIGN:128:: real_kind,im,ic,bases_in,bases_tr,segs
#endif

	ierr=0
!	time_beg=thread_wtime() !debug
	if(rank_in.gt.0.and.mod(rank_in,2).eq.0) then !even rank since index contractions are pairwise
!Set index links:
	 do i=1,rank_in
	  j=contr_ptrn(i)
	  if(j.lt.0) then !contracted index
	   if(-j.gt.rank_in) then; ierr=1; return; endif
	   if(contr_ptrn(-j).ne.-i) then; ierr=2; return; endif
	   if(dims_in(-j).ne.dims_in(i)) then; ierr=3; return; endif
	   if(-j.gt.i) then; ic(i)=-j; elseif(-j.lt.i) then; ic(i)=0; else; ierr=4; return; endif
	  else
	   ierr=5; return
	  endif
	 enddo
!Compute indexing bases:
	 ls=1_LONGINT; do i=1,rank_in; bases_in(i)=ls; ls=ls*dims_in(i); enddo !total size of the tensor block
	 lc=1_LONGINT
	 do i=1,rank_in
	  if(ic(i).gt.0) then; bases_tr(i)=lc; lc=lc*dims_in(i); else; bases_tr(i)=1_LONGINT; endif !size of the trace range
	 enddo
!Trace:
	 if(lc.gt.1_LONGINT) then !tracing over multiple elements
	  val_tr=0d0
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,j,m,n,kf,l0,l_in,im) REDUCTION(+:val_tr)
#ifndef NO_OMP
	  n=omp_get_thread_num(); m=omp_get_num_threads()
#else
	  n=0; m=1
#endif
!$OMP MASTER
	  segs(0)=0_LONGINT; call divide_segment(lc,int(m,LONGINT),segs(1:),ierr); do i=2,m; segs(i)=segs(i)+segs(i-1); enddo
!$OMP END MASTER
!$OMP BARRIER
!$OMP FLUSH(segs)
	  l0=segs(n)
	  do i=rank_in,1,-1 !init multiindex for each thread
	   if(ic(i).gt.0) then; im(i)=l0/bases_tr(i); l0=mod(l0,bases_tr(i)); im(ic(i))=im(i); endif
	  enddo
	  kf=0; l_in=im(1); do i=2,rank_in; l_in=l_in+im(i)*bases_in(i); enddo !start offset for each thread
	  tloop: do l0=segs(n),segs(n+1)-1_LONGINT
	   val_tr=val_tr+tens_in(l_in)
	   do i=1,rank_in
	    j=ic(i)
	    if(j.gt.0) then
	     if(im(i)+1.lt.dims_in(i)) then
	      im(i)=im(i)+1; im(j)=im(j)+1; l_in=l_in+bases_in(i)+bases_in(j)
	      kf=kf+1; exit
	     else
	      l_in=l_in-im(i)*bases_in(i)-im(j)*bases_in(j); im(i)=0; im(j)=0
	     endif
	    endif
	   enddo
	   kf=kf-1; if(kf.lt.0) exit tloop
	  enddo tloop
!$OMP END PARALLEL
	  val_out=val_out+val_tr
	 elseif(lc.eq.1_LONGINT) then !tracing over only one element
	  if(ls.ne.1_LONGINT) then; ierr=6; return; endif
	  val_out=val_out+tens_in(0)
	 else
	  ierr=7 !negative or zero trace range
	 endif
	else
	 ierr=8 !negative or zero tensor rank
	endif
!	write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_ftrace_dlf_r8): kernel time/error code: ",F10.4,1x,i3)') &
!        thread_wtime(time_beg),ierr !debug
	return
	end subroutine tensor_block_ftrace_dlf_r8
!------------------------------------------------------------------------------------------------------
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: tensor_block_ftrace_dlf_c4
#endif
	subroutine tensor_block_ftrace_dlf_c4(contr_ptrn,ord_rest,tens_in,rank_in,dims_in,val_out,ierr) !PARALLEL
!This subroutine takes a full trace in a tensor block and accumulates it into a scalar.
!A full trace consists of one or more pairwise index contractions such that no single index is left uncontracted.
!Consequently, only even rank tensor blocks can be passed here.
!INPUT:
! - contr_ptrn(1:rank_in) - index contraction pattern;
! - ord_rest(1:rank_in) - index ordering restrictions (for contracted indices);
! - tens_in - input tensor block;
! - rank_in - rank of <tens_in>;
! - dims_in(1:rank_in) - dimension extents of <tens_in>;
! - val_out - initialized! scalar;
!OUTPUT:
! - val_out - modified scalar (the trace has been accumulated in);
! - ierr - error code (0:success).
!NOTES:
! - The algorithm used here is not cache-efficient, and I doubt there is any (D.I.L.).
! - No thorough argument checks.
!`Enable index ordering restrictions.
	implicit none
!---------------------------------------
	integer, parameter:: real_kind=4
!---------------------------------------
	integer, intent(in):: rank_in,contr_ptrn(1:rank_in),ord_rest(1:rank_in),dims_in(1:rank_in)
	complex(real_kind), intent(in):: tens_in(0:*)
	complex(real_kind), intent(inout):: val_out
	integer, intent(inout):: ierr
	integer i,j,k,l,m,n,ks,kf,im(1:rank_in),ic(1:rank_in)
	integer(LONGINT) bases_in(1:rank_in),bases_tr(1:rank_in),segs(0:MAX_THREADS),ls,lc,l_in,l0  !`Is segs(:) threadsafe?
	complex(real_kind) val_tr
	real(8) time_beg
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: real_kind
!DIR$ ATTRIBUTES ALIGN:128:: real_kind,im,ic,bases_in,bases_tr,segs
#endif

	ierr=0
!	time_beg=thread_wtime() !debug
	if(rank_in.gt.0.and.mod(rank_in,2).eq.0) then !even rank since index contractions are pairwise
!Set index links:
	 do i=1,rank_in
	  j=contr_ptrn(i)
	  if(j.lt.0) then !contracted index
	   if(-j.gt.rank_in) then; ierr=1; return; endif
	   if(contr_ptrn(-j).ne.-i) then; ierr=2; return; endif
	   if(dims_in(-j).ne.dims_in(i)) then; ierr=3; return; endif
	   if(-j.gt.i) then; ic(i)=-j; elseif(-j.lt.i) then; ic(i)=0; else; ierr=4; return; endif
	  else
	   ierr=5; return
	  endif
	 enddo
!Compute indexing bases:
	 ls=1_LONGINT; do i=1,rank_in; bases_in(i)=ls; ls=ls*dims_in(i); enddo !total size of the tensor block
	 lc=1_LONGINT
	 do i=1,rank_in
	  if(ic(i).gt.0) then; bases_tr(i)=lc; lc=lc*dims_in(i); else; bases_tr(i)=1_LONGINT; endif !size of the trace range
	 enddo
!Trace:
	 if(lc.gt.1_LONGINT) then !tracing over multiple elements
	  val_tr=(0.0,0.0)
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,j,m,n,kf,l0,l_in,im) REDUCTION(+:val_tr)
#ifndef NO_OMP
	  n=omp_get_thread_num(); m=omp_get_num_threads()
#else
	  n=0; m=1
#endif
!$OMP MASTER
	  segs(0)=0_LONGINT; call divide_segment(lc,int(m,LONGINT),segs(1:),ierr); do i=2,m; segs(i)=segs(i)+segs(i-1); enddo
!$OMP END MASTER
!$OMP BARRIER
!$OMP FLUSH(segs)
	  l0=segs(n)
	  do i=rank_in,1,-1 !init multiindex for each thread
	   if(ic(i).gt.0) then; im(i)=l0/bases_tr(i); l0=mod(l0,bases_tr(i)); im(ic(i))=im(i); endif
	  enddo
	  kf=0; l_in=im(1); do i=2,rank_in; l_in=l_in+im(i)*bases_in(i); enddo !start offset for each thread
	  tloop: do l0=segs(n),segs(n+1)-1_LONGINT
	   val_tr=val_tr+tens_in(l_in)
	   do i=1,rank_in
	    j=ic(i)
	    if(j.gt.0) then
	     if(im(i)+1.lt.dims_in(i)) then
	      im(i)=im(i)+1; im(j)=im(j)+1; l_in=l_in+bases_in(i)+bases_in(j)
	      kf=kf+1; exit
	     else
	      l_in=l_in-im(i)*bases_in(i)-im(j)*bases_in(j); im(i)=0; im(j)=0
	     endif
	    endif
	   enddo
	   kf=kf-1; if(kf.lt.0) exit tloop
	  enddo tloop
!$OMP END PARALLEL
	  val_out=val_out+val_tr
	 elseif(lc.eq.1_LONGINT) then !tracing over only one element
	  if(ls.ne.1_LONGINT) then; ierr=6; return; endif
	  val_out=val_out+tens_in(0)
	 else
	  ierr=7 !negative or zero trace range
	 endif
	else
	 ierr=8 !negative or zero tensor rank
	endif
!	write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_ftrace_dlf_c4): kernel time/error code: ",F10.4,1x,i3)') &
!        thread_wtime(time_beg),ierr !debug
	return
	end subroutine tensor_block_ftrace_dlf_c4
!------------------------------------------------------------------------------------------------------
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: tensor_block_ftrace_dlf_c8
#endif
	subroutine tensor_block_ftrace_dlf_c8(contr_ptrn,ord_rest,tens_in,rank_in,dims_in,val_out,ierr) !PARALLEL
!This subroutine takes a full trace in a tensor block and accumulates it into a scalar.
!A full trace consists of one or more pairwise index contractions such that no single index is left uncontracted.
!Consequently, only even rank tensor blocks can be passed here.
!INPUT:
! - contr_ptrn(1:rank_in) - index contraction pattern;
! - ord_rest(1:rank_in) - index ordering restrictions (for contracted indices);
! - tens_in - input tensor block;
! - rank_in - rank of <tens_in>;
! - dims_in(1:rank_in) - dimension extents of <tens_in>;
! - val_out - initialized! scalar;
!OUTPUT:
! - val_out - modified scalar (the trace has been accumulated in);
! - ierr - error code (0:success).
!NOTES:
! - The algorithm used here is not cache-efficient, and I doubt there is any (D.I.L.).
! - No thorough argument checks.
!`Enable index ordering restrictions.
	implicit none
!---------------------------------------
	integer, parameter:: real_kind=8
!---------------------------------------
	integer, intent(in):: rank_in,contr_ptrn(1:rank_in),ord_rest(1:rank_in),dims_in(1:rank_in)
	complex(real_kind), intent(in):: tens_in(0:*)
	complex(real_kind), intent(inout):: val_out
	integer, intent(inout):: ierr
	integer i,j,k,l,m,n,ks,kf,im(1:rank_in),ic(1:rank_in)
	integer(LONGINT) bases_in(1:rank_in),bases_tr(1:rank_in),segs(0:MAX_THREADS),ls,lc,l_in,l0  !`Is segs(:) threadsafe?
	complex(real_kind) val_tr
	real(8) time_beg
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: real_kind
!DIR$ ATTRIBUTES ALIGN:128:: real_kind,im,ic,bases_in,bases_tr,segs
#endif

	ierr=0
!	time_beg=thread_wtime() !debug
	if(rank_in.gt.0.and.mod(rank_in,2).eq.0) then !even rank since index contractions are pairwise
!Set index links:
	 do i=1,rank_in
	  j=contr_ptrn(i)
	  if(j.lt.0) then !contracted index
	   if(-j.gt.rank_in) then; ierr=1; return; endif
	   if(contr_ptrn(-j).ne.-i) then; ierr=2; return; endif
	   if(dims_in(-j).ne.dims_in(i)) then; ierr=3; return; endif
	   if(-j.gt.i) then; ic(i)=-j; elseif(-j.lt.i) then; ic(i)=0; else; ierr=4; return; endif
	  else
	   ierr=5; return
	  endif
	 enddo
!Compute indexing bases:
	 ls=1_LONGINT; do i=1,rank_in; bases_in(i)=ls; ls=ls*dims_in(i); enddo !total size of the tensor block
	 lc=1_LONGINT
	 do i=1,rank_in
	  if(ic(i).gt.0) then; bases_tr(i)=lc; lc=lc*dims_in(i); else; bases_tr(i)=1_LONGINT; endif !size of the trace range
	 enddo
!Trace:
	 if(lc.gt.1_LONGINT) then !tracing over multiple elements
	  val_tr=(0d0,0d0)
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,j,m,n,kf,l0,l_in,im) REDUCTION(+:val_tr)
#ifndef NO_OMP
	  n=omp_get_thread_num(); m=omp_get_num_threads()
#else
	  n=0; m=1
#endif
!$OMP MASTER
	  segs(0)=0_LONGINT; call divide_segment(lc,int(m,LONGINT),segs(1:),ierr); do i=2,m; segs(i)=segs(i)+segs(i-1); enddo
!$OMP END MASTER
!$OMP BARRIER
!$OMP FLUSH(segs)
	  l0=segs(n)
	  do i=rank_in,1,-1 !init multiindex for each thread
	   if(ic(i).gt.0) then; im(i)=l0/bases_tr(i); l0=mod(l0,bases_tr(i)); im(ic(i))=im(i); endif
	  enddo
	  kf=0; l_in=im(1); do i=2,rank_in; l_in=l_in+im(i)*bases_in(i); enddo !start offset for each thread
	  tloop: do l0=segs(n),segs(n+1)-1_LONGINT
	   val_tr=val_tr+tens_in(l_in)
	   do i=1,rank_in
	    j=ic(i)
	    if(j.gt.0) then
	     if(im(i)+1.lt.dims_in(i)) then
	      im(i)=im(i)+1; im(j)=im(j)+1; l_in=l_in+bases_in(i)+bases_in(j)
	      kf=kf+1; exit
	     else
	      l_in=l_in-im(i)*bases_in(i)-im(j)*bases_in(j); im(i)=0; im(j)=0
	     endif
	    endif
	   enddo
	   kf=kf-1; if(kf.lt.0) exit tloop
	  enddo tloop
!$OMP END PARALLEL
	  val_out=val_out+val_tr
	 elseif(lc.eq.1_LONGINT) then !tracing over only one element
	  if(ls.ne.1_LONGINT) then; ierr=6; return; endif
	  val_out=val_out+tens_in(0)
	 else
	  ierr=7 !negative or zero trace range
	 endif
	else
	 ierr=8 !negative or zero tensor rank
	endif
!	write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_ftrace_dlf_c8): kernel time/error code: ",F10.4,1x,i3)') &
!        thread_wtime(time_beg),ierr !debug
	return
	end subroutine tensor_block_ftrace_dlf_c8
!-------------------------------------------------------------------------------------------------------------------------
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: tensor_block_ptrace_dlf_r4
#endif
	subroutine tensor_block_ptrace_dlf_r4(contr_ptrn,ord_rest,tens_in,rank_in,dims_in,tens_out,rank_out,dims_out,ierr) !PARALLEL
!This subroutine takes a partial trace in a tensor block and accumulates it into the destination tensor block.
!A partial trace consists of one or more pairwise index contractions such that at least one index is left uncontracted.
!INPUT:
! - contr_ptrn(1:rank_in) - index contraction pattern;
! - ord_rest(1:rank_in) - index ordering restrictions (for contracted indices);
! - tens_in - input tensor block;
! - rank_in - rank of <tens_in>;
! - dims_in(1:rank_in) - dimension extents of <tens_in>;
! - tens_out - initialized! output tensor block;
! - rank_out - rank of <tens_out>;
! - dims_out(1:rank_out) - dimension extents of <tens_out>;
!OUTPUT:
! - tens_out - modified output tensor block;
! - ierr - error code (0:success).
!NOTES:
! - The algorithm used here is not cache-efficient, and I doubt there is any (D.I.L.).
! - No thorough argument checks.
!`Enable index ordering restrictions.
	implicit none
!---------------------------------------
	integer, parameter:: real_kind=4
!---------------------------------------
	integer, intent(in):: rank_in,rank_out,contr_ptrn(1:rank_in),ord_rest(1:rank_in),dims_in(1:rank_in),dims_out(1:rank_out)
	real(real_kind), intent(in):: tens_in(0:*)
	real(real_kind), intent(inout):: tens_out(0:*)
	integer, intent(inout):: ierr
	integer i,j,k,l,m,n,ks,kf,im(1:rank_in),ic(1:rank_in),ip(1:rank_out)
	integer(LONGINT) bases_in(1:rank_in),bases_out(1:rank_out),bases_tr(1:rank_in),segs(0:MAX_THREADS)  !`Is segs(:) threadsafe?
	integer(LONGINT) li,lo,lc,l_in,l_out,l0
	real(real_kind) val_tr
	real(8) time_beg
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: real_kind
!DIR$ ATTRIBUTES ALIGN:128:: real_kind,im,ic,ip,bases_in,bases_out,bases_tr,segs
#endif

	ierr=0
!	time_beg=thread_wtime() !debug
	if(rank_out.gt.0.and.rank_out.lt.rank_in.and.mod(rank_in-rank_out,2).eq.0) then !even rank difference because of pairwise index contractions
!Set index links:
	 ip(1:rank_out)=0
	 do i=1,rank_in
	  j=contr_ptrn(i)
	  if(j.lt.0) then !contracted index
	   if(-j.gt.rank_in) then; ierr=1; return; endif
	   if(contr_ptrn(-j).ne.-i) then; ierr=2; return; endif
	   if(dims_in(-j).ne.dims_in(i)) then; ierr=3; return; endif
	   if(-j.gt.i) then; ic(i)=-j; elseif(-j.lt.i) then; ic(i)=0; else; ierr=4; return; endif
	  elseif(j.gt.0) then !uncontracted index
	   if(j.gt.rank_out) then; ierr=5; return; endif
	   if(dims_out(j).ne.dims_in(i)) then; ierr=6; return; endif
	   if(ip(j).eq.0) then; ip(j)=ip(j)+1; else; ierr=7; return; endif
	   ic(i)=-j
	  else
	   ierr=8; return
	  endif
	 enddo
	 do i=1,rank_out; if(ip(i).ne.1) then; ierr=9; return; endif; enddo
	 do i=1,rank_in; if(ic(i).lt.0) then; ip(-ic(i))=i; endif; enddo
!Compute indexing bases:
	 li=1_LONGINT; lc=1_LONGINT
	 do i=1,rank_in
	  bases_in(i)=li; li=li*dims_in(i) !input indexing bases
	  if(ic(i).gt.0) then; bases_tr(i)=lc; lc=lc*dims_in(i); else; bases_tr(i)=1_LONGINT; endif !trace range bases
	 enddo
	 lo=1_LONGINT; do i=1,rank_out; bases_out(i)=lo; lo=lo*dims_out(i); enddo !output indexing bases
!Trace:
	 if(lo.ge.1_LONGINT.and.li.gt.1_LONGINT) then
	  if(lo.gt.lc) then !Scheme 1
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,j,m,n,kf,l0,l_in,l_out,im,val_tr)
#ifndef NO_OMP
	   n=omp_get_thread_num(); m=omp_get_num_threads()
#else
	   n=0; m=1
#endif
!$OMP MASTER
	   segs(0)=0_LONGINT; call divide_segment(lo,int(m,LONGINT),segs(1:),ierr); do i=2,m; segs(i)=segs(i)+segs(i-1); enddo
!$OMP END MASTER
!$OMP BARRIER
!$OMP FLUSH(segs)
	   do l_out=segs(n),segs(n+1)-1_LONGINT
	    im(1:rank_in)=0; l0=l_out; do i=rank_out,1,-1; im(ip(i))=l0/bases_out(i); l0=mod(l0,bases_out(i)); enddo
	    l_in=im(1); do i=2,rank_in; l_in=l_in+im(i)*bases_in(i); enddo
	    val_tr=0.; kf=0
	    cloop: do l0=0_LONGINT,lc-1_LONGINT
	     val_tr=val_tr+tens_in(l_in)
	     do i=1,rank_in
	      j=ic(i)
	      if(j.gt.0) then
	       if(im(i)+1.lt.dims_in(i)) then
	        im(i)=im(i)+1; im(j)=im(j)+1; l_in=l_in+bases_in(i)+bases_in(j)
	        kf=kf+1; exit
	       else
	        l_in=l_in-im(i)*bases_in(i)-im(j)*bases_in(j); im(i)=0; im(j)=0
	       endif
	      endif
	     enddo
	     kf=kf-1; if(kf.lt.0) exit cloop
	    enddo cloop
	    tens_out(l_out)=tens_out(l_out)+val_tr
	   enddo
!$OMP END PARALLEL
	  else !Scheme 2
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,j,m,n,kf,l0,l_in,l_out,im,val_tr)
#ifndef NO_OMP
	   n=omp_get_thread_num(); m=omp_get_num_threads()
#else
	   n=0; m=1
#endif
!$OMP MASTER
	   segs(0)=0_LONGINT; call divide_segment(lc,int(m,LONGINT),segs(1:),ierr); do i=2,m; segs(i)=segs(i)+segs(i-1); enddo
!$OMP END MASTER
!$OMP BARRIER
!$OMP FLUSH(segs)
	   do l_out=0_LONGINT,lo-1_LONGINT
	    l0=l_out; do i=rank_out,1,-1; im(ip(i))=l0/bases_out(i); l0=mod(l0,bases_out(i)); enddo
	    l0=segs(n)
	    do i=rank_in,1,-1
	     if(ic(i).gt.0) then; im(i)=l0/bases_tr(i); l0=mod(l0,bases_tr(i)); im(ic(i))=im(i); endif
	    enddo
	    l_in=im(1); do i=2,rank_in; l_in=l_in+im(i)*bases_in(i); enddo
	    val_tr=0.; kf=0
	    tloop: do l0=segs(n),segs(n+1)-1_LONGINT
	     val_tr=val_tr+tens_in(l_in)
	     do i=1,rank_in
	      j=ic(i)
	      if(j.gt.0) then
	       if(im(i)+1.lt.dims_in(i)) then
	        im(i)=im(i)+1; im(j)=im(j)+1; l_in=l_in+bases_in(i)+bases_in(j)
	        kf=kf+1; exit
	       else
	        l_in=l_in-im(i)*bases_in(i)-im(j)*bases_in(j); im(i)=0; im(j)=0
	       endif
	      endif
	     enddo
	     kf=kf-1; if(kf.lt.0) exit tloop
	    enddo tloop
!$OMP ATOMIC UPDATE
	    tens_out(l_out)=tens_out(l_out)+val_tr
	   enddo
!$OMP END PARALLEL
	  endif
	 elseif(lo.eq.1_LONGINT.and.li.eq.1_LONGINT) then
	  tens_out(0)=tens_out(0)+tens_in(0)
	 else
	  ierr=10
	 endif
	else
	 ierr=11
	endif
!	write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_ptrace_dlf_r4): kernel time/error code: ",F10.4,1x,i3)') &
!        thread_wtime(time_beg),ierr !debug
	return
	end subroutine tensor_block_ptrace_dlf_r4
!-------------------------------------------------------------------------------------------------------------------------
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: tensor_block_ptrace_dlf_r8
#endif
	subroutine tensor_block_ptrace_dlf_r8(contr_ptrn,ord_rest,tens_in,rank_in,dims_in,tens_out,rank_out,dims_out,ierr) !PARALLEL
!This subroutine takes a partial trace in a tensor block and accumulates it into the destination tensor block.
!A partial trace consists of one or more pairwise index contractions such that at least one index is left uncontracted.
!INPUT:
! - contr_ptrn(1:rank_in) - index contraction pattern;
! - ord_rest(1:rank_in) - index ordering restrictions (for contracted indices);
! - tens_in - input tensor block;
! - rank_in - rank of <tens_in>;
! - dims_in(1:rank_in) - dimension extents of <tens_in>;
! - tens_out - initialized! output tensor block;
! - rank_out - rank of <tens_out>;
! - dims_out(1:rank_out) - dimension extents of <tens_out>;
!OUTPUT:
! - tens_out - modified output tensor block;
! - ierr - error code (0:success).
!NOTES:
! - The algorithm used here is not cache-efficient, and I doubt there is any (D.I.L.).
! - No thorough argument checks.
!`Enable index ordering restrictions.
	implicit none
!---------------------------------------
	integer, parameter:: real_kind=8
!---------------------------------------
	integer, intent(in):: rank_in,rank_out,contr_ptrn(1:rank_in),ord_rest(1:rank_in),dims_in(1:rank_in),dims_out(1:rank_out)
	real(real_kind), intent(in):: tens_in(0:*)
	real(real_kind), intent(inout):: tens_out(0:*)
	integer, intent(inout):: ierr
	integer i,j,k,l,m,n,ks,kf,im(1:rank_in),ic(1:rank_in),ip(1:rank_out)
	integer(LONGINT) bases_in(1:rank_in),bases_out(1:rank_out),bases_tr(1:rank_in),segs(0:MAX_THREADS)  !`Is segs(:) threadsafe?
	integer(LONGINT) li,lo,lc,l_in,l_out,l0
	real(real_kind) val_tr
	real(8) time_beg
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: real_kind
!DIR$ ATTRIBUTES ALIGN:128:: real_kind,im,ic,ip,bases_in,bases_out,bases_tr,segs
#endif

	ierr=0
!	time_beg=thread_wtime() !debug
	if(rank_out.gt.0.and.rank_out.lt.rank_in.and.mod(rank_in-rank_out,2).eq.0) then !even rank difference because of pairwise index contractions
!Set index links:
	 ip(1:rank_out)=0
	 do i=1,rank_in
	  j=contr_ptrn(i)
	  if(j.lt.0) then !contracted index
	   if(-j.gt.rank_in) then; ierr=1; return; endif
	   if(contr_ptrn(-j).ne.-i) then; ierr=2; return; endif
	   if(dims_in(-j).ne.dims_in(i)) then; ierr=3; return; endif
	   if(-j.gt.i) then; ic(i)=-j; elseif(-j.lt.i) then; ic(i)=0; else; ierr=4; return; endif
	  elseif(j.gt.0) then !uncontracted index
	   if(j.gt.rank_out) then; ierr=5; return; endif
	   if(dims_out(j).ne.dims_in(i)) then; ierr=6; return; endif
	   if(ip(j).eq.0) then; ip(j)=ip(j)+1; else; ierr=7; return; endif
	   ic(i)=-j
	  else
	   ierr=8; return
	  endif
	 enddo
	 do i=1,rank_out; if(ip(i).ne.1) then; ierr=9; return; endif; enddo
	 do i=1,rank_in; if(ic(i).lt.0) then; ip(-ic(i))=i; endif; enddo
!Compute indexing bases:
	 li=1_LONGINT; lc=1_LONGINT
	 do i=1,rank_in
	  bases_in(i)=li; li=li*dims_in(i) !input indexing bases
	  if(ic(i).gt.0) then; bases_tr(i)=lc; lc=lc*dims_in(i); else; bases_tr(i)=1_LONGINT; endif !trace range bases
	 enddo
	 lo=1_LONGINT; do i=1,rank_out; bases_out(i)=lo; lo=lo*dims_out(i); enddo !output indexing bases
!Trace:
	 if(lo.ge.1_LONGINT.and.li.gt.1_LONGINT) then
	  if(lo.gt.lc) then !Scheme 1
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,j,m,n,kf,l0,l_in,l_out,im,val_tr)
#ifndef NO_OMP
	   n=omp_get_thread_num(); m=omp_get_num_threads()
#else
	   n=0; m=1
#endif
!$OMP MASTER
	   segs(0)=0_LONGINT; call divide_segment(lo,int(m,LONGINT),segs(1:),ierr); do i=2,m; segs(i)=segs(i)+segs(i-1); enddo
!$OMP END MASTER
!$OMP BARRIER
!$OMP FLUSH(segs)
	   do l_out=segs(n),segs(n+1)-1_LONGINT
	    im(1:rank_in)=0; l0=l_out; do i=rank_out,1,-1; im(ip(i))=l0/bases_out(i); l0=mod(l0,bases_out(i)); enddo
	    l_in=im(1); do i=2,rank_in; l_in=l_in+im(i)*bases_in(i); enddo
	    val_tr=0d0; kf=0
	    cloop: do l0=0_LONGINT,lc-1_LONGINT
	     val_tr=val_tr+tens_in(l_in)
	     do i=1,rank_in
	      j=ic(i)
	      if(j.gt.0) then
	       if(im(i)+1.lt.dims_in(i)) then
	        im(i)=im(i)+1; im(j)=im(j)+1; l_in=l_in+bases_in(i)+bases_in(j)
	        kf=kf+1; exit
	       else
	        l_in=l_in-im(i)*bases_in(i)-im(j)*bases_in(j); im(i)=0; im(j)=0
	       endif
	      endif
	     enddo
	     kf=kf-1; if(kf.lt.0) exit cloop
	    enddo cloop
	    tens_out(l_out)=tens_out(l_out)+val_tr
	   enddo
!$OMP END PARALLEL
	  else !Scheme 2
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,j,m,n,kf,l0,l_in,l_out,im,val_tr)
#ifndef NO_OMP
	   n=omp_get_thread_num(); m=omp_get_num_threads()
#else
	   n=0; m=1
#endif
!$OMP MASTER
	   segs(0)=0_LONGINT; call divide_segment(lc,int(m,LONGINT),segs(1:),ierr); do i=2,m; segs(i)=segs(i)+segs(i-1); enddo
!$OMP END MASTER
!$OMP BARRIER
!$OMP FLUSH(segs)
	   do l_out=0_LONGINT,lo-1_LONGINT
	    l0=l_out; do i=rank_out,1,-1; im(ip(i))=l0/bases_out(i); l0=mod(l0,bases_out(i)); enddo
	    l0=segs(n)
	    do i=rank_in,1,-1
	     if(ic(i).gt.0) then; im(i)=l0/bases_tr(i); l0=mod(l0,bases_tr(i)); im(ic(i))=im(i); endif
	    enddo
	    l_in=im(1); do i=2,rank_in; l_in=l_in+im(i)*bases_in(i); enddo
	    val_tr=0d0; kf=0
	    tloop: do l0=segs(n),segs(n+1)-1_LONGINT
	     val_tr=val_tr+tens_in(l_in)
	     do i=1,rank_in
	      j=ic(i)
	      if(j.gt.0) then
	       if(im(i)+1.lt.dims_in(i)) then
	        im(i)=im(i)+1; im(j)=im(j)+1; l_in=l_in+bases_in(i)+bases_in(j)
	        kf=kf+1; exit
	       else
	        l_in=l_in-im(i)*bases_in(i)-im(j)*bases_in(j); im(i)=0; im(j)=0
	       endif
	      endif
	     enddo
	     kf=kf-1; if(kf.lt.0) exit tloop
	    enddo tloop
!$OMP ATOMIC UPDATE
	    tens_out(l_out)=tens_out(l_out)+val_tr
	   enddo
!$OMP END PARALLEL
	  endif
	 elseif(lo.eq.1_LONGINT.and.li.eq.1_LONGINT) then
	  tens_out(0)=tens_out(0)+tens_in(0)
	 else
	  ierr=10
	 endif
	else
	 ierr=11
	endif
!	write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_ptrace_dlf_r8): kernel time/error code: ",F10.4,1x,i3)') &
!        thread_wtime(time_beg),ierr !debug
	return
	end subroutine tensor_block_ptrace_dlf_r8
!-------------------------------------------------------------------------------------------------------------------------
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: tensor_block_ptrace_dlf_c4
#endif
	subroutine tensor_block_ptrace_dlf_c4(contr_ptrn,ord_rest,tens_in,rank_in,dims_in,tens_out,rank_out,dims_out,ierr) !PARALLEL
!This subroutine takes a partial trace in a tensor block and accumulates it into the destination tensor block.
!A partial trace consists of one or more pairwise index contractions such that at least one index is left uncontracted.
!INPUT:
! - contr_ptrn(1:rank_in) - index contraction pattern;
! - ord_rest(1:rank_in) - index ordering restrictions (for contracted indices);
! - tens_in - input tensor block;
! - rank_in - rank of <tens_in>;
! - dims_in(1:rank_in) - dimension extents of <tens_in>;
! - tens_out - initialized! output tensor block;
! - rank_out - rank of <tens_out>;
! - dims_out(1:rank_out) - dimension extents of <tens_out>;
!OUTPUT:
! - tens_out - modified output tensor block;
! - ierr - error code (0:success).
!NOTES:
! - The algorithm used here is not cache-efficient, and I doubt there is any (D.I.L.).
! - No thorough argument checks.
!`Enable index ordering restrictions.
	implicit none
!---------------------------------------
	integer, parameter:: real_kind=4
!---------------------------------------
	integer, intent(in):: rank_in,rank_out,contr_ptrn(1:rank_in),ord_rest(1:rank_in),dims_in(1:rank_in),dims_out(1:rank_out)
	complex(real_kind), intent(in):: tens_in(0:*)
	complex(real_kind), intent(inout):: tens_out(0:*)
	integer, intent(inout):: ierr
	integer i,j,k,l,m,n,ks,kf,im(1:rank_in),ic(1:rank_in),ip(1:rank_out)
	integer(LONGINT) bases_in(1:rank_in),bases_out(1:rank_out),bases_tr(1:rank_in),segs(0:MAX_THREADS)  !`Is segs(:) threadsafe?
	integer(LONGINT) li,lo,lc,l_in,l_out,l0
        complex(real_kind) val_tr
	real(8) time_beg
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: real_kind
!DIR$ ATTRIBUTES ALIGN:128:: real_kind,im,ic,ip,bases_in,bases_out,bases_tr,segs
#endif

	ierr=0
!	time_beg=thread_wtime() !debug
	if(rank_out.gt.0.and.rank_out.lt.rank_in.and.mod(rank_in-rank_out,2).eq.0) then !even rank difference because of pairwise index contractions
!Set index links:
	 ip(1:rank_out)=0
	 do i=1,rank_in
	  j=contr_ptrn(i)
	  if(j.lt.0) then !contracted index
	   if(-j.gt.rank_in) then; ierr=1; return; endif
	   if(contr_ptrn(-j).ne.-i) then; ierr=2; return; endif
	   if(dims_in(-j).ne.dims_in(i)) then; ierr=3; return; endif
	   if(-j.gt.i) then; ic(i)=-j; elseif(-j.lt.i) then; ic(i)=0; else; ierr=4; return; endif
	  elseif(j.gt.0) then !uncontracted index
	   if(j.gt.rank_out) then; ierr=5; return; endif
	   if(dims_out(j).ne.dims_in(i)) then; ierr=6; return; endif
	   if(ip(j).eq.0) then; ip(j)=ip(j)+1; else; ierr=7; return; endif
	   ic(i)=-j
	  else
	   ierr=8; return
	  endif
	 enddo
	 do i=1,rank_out; if(ip(i).ne.1) then; ierr=9; return; endif; enddo
	 do i=1,rank_in; if(ic(i).lt.0) then; ip(-ic(i))=i; endif; enddo
!Compute indexing bases:
	 li=1_LONGINT; lc=1_LONGINT
	 do i=1,rank_in
	  bases_in(i)=li; li=li*dims_in(i) !input indexing bases
	  if(ic(i).gt.0) then; bases_tr(i)=lc; lc=lc*dims_in(i); else; bases_tr(i)=1_LONGINT; endif !trace range bases
	 enddo
	 lo=1_LONGINT; do i=1,rank_out; bases_out(i)=lo; lo=lo*dims_out(i); enddo !output indexing bases
!Trace:
	 if(lo.ge.1_LONGINT.and.li.gt.1_LONGINT) then
	  if(lo.gt.lc) then !Scheme 1
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,j,m,n,kf,l0,l_in,l_out,im,val_tr)
#ifndef NO_OMP
	   n=omp_get_thread_num(); m=omp_get_num_threads()
#else
	   n=0; m=1
#endif
!$OMP MASTER
	   segs(0)=0_LONGINT; call divide_segment(lo,int(m,LONGINT),segs(1:),ierr); do i=2,m; segs(i)=segs(i)+segs(i-1); enddo
!$OMP END MASTER
!$OMP BARRIER
!$OMP FLUSH(segs)
	   do l_out=segs(n),segs(n+1)-1_LONGINT
	    im(1:rank_in)=0; l0=l_out; do i=rank_out,1,-1; im(ip(i))=l0/bases_out(i); l0=mod(l0,bases_out(i)); enddo
	    l_in=im(1); do i=2,rank_in; l_in=l_in+im(i)*bases_in(i); enddo
	    val_tr=(0.0,0.0); kf=0
	    cloop: do l0=0_LONGINT,lc-1_LONGINT
	     val_tr=val_tr+tens_in(l_in)
	     do i=1,rank_in
	      j=ic(i)
	      if(j.gt.0) then
	       if(im(i)+1.lt.dims_in(i)) then
	        im(i)=im(i)+1; im(j)=im(j)+1; l_in=l_in+bases_in(i)+bases_in(j)
	        kf=kf+1; exit
	       else
	        l_in=l_in-im(i)*bases_in(i)-im(j)*bases_in(j); im(i)=0; im(j)=0
	       endif
	      endif
	     enddo
	     kf=kf-1; if(kf.lt.0) exit cloop
	    enddo cloop
	    tens_out(l_out)=tens_out(l_out)+val_tr
	   enddo
!$OMP END PARALLEL
	  else !Scheme 2
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,j,m,n,kf,l0,l_in,l_out,im,val_tr)
#ifndef NO_OMP
	   n=omp_get_thread_num(); m=omp_get_num_threads()
#else
	   n=0; m=1
#endif
!$OMP MASTER
	   segs(0)=0_LONGINT; call divide_segment(lc,int(m,LONGINT),segs(1:),ierr); do i=2,m; segs(i)=segs(i)+segs(i-1); enddo
!$OMP END MASTER
!$OMP BARRIER
!$OMP FLUSH(segs)
	   do l_out=0_LONGINT,lo-1_LONGINT
	    l0=l_out; do i=rank_out,1,-1; im(ip(i))=l0/bases_out(i); l0=mod(l0,bases_out(i)); enddo
	    l0=segs(n)
	    do i=rank_in,1,-1
	     if(ic(i).gt.0) then; im(i)=l0/bases_tr(i); l0=mod(l0,bases_tr(i)); im(ic(i))=im(i); endif
	    enddo
	    l_in=im(1); do i=2,rank_in; l_in=l_in+im(i)*bases_in(i); enddo
	    val_tr=(0.0,0.0); kf=0
	    tloop: do l0=segs(n),segs(n+1)-1_LONGINT
	     val_tr=val_tr+tens_in(l_in)
	     do i=1,rank_in
	      j=ic(i)
	      if(j.gt.0) then
	       if(im(i)+1.lt.dims_in(i)) then
	        im(i)=im(i)+1; im(j)=im(j)+1; l_in=l_in+bases_in(i)+bases_in(j)
	        kf=kf+1; exit
	       else
	        l_in=l_in-im(i)*bases_in(i)-im(j)*bases_in(j); im(i)=0; im(j)=0
	       endif
	      endif
	     enddo
	     kf=kf-1; if(kf.lt.0) exit tloop
	    enddo tloop
!$OMP ATOMIC UPDATE
	    tens_out(l_out)=tens_out(l_out)+val_tr
	   enddo
!$OMP END PARALLEL
	  endif
	 elseif(lo.eq.1_LONGINT.and.li.eq.1_LONGINT) then
	  tens_out(0)=tens_out(0)+tens_in(0)
	 else
	  ierr=10
	 endif
	else
	 ierr=11
	endif
!	write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_ptrace_dlf_c4): kernel time/error code: ",F10.4,1x,i3)') &
!        thread_wtime(time_beg),ierr !debug
	return
	end subroutine tensor_block_ptrace_dlf_c4
!-------------------------------------------------------------------------------------------------------------------------
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: tensor_block_ptrace_dlf_c8
#endif
	subroutine tensor_block_ptrace_dlf_c8(contr_ptrn,ord_rest,tens_in,rank_in,dims_in,tens_out,rank_out,dims_out,ierr) !PARALLEL
!This subroutine takes a partial trace in a tensor block and accumulates it into the destination tensor block.
!A partial trace consists of one or more pairwise index contractions such that at least one index is left uncontracted.
!INPUT:
! - contr_ptrn(1:rank_in) - index contraction pattern;
! - ord_rest(1:rank_in) - index ordering restrictions (for contracted indices);
! - tens_in - input tensor block;
! - rank_in - rank of <tens_in>;
! - dims_in(1:rank_in) - dimension extents of <tens_in>;
! - tens_out - initialized! output tensor block;
! - rank_out - rank of <tens_out>;
! - dims_out(1:rank_out) - dimension extents of <tens_out>;
!OUTPUT:
! - tens_out - modified output tensor block;
! - ierr - error code (0:success).
!NOTES:
! - The algorithm used here is not cache-efficient, and I doubt there is any (D.I.L.).
! - No thorough argument checks.
!`Enable index ordering restrictions.
	implicit none
!---------------------------------------
	integer, parameter:: real_kind=8
!---------------------------------------
	integer, intent(in):: rank_in,rank_out,contr_ptrn(1:rank_in),ord_rest(1:rank_in),dims_in(1:rank_in),dims_out(1:rank_out)
	complex(real_kind), intent(in):: tens_in(0:*)
	complex(real_kind), intent(inout):: tens_out(0:*)
	integer, intent(inout):: ierr
	integer i,j,k,l,m,n,ks,kf,im(1:rank_in),ic(1:rank_in),ip(1:rank_out)
	integer(LONGINT) bases_in(1:rank_in),bases_out(1:rank_out),bases_tr(1:rank_in),segs(0:MAX_THREADS)  !`Is segs(:) threadsafe?
	integer(LONGINT) li,lo,lc,l_in,l_out,l0
        complex(real_kind) val_tr
	real(8) time_beg
#ifndef NO_PHI
!DIR$ ATTRIBUTES OFFLOAD:mic:: real_kind
!DIR$ ATTRIBUTES ALIGN:128:: real_kind,im,ic,ip,bases_in,bases_out,bases_tr,segs
#endif

	ierr=0
!	time_beg=thread_wtime() !debug
	if(rank_out.gt.0.and.rank_out.lt.rank_in.and.mod(rank_in-rank_out,2).eq.0) then !even rank difference because of pairwise index contractions
!Set index links:
	 ip(1:rank_out)=0
	 do i=1,rank_in
	  j=contr_ptrn(i)
	  if(j.lt.0) then !contracted index
	   if(-j.gt.rank_in) then; ierr=1; return; endif
	   if(contr_ptrn(-j).ne.-i) then; ierr=2; return; endif
	   if(dims_in(-j).ne.dims_in(i)) then; ierr=3; return; endif
	   if(-j.gt.i) then; ic(i)=-j; elseif(-j.lt.i) then; ic(i)=0; else; ierr=4; return; endif
	  elseif(j.gt.0) then !uncontracted index
	   if(j.gt.rank_out) then; ierr=5; return; endif
	   if(dims_out(j).ne.dims_in(i)) then; ierr=6; return; endif
	   if(ip(j).eq.0) then; ip(j)=ip(j)+1; else; ierr=7; return; endif
	   ic(i)=-j
	  else
	   ierr=8; return
	  endif
	 enddo
	 do i=1,rank_out; if(ip(i).ne.1) then; ierr=9; return; endif; enddo
	 do i=1,rank_in; if(ic(i).lt.0) then; ip(-ic(i))=i; endif; enddo
!Compute indexing bases:
	 li=1_LONGINT; lc=1_LONGINT
	 do i=1,rank_in
	  bases_in(i)=li; li=li*dims_in(i) !input indexing bases
	  if(ic(i).gt.0) then; bases_tr(i)=lc; lc=lc*dims_in(i); else; bases_tr(i)=1_LONGINT; endif !trace range bases
	 enddo
	 lo=1_LONGINT; do i=1,rank_out; bases_out(i)=lo; lo=lo*dims_out(i); enddo !output indexing bases
!Trace:
	 if(lo.ge.1_LONGINT.and.li.gt.1_LONGINT) then
	  if(lo.gt.lc) then !Scheme 1
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,j,m,n,kf,l0,l_in,l_out,im,val_tr)
#ifndef NO_OMP
	   n=omp_get_thread_num(); m=omp_get_num_threads()
#else
	   n=0; m=1
#endif
!$OMP MASTER
	   segs(0)=0_LONGINT; call divide_segment(lo,int(m,LONGINT),segs(1:),ierr); do i=2,m; segs(i)=segs(i)+segs(i-1); enddo
!$OMP END MASTER
!$OMP BARRIER
!$OMP FLUSH(segs)
	   do l_out=segs(n),segs(n+1)-1_LONGINT
	    im(1:rank_in)=0; l0=l_out; do i=rank_out,1,-1; im(ip(i))=l0/bases_out(i); l0=mod(l0,bases_out(i)); enddo
	    l_in=im(1); do i=2,rank_in; l_in=l_in+im(i)*bases_in(i); enddo
	    val_tr=(0d0,0d0); kf=0
	    cloop: do l0=0_LONGINT,lc-1_LONGINT
	     val_tr=val_tr+tens_in(l_in)
	     do i=1,rank_in
	      j=ic(i)
	      if(j.gt.0) then
	       if(im(i)+1.lt.dims_in(i)) then
	        im(i)=im(i)+1; im(j)=im(j)+1; l_in=l_in+bases_in(i)+bases_in(j)
	        kf=kf+1; exit
	       else
	        l_in=l_in-im(i)*bases_in(i)-im(j)*bases_in(j); im(i)=0; im(j)=0
	       endif
	      endif
	     enddo
	     kf=kf-1; if(kf.lt.0) exit cloop
	    enddo cloop
	    tens_out(l_out)=tens_out(l_out)+val_tr
	   enddo
!$OMP END PARALLEL
	  else !Scheme 2
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,j,m,n,kf,l0,l_in,l_out,im,val_tr)
#ifndef NO_OMP
	   n=omp_get_thread_num(); m=omp_get_num_threads()
#else
	   n=0; m=1
#endif
!$OMP MASTER
	   segs(0)=0_LONGINT; call divide_segment(lc,int(m,LONGINT),segs(1:),ierr); do i=2,m; segs(i)=segs(i)+segs(i-1); enddo
!$OMP END MASTER
!$OMP BARRIER
!$OMP FLUSH(segs)
	   do l_out=0_LONGINT,lo-1_LONGINT
	    l0=l_out; do i=rank_out,1,-1; im(ip(i))=l0/bases_out(i); l0=mod(l0,bases_out(i)); enddo
	    l0=segs(n)
	    do i=rank_in,1,-1
	     if(ic(i).gt.0) then; im(i)=l0/bases_tr(i); l0=mod(l0,bases_tr(i)); im(ic(i))=im(i); endif
	    enddo
	    l_in=im(1); do i=2,rank_in; l_in=l_in+im(i)*bases_in(i); enddo
	    val_tr=(0d0,0d0); kf=0
	    tloop: do l0=segs(n),segs(n+1)-1_LONGINT
	     val_tr=val_tr+tens_in(l_in)
	     do i=1,rank_in
	      j=ic(i)
	      if(j.gt.0) then
	       if(im(i)+1.lt.dims_in(i)) then
	        im(i)=im(i)+1; im(j)=im(j)+1; l_in=l_in+bases_in(i)+bases_in(j)
	        kf=kf+1; exit
	       else
	        l_in=l_in-im(i)*bases_in(i)-im(j)*bases_in(j); im(i)=0; im(j)=0
	       endif
	      endif
	     enddo
	     kf=kf-1; if(kf.lt.0) exit tloop
	    enddo tloop
!$OMP ATOMIC UPDATE
	    tens_out(l_out)=tens_out(l_out)+val_tr
	   enddo
!$OMP END PARALLEL
	  endif
	 elseif(lo.eq.1_LONGINT.and.li.eq.1_LONGINT) then
	  tens_out(0)=tens_out(0)+tens_in(0)
	 else
	  ierr=10
	 endif
	else
	 ierr=11
	endif
!	write(CONS_OUT,'("DEBUG(tensor_algebra::tensor_block_ptrace_dlf_c8): kernel time/error code: ",F10.4,1x,i3)') &
!        thread_wtime(time_beg),ierr !debug
	return
	end subroutine tensor_block_ptrace_dlf_c8

       end module tensor_algebra_cpu
