       module combinatoric
!Combinatoric Procedures.
!AUTHOR: Dmitry I. Lyakh (Liakh): quant4me@gmail.com
!Revision: 2016/01/08

!PROCEDURES:
! - TRNG(i:ctrl,i:ni,i[1]:trn,i[1]:ngt): permutation generator which returns each new permutation.
! - TRSIGN(i:n,i[1]:itr): reorders indices in an ascending order and determines the sign of the corresponding permutation (Bubble). Use MERGE_SORT for fast.
! - i8:FACTORIAL(i:n): factorial of a number.
! - i:NOID(i:m,i:n): this function returns a binomial coefficient (integer).
! - i8:NOID8(i8:m,i8:n): this function returns a binomial coefficient (integer*8).
! - GPGEN(i:ctrl,i:ni,i[1]:vh,i[1]:trn,i[2]:cil): permutation generator with partial-ordering restrictions, returns each new permutation.
! - TR_CYCLE(i:ni,i[1]:trn,i:nc,i[2]:cyc): decomposes a permutation into permutation cycles and determines the sign of the permutation.
! - l:PERM_TRIVIAL(i:ni,i[1]:trn): checks whether the given permutation is trivial or not.
! - l:PERM_TRIVIAL_INT8(i8:ni,i8[1]:trn): checks whether the given permutation is trivial or not.
! - l:PERM_OK(i:ni,i[1]:trn): checks whether the given permutation is legitimate or not.
! - PERM2TRANS(i:ni,i[1]:trn1,i[1]:trn2,i:ntrp,i[2]:trp): creates a list of elementary transpositions relating one permutation to another.
! - PERMUTATION_CONVERTER(l:seq2pos,i:ni,i[1]:n2o,i[1]:o2n): converts between two permutation representations (N2O and O2N).
! - i:HASH_ARR_INT(i:hash_range,i:ni,i[1]:arr): returns a hash-mask for an integer array.
! - i:HASH_ARR_INT8(i:hash_range,i:ni,i8[1]:arr): returns a hash-mask for an integer*8 array.
! - i:CMP_MULTINDS(i:ml1,i[1]:m1,i:ml2,i[1]:m2): compares two integer multiindices.
! - i:CMP_MULTINDS_INT8(i:ml1,i8[1]:m1,i:ml2,i8[1]:m2): compares two integer*8 multiindices.
! - MULTINDX_MERGE(i:ml1,i[1]:m1,i:ml2,i[1]:m2,i:mlr,i[1]:mr,i:sign_corr): merges two multiindices, providing a potential sign correction.
! - i:CMP_ARRAYS_INT(l:preorder,i:ml1,i[1]:m1,i:ml2,i[1]:m2,i[1]:trn): compares two integer arrays with an optional preodering.
! - i:CMP_ARRAYS_INT8(l:preorder,i:ml1,i8[1]:m1,i:ml2,i8[1]:m2,i[1]:trn): compares two integer(8) arrays with an optional preodering.
! - CLANAL(i:nt,r8:ctol,r8[1]:dta,i:ncl,i[1]:cdta,r8[1]:cmv): cluster analysis of a one-dimensional set of points.
! - RANDOM_PERMUTATION(i:ni,i[1]:trn,l:no_trivial): returns a random permutation (default integer).
! - RANDOM_PERMUTATION_INT8(i8:ni,i8[1]:trn,l:no_trivial): returns a random permutation (integer*8 variant).
! - RANDOM_COMPOSITION(l:ordered,i:irange,i:ni,i[1]:trn): returns a random sequence of ni natural numbers from the range [1..irange] without repeats.
! - MERGE_SORT_INT(i:ni,i[1]:trn): fast sorting algorithm for an integer array (default integer).
! - MERGE_SORT_KEY_INT(i:ni,i[1]:key,i[1]:trn): fast sorting algorithm for an integer array, based on integer keys (default integer).
! - MERGE_SORT_INT8(i8:ni,i8[1]:trn): fast sorting algorithm for an integer*8 array.
! - MERGE_SORT_KEY_INT8(i8:ni,i8[1]:key,i8[1]:trn): fast sorting algorithm for an integer*8 array, based on integer keys (integer*8).
! - MERGE_SORT_REAL8(i:ni,r8[1]:trn): fast sorting algorithm for a real*8 array.
! - MERGE_SORT_KEY_REAL8(i:ni,r8[1]:key,i[1]:trn): fast sorting algorithm for an integer array, based on real*8 keys.
! - MERGE_SORT_CMPLX8(i:ni,c8[1]:trn): fast sorting algorithm for a complex*8 array.
! - MERGE_SORT_KEY_CMPLX8(i:ni,c8[1]:key,i[1]:trn): fast sorting algorithm for an integer array, based on comlex*8 keys.
!NOTES:
! - As a rule, a permutation is passed as an integer array (0:n), where the 0th element is the sign of the permutation {1..n}.
! - TRSIGN uses the buble algorithm => will be very slow for long permutations.
! - For some subroutines, the permutation {1..n} must contain all integers from the segment [1..n].
        implicit none
        private
!GLOBAL PARAMETERS:
        real(8), parameter, public:: DP_ZERO_THRESH=1d-13 !certified guaranteed precision of a double-precision real number (one order lower than the epsilon)
!PROCEDURE VISIBILITY:
        public trng
        public trsign
        public factorial
        public noid
        public noid8
        public gpgen
        public tr_cycle
        public perm_trivial
        public perm_trivial_int8
        public perm_ok
        public perm2trans
        public permutation_converter
        public hash_arr_int
        public hash_arr_int8
        public cmp_multinds
        public cmp_multinds_int8
        public multindx_merge
        public cmp_arrays_int
        public cmp_arrays_int8
        public clanal
        public random_permutation
        public random_permutation_int8
        public random_composition
        public merge_sort_int
        public merge_sort_key_int
        public merge_sort_int8
        public merge_sort_key_int8
        public merge_sort_real8
        public merge_sort_key_real8
        public merge_sort_cmplx8
        public merge_sort_key_cmplx8
!-----------------------------------
       contains
!METHODS:
        subroutine trng(ctrl,ni,trn,ngt)
!Permutation generator: Returns each subsequent permutation.
! CTRL - control argument. The first call must have CTRL<>0 and initialized TRN!
!        Then, on a regular output, CTRL will be 0 unless permutations are over (CTRL=-1).
! NI - number of indices to permute;
! TRN - current permutation (TRN(0) is the sign of the permutation);
! NGT - work structure. Do not change it outside this subroutine!
!NOTES:
! - Permutation index values are not restricted (any array of integer numbers).
	implicit none
	integer, intent(in):: ni
	integer, intent(inout):: ctrl,trn(0:*),ngt(0:*)
	integer j,k

	if(ctrl.ne.0) then !first call: NGT initialization. The original permutation is returned.
	 ngt(0)=0; do j=1,ni; ngt(j)=j-1; enddo
	 ctrl=0
	else !subsequent call: a new permutation is returned.
	 k=1 !maybe k=2 will accelerate
	 do while(k.le.ni)
	  if(ngt(k).ne.k-1) call transp(k,ngt(k)+1)
	  if(ngt(k).ne.0) then
	   call transp(k,ngt(k))
	   ngt(k)=ngt(k)-1
	   return
	  else
	   ngt(k)=k-1
	   k=k+1
	  endif
	 enddo
	 ctrl=-1
	endif
	return

	contains

	 subroutine transp(m,n)
	  implicit none
	  integer m,n,l
	  l=trn(m); trn(m)=trn(n); trn(n)=l; trn(0)=-trn(0)
	  return
	 end subroutine transp

        end subroutine trng
!------------------------------------------------------------------
	subroutine trsign(n,itr) !ITR - permutation of the length N
!This subroutine orders integers in ITR in an ascending order.
! ITR - integers. ITR(0) will be the sign of the ordered permutation.
! N - number of integers to be ordered.
!NOTES:
! - Permutation index values are not restricted (any array of integers).
! - Bubble algorithm (only for small arrays).
	implicit none
	integer, intent(in):: n
	integer, intent(inout):: itr(0:*)
	integer k,l,isgn
	k=1
	isgn=+1
	do while(k.lt.n)
	 if(itr(k).gt.itr(k+1)) then
	  l=itr(k); itr(k)=itr(k+1); itr(k+1)=l; isgn=-isgn
	  if(k.gt.1) then; k=k-1; else; k=2; endif
	 else
	  k=k+1
	 endif
	enddo
	itr(0)=isgn
	return
	end subroutine trsign
!---------------------------------------
	integer(8) function factorial(n) !returns N! for N>=0, and -1 otherwise
	implicit none
	integer, intent(in):: n
	integer(8) k

	if(n.ge.0) then
	 factorial=1_8
	 do k=2_8,int(n,8)
	  factorial=factorial*k
	  if(factorial.lt.0_8) then; write(*,*)'ERROR(combinatoric:factorial): integer(8) overflow!'; stop; endif !trap
	 enddo
	else
	 factorial=-1_8
	endif
	return
	end function factorial
!------------------------------------------------------------------------------------------------------
	integer function noid(m,n) !returns the number of unique distributions of N objects on M places
	implicit none
	integer, intent(in):: m,n
	integer k,l

	if(n.gt.m.or.n.lt.0.or.m.lt.0) then
	 noid=0
	 return
	elseif(n.eq.m.or.n.eq.0) then
	 noid=1
	 return
	endif
	noid=1; l=m
	do k=1,n; noid=noid*l/k; l=l-1; enddo
	if(noid.le.0) then; write(*,*)'ERROR(combinatoric:noid): integer overflow:',m,n,noid; stop; endif !trap
	return
	end function noid
!----------------------------------------------------------------------------------------------------------
	integer(8) function noid8(m,n) !returns the number of unique distributions of N objects on M places
	implicit none
	integer(8), intent(in):: m,n
	integer(8) k,l

	if(n.gt.m.or.n.lt.0.or.m.lt.0) then
	 noid8=0_8
	 return
	elseif(n.eq.m.or.n.eq.0) then
	 noid8=1_8
	 return
	endif
	noid8=1_8; l=m
	do k=1,n; noid8=noid8*l/k; l=l-1; enddo
	if(noid8.le.0_8) then; write(*,*)'ERROR(combinatoric:noid8): integer*8 overflow: ',m,n,noid8; stop; endif !trap
	return
	end function noid8
!-------------------------------------------
	subroutine gpgen(ctrl,ni,vh,trn,cil)
!This subroutine generates all unique permutations of NI items,
!in which the items belonging to the same host are always ordered.
!INPUT:
! - ctrl - control argument: at the begining must be <>0; 0 - next permutation; -1 - permutations are over.
! - ni - number of items;
! - vh(1:ni) - index hosts;
!INPUT(dummy)/OUTPUT:
! - trn(0:ni) - current permutation (trn(0) is the sign);
! - cil(0:1,0:ni) - connected list (for internal use only, do not set or change it outside!).
!OUTPUT:
! - trn(0:ni) - current permutation (trn(0) is the sign).
!NOTES:
! - The first permutation is created here (fully ordered one).
!   Its sign is +1. Each next permutation is generated from the previous one.
	implicit none
!------------------------------------------
	integer, parameter:: max_item=16384 !max allowed number of items (because of the permutation sign determination, see below)
!------------------------------------------
	integer i,j,k,l,m,n,k1,k2,k3,k4,k5,k6,ks,kf,ierr
	integer, intent(in):: ni,vh(1:ni)
	integer, intent(inout):: ctrl,trn(0:ni),cil(0:1,0:ni)

	if(ni.gt.0) then
	 if(ni.gt.max_item) then
	  write(*,*)'ERROR(gpgen): legnth of the permutation exceeds the maximal value: ',max_item,ni
	  stop
	 endif
	 if(ctrl.ne.0) then
	  call first_call
	 else
!free the last box:
	  m=vh(ni); n=ni
	  do while(vh(n).eq.m)
	   call free_item(trn(n)); n=n-1
	   if(n.eq.0) exit
	  enddo
!get the next composition:
	  if(n.gt.0) then
	   ks=-1
	   do while(n.gt.0)
!	    write(*,'(''DEBUG1: '',128(i1,1x))') trn(1:n) !debug
!	    write(*,'(4x,128(i2,1x))') (k6,k6=0,ni); write(*,'(4x,128(i2,1x))') cil(0,0:ni) !debug
!           write(*,'(4x,128(i2,1x))') cil(1,0:ni) !debug
	    if(ks.lt.0) then
	     i=trn(n); call free_item(i); j=cil(1,i)
	     if(j.gt.0) then
	      m=cil(1,j); call engage_item(j); trn(n)=j; ks=+1
	     endif
	    else
	     if(vh(n).eq.vh(n-1)) then
	      j=m
	      if(j.gt.0) then
	       m=cil(1,j); call engage_item(j); trn(n)=j
	      else
	       ks=-1
	      endif
	     else
	      j=cil(0,0); m=cil(1,j); call engage_item(j); trn(n)=j
	     endif
	    endif
!	    write(*,'(''DEBUG2: '',128(i1,1x))') trn(1:n) !debug
!	    write(*,'(4x,128(i2,1x))') (k6,k6=0,ni); write(*,'(4x,128(i2,1x))') cil(0,0:ni) !debug
!           write(*,'(4x,128(i2,1x))') cil(1,0:ni) !debug
	    n=n+ks
	    if(n.gt.ni) then !success
	     call determine_trn_sign
	     return
	    endif
	   enddo !n
	   ctrl=-1 !end
	  else
	   ctrl=-1
	  endif
	 endif
	else
	 ctrl=-1
	endif
	return

	contains

	 subroutine first_call
	  implicit none
	  integer j1
	  trn(0)=+1; do j1=1,ni; trn(j1)=j1; enddo
	  cil(0:1,1:ni)=-1; cil(0:1,0)=(/ni+1,0/)
	  ctrl=0
	  return
	 end subroutine first_call

	 subroutine free_item(it)
	  implicit none
	  integer, intent(in):: it
	  integer j1,j2
	  if(it.lt.cil(0,0)) then
	   if(cil(0,0).le.ni) then
	    cil(0:1,it)=(/0,cil(0,0)/); cil(0,cil(0,0))=it; cil(0,0)=it
	   else
	    cil(0:1,it)=(/0,0/); cil(0:1,0)=it
	   endif
	  elseif(it.gt.cil(1,0)) then
	   if(cil(1,0).ge.1) then
	    cil(0:1,it)=(/cil(1,0),0/); cil(1,cil(1,0))=it; cil(1,0)=it
	   else
	    cil(0:1,it)=(/0,0/); cil(0:1,0)=it
	   endif
	  else
	   if(it-cil(0,0).le.cil(1,0)-it) then !moving up: insert the new item between the minimal and maximal elements
	    j1=cil(0,0); do while(cil(1,j1).lt.it); j1=cil(1,j1); enddo
	    j2=cil(1,j1); cil(0:1,it)=(/j1,j2/); cil(1,j1)=it; cil(0,j2)=it
	   else !moving down: insert the new item between the minimal and maximal elements
	    j1=cil(1,0); do while(cil(0,j1).gt.it); j1=cil(0,j1); enddo
	    j2=cil(0,j1); cil(0:1,it)=(/j2,j1/); cil(1,j2)=it; cil(0,j1)=it
	   endif
	  endif
	  return
	 end subroutine free_item

	 subroutine engage_item(it)
	  implicit none
	  integer, intent(in):: it
	  integer j1,jd,ju
	  jd=1; ju=1
	  if(it.eq.cil(0,0)) then
	   j1=cil(1,it)
	   if(j1.gt.0) then; cil(0,j1)=0; cil(0,0)=j1; else; cil(0,0)=ni+1; endif
	   jd=0
	  endif
	  if(it.eq.cil(1,0)) then
	   j1=cil(0,it)
	   if(j1.gt.0) then; cil(1,j1)=0; cil(1,0)=j1; else; cil(1,0)=0; endif
	   ju=0
	  endif
	  if(jd*ju.eq.1) then; cil(1,cil(0,it))=cil(1,it); cil(0,cil(1,it))=cil(0,it); endif
	  cil(0:1,it)=-1
	  return
	 end subroutine engage_item

	 subroutine determine_trn_sign
	  implicit none
	  integer occ(max_item),j1,j2,j3,js
	  js=+1; occ(1:ni)=0; j1=1; j2=0; j3=0
	  do
	   if(occ(j1).eq.0) then
	    j3=j3+1; j2=j2+1; occ(j1)=1; j1=trn(j1)
	   else
	    if(mod(j2,2).eq.0) js=-js
	    if(j3.eq.ni) exit
	    j2=0; j1=1; do while(occ(j1).ne.0); j1=j1+1; enddo
	   endif
	  enddo
	  trn(0)=js
	  return
	 end subroutine determine_trn_sign

	end subroutine gpgen
!-----------------------------------------
	subroutine tr_cycle(ni,trn,nc,cyc)
!This subroutine extracts permutation cycles and the sign from a given permutation.
!INPUT:
! - ni - number of indices;
! - trn(0:ni) - permutation, in which trn(0) is the sign returned;
!OUTPUT:
! - trn(0) - sign of the permutation;
! - nc - number of permutation cycles;
! - cyc(0:1,1:ni) - permutation cycles: cyc(1,:) is an index of a cycle; cyc(0,:) is the number of the cycle the index belongs to.
!NOTE:
! - nc=-666 - means ERROR.
! - permutation index values must lie within the range [1..ni].
	implicit none
	integer, intent(in):: ni
	integer, intent(inout):: trn(0:*)
	integer, intent(out):: nc,cyc(0:1,*)
	integer, parameter:: max_in_mem=1024
	integer i,j,k,l,m,n,k1,k2,k3,k4,k5,k6,ks,kf,ierr
	integer, target:: ibuss(1:max_in_mem)
	integer, allocatable, target:: ibusa(:)
	integer, pointer:: ibus(:)

	nc=0
	if(ni.gt.0) then
	 if(ni.gt.max_in_mem) then; allocate(ibusa(1:ni)); ibus=>ibusa; else; ibus=>ibuss; endif
	 if(trn_ok()) then
!	  ibus(1:ni)=0 !busy flags
	  trn(0)=+1; n=0; m=0
	  do while(n.lt.ni)
	   i=m+1; do while(ibus(i).ne.0); i=i+1; enddo; m=i
	   nc=nc+1; l=i; j=0
	   do
	    n=n+1; j=j+1; cyc(0:1,n)=(/nc,trn(i)/); ibus(i)=nc
	    if(trn(i).eq.l) then; exit; else; i=trn(i); endif
	   enddo
	   if(mod(j,2).eq.0) trn(0)=-trn(0)
	  enddo
	 else
	  trn(0)=-667; nc=-666
	 endif
	 nullify(ibus); if(ni.gt.max_in_mem) deallocate(ibusa)
	else
	 trn(0)=-666; nc=-666
	endif
	return

	contains

	 logical function trn_ok()
	  integer j1
	  ibus(1:ni)=0
	  do j1=1,ni
	   if(trn(j1).le.0.or.trn(j1).gt.ni) then; trn_ok=.false.; return; endif
	   if(ibus(trn(j1)).ne.0) then; trn_ok=.false.; return; endif
	   ibus(trn(j1))=j1
	  enddo
	  ibus(1:ni)=0
	  trn_ok=.true.
	  return
	 end function trn_ok

	end subroutine tr_cycle
!--------------------------------------------
	logical function perm_trivial(ni,trn)
!Checks whether the given permutation trn(0:ni) is trivial or not.
	implicit none
	integer, intent(in):: ni,trn(0:*)
	integer i
	perm_trivial=.true.
	do i=1,ni; if(trn(i).ne.i) then; perm_trivial=.false.; exit; endif; enddo
	return
	end function perm_trivial
!-------------------------------------------------
	logical function perm_trivial_int8(ni,trn)
!Checks whether the given permutation trn(0:ni) is trivial or not.
	implicit none
	integer(8), intent(in):: ni,trn(0:*)
	integer(8) i
	perm_trivial_int8=.true.
	do i=1,ni; if(trn(i).ne.i) then; perm_trivial_int8=.false.; exit; endif; enddo
	return
	end function perm_trivial_int8
!---------------------------------------
	logical function perm_ok(ni,trn)
!Checks whether the given permutation trn(0:ni) is legitimate or not.
!NOTE: keep the permutation fit into the stack!
	implicit none
	integer, intent(in):: ni,trn(0:*)
	integer i,j,ibus(1:ni)
	ibus(1:ni)=0
	do i=1,ni
	 j=trn(i)
	 if(j.le.0.or.j.gt.ni) then; perm_ok=.false.; return; endif
	 if(ibus(j).ne.0) then; perm_ok=.false.; return; else; ibus(j)=i; endif
	enddo
	perm_ok=.true.
	return
	end function perm_ok
!---------------------------------------------------
	subroutine perm2trans(ni,trn1,trn2,ntrp,trp)
!This subroutine creates a sequence of elementary pair transpositions
!which connect one permutation (old) with another one (new).
!The sign of the second (new) permutation is changed accordingly.
!Schematically: TRN1 ---list_of_transpositions---> TRN2 with a modified sign.
!INPUT:
! - ni - number of indices;
! - trn1(0:ni) - old permutation (&0 - sign);
! - trn2(0:ni) - new permutation (&0 - sign);
!OUTPUT:
! - ntrp - number of elementary transpositions;
! - trp(1:2,1:ntrp) - elementary transpositions in terms of old index numbers.
!                     Each transposition interchanges two old indices according to their original numbers;
! - trn2(0) - sign of the permutation TRN2 with respect to the permutation TRN1.
!NOTES:
! - Permutation index values must span the range [1..ni].
! - Because AUTOMATIC arrays are employed, the number of indices cannot be too large.
!   Switch to pointers if you want to process larger permutations.
	implicit none
	integer, intent(in):: ni,trn1(0:*),trn2(0:*)
	integer, intent(out):: ntrp,trp(2,*)
	integer i,j,k,l,m,n,k1,k2,k3,k4,k5,k6,ks,kf,ierr
	integer trn(1:ni),ipos(1:ni)
	integer isgn

	ntrp=0
	if(ni.gt.1) then
	 if(trn_ok()) then
	  isgn=+1; do j=1,ni; trn(j)=trn1(j); enddo; do j=1,ni; ipos(trn1(j))=j; enddo
	  do i=1,ni
	   if(trn(i).ne.trn2(i)) then
	    ntrp=ntrp+1; trp(1:2,ntrp)=(/trn(i),trn2(i)/)
	    j=trn(i); trn(i)=trn2(i); trn(ipos(trn2(i)))=j
	    ipos(j)=ipos(trn2(i)); ipos(trn2(i))=i
	    isgn=-isgn
	   endif
	  enddo
	 else
	  write(*,*)'ERROR(perm2trans): invalid input permutation: ',ni,trn1(1:ni),trn2(1:ni)
	  stop
	 endif
	endif
	return

	contains

	 logical function trn_ok()
	  integer j1
	  trn_ok=.true.
	  do j1=1,ni
	   if(trn1(j1).lt.1.or.trn1(j1).gt.ni.or.trn2(j1).lt.1.or.trn2(j1).gt.ni) then; trn_ok=.false.; exit; endif
	  enddo
	  return
	 end function trn_ok

	end subroutine perm2trans
!-----------------------------------------------------------
	subroutine permutation_converter(seq2pos,ni,n2o,o2n)
!This subroutine converts between two permutation representations:
!(n2o) new_to_old: a sequence of original item numbers (position --> ID);
!(o2n) old_to_new: new positions of items (ID --> new position).
!Briefly: n2o(POSITION)=ID; o2n(ID)=POSITION. They are inverse permutations.
!INPUT:
! - seq2pos - .TRUE. means a conversion from n2o to o2n, .FALSE. o2n to n2o;
! - ni - number of items;
! - n2o(0:ni)/o2n(0:ni);
!OUTPUT:
! - n2o(0:ni)/o2n(0:ni).
!NOTES:
! - The elements of n2o(1:ni)/o2n(1:ni) must span the range [1..ni] (no argument-validity check is done here!).
	implicit none
	logical, intent(in):: seq2pos
	integer, intent(in):: ni
	integer, intent(inout):: n2o(0:ni),o2n(0:ni)
	integer i,j,k,l,m,n,k0,k1,k2,k3,ks,kf,ierr

	if(seq2pos) then
	 o2n(0)=n2o(0); do i=1,ni; o2n(n2o(i))=i; enddo !loop over positions
	else
	 n2o(0)=o2n(0); do i=1,ni; n2o(o2n(i))=i; enddo !loop over item IDs
	endif
	return
	end subroutine permutation_converter
!-------------------------------------------------------
	integer function hash_arr_int(hash_range,ni,arr)
!This function returns a hash-mask for a given integer array.
!INPUT:
! - hash_range - defines the range of this function [0..hash_range-1];
! - ni - number of items in the array;
! - arr(0:ni-1) - items;
!OUTPUT:
! - hash_arr_int - hash-mask.
	implicit none
	integer, intent(in):: hash_range,ni
	integer, intent(in):: arr(0:*)
	integer i,j,k,l,m,n,k0,k1,k2,k3,ks,kf,ierr

	hash_arr_int=0
	if(hash_range.gt.0.and.ni.ge.0) then
	 do i=0,ni-1
	  hash_arr_int=hash_arr_int+mod(arr(i),hash_range)
	  hash_arr_int=mod(hash_arr_int,hash_range)
	 enddo
	else
	 write(*,*)'ERROR(combinatoric:hash_arr_int): invalid arguments: ',hash_range,ni
	 stop
	endif
	return
	end function hash_arr_int
!--------------------------------------------------------
	integer function hash_arr_int8(hash_range,ni,arr)
!This function returns a hash-mask for a given integer*8 array.
!INPUT:
! - hash_range - defines the range of this function [0..hash_range-1];
! - ni - number of items in the array;
! - arr(0:ni-1) - items (integer*8);
!OUTPUT:
! - hash_arr_int - hash-mask.
	implicit none
	integer, intent(in):: hash_range,ni
	integer(8), intent(in):: arr(0:*)
	integer i,j,k,l,m,n,k0,k1,k2,k3,ks,kf,ierr
	integer(8) hr

	hash_arr_int8=0
	if(hash_range.gt.0.and.ni.ge.0) then
	 hr=int(hash_range,8)
	 do i=0,ni-1
	  hash_arr_int8=hash_arr_int8+int(mod(arr(i),hr),4)
	  hash_arr_int8=mod(hash_arr_int8,hash_range)
	 enddo
	else
	 write(*,*)'ERROR(combinatoric:hash_arr_int8): invalid arguments: ',hash_range,ni
	 stop
	endif
	return
	end function hash_arr_int8
!---------------------------------------------------
	integer function cmp_multinds(ml1,m1,ml2,m2)
!This function compares two integer multiindices.
!INPUT:
! - m1(1:ml1) - 1st multiindex;
! - m2(1:ml2) - 2nd multiindex.
!OUTPUT:
! - cmp_multinds: -X (1<=X<=ml1,ml1=ml2): first difference in two multiindices occurs at position X (left-to-right) and m1(X)<m2(X);
!                 +X (1<=X<=ml1,ml1=ml2): first difference in two multiindices occurs at position X (left-to-right) and m1(X)>m2(X);
!                 -X (X>ml1,X>ml2): first multiindex is shorter than the second one (ml1<ml2);
!                 +X (X>ml1,X>ml2): first multiindex is longer than the second one (ml1>ml2);
!                  0 (ml1=ml2): the two multiindices are equal.
	implicit none
	integer, intent(in):: ml1,ml2,m1(1:*),m2(1:*)
	integer i,j,k,l,m,n,k0,k1,k2,k3,ks,kf,ierr

	if(ml1.ge.0.and.ml2.ge.0) then
	 if(ml1.eq.ml2) then
	  cmp_multinds=0
	  do i=1,ml1
	   if(m1(i).ne.m2(i)) then; cmp_multinds=sign(i,m1(i)-m2(i)); exit; endif
	  enddo
	 elseif(ml1.lt.ml2) then
	  cmp_multinds=-(max(ml1,ml2)+1)
	 else
	  cmp_multinds=+(max(ml1,ml2)+1)
	 endif
	else
	 write(*,'("ERROR(combinatoric:cmp_multinds): negative multiindex length passed: ",i10,1x,i10)') ml1,ml2
	 stop
	endif
	return
	end function cmp_multinds
!--------------------------------------------------------
	integer function cmp_multinds_int8(ml1,m1,ml2,m2)
!This function compares two integer*8 multiindices.
!INPUT:
! - m1(1:ml1) - 1st multiindex (integer*8);
! - m2(1:ml2) - 2nd multiindex (integer*8).
!OUTPUT:
! - cmp_multinds: -X (1<=X<=ml1,ml1=ml2): first difference in two multiindices occurs at position X (left-to-right) and m1(X)<m2(X);
!                 +X (1<=X<=ml1,ml1=ml2): first difference in two multiindices occurs at position X (left-to-right) and m1(X)>m2(X);
!                 -X (X>ml1,X>ml2): first multiindex is shorter than the second one (ml1<ml2);
!                 +X (X>ml1,X>ml2): first multiindex is longer than the second one (ml1>ml2);
!                  0 (ml1=ml2): the two multiindices are equal.
	implicit none
	integer, intent(in):: ml1,ml2
	integer(8), intent(in):: m1(1:*),m2(1:*)
	integer i,j,k,l,m,n,k0,k1,k2,k3,ks,kf,ierr

	if(ml1.ge.0.and.ml2.ge.0) then
	 if(ml1.eq.ml2) then
	  cmp_multinds_int8=0
	  do i=1,ml1
	   if(m1(i).ne.m2(i)) then; cmp_multinds_int8=int(sign(int(i,8),m1(i)-m2(i)),4); exit; endif
	  enddo
	 elseif(ml1.lt.ml2) then
	  cmp_multinds_int8=-(max(ml1,ml2)+1)
	 else
	  cmp_multinds_int8=+(max(ml1,ml2)+1)
	 endif
	else
	 write(*,'("ERROR(combinatoric:cmp_multinds_int8): negative multiindex length passed: ",i10,1x,i10)') ml1,ml2
	 stop
	endif
	return
	end function cmp_multinds_int8
!----------------------------------------------------------------
	subroutine multindx_merge(ml1,m1,ml2,m2,mlr,mr,sign_corr)
!This subroutine merges two multiindices (with index reodering).
!INPUT:
! - ml1 - length of the 1st multiindex;
! - m1(1:ml1) - 1st multiindex (left);
! - ml2 - length of the 2nd multiindex;
! - m2(1:ml2) - 2nd multiindex (right);
!OUTPUT:
! - mlr - length of the multiindex-result (ml1+ml2);
! - mr(1:mlr) - multiindex-result;
! - sign_corr - sign correction (+1/-1/0): 0 when a repeated index is present.
!NOTES:
! - No error checks.
	implicit none
	integer, intent(in):: ml1,m1(1:*),ml2,m2(1:*)
	integer, intent(out):: mlr,mr(1:*),sign_corr
	integer i,j,k,l,m,n,k0,k1,k2,k3,ks,kf,ierr

	mlr=0; sign_corr=+1
!merge:
	if(ml1.gt.0.and.ml2.gt.0) then
	 k1=1; k2=1
	 mloop: do
	  mlr=mlr+1
	  if(m1(k1).le.m2(k2)) then
	   mr(mlr)=m1(k1)
	   if(k1.lt.ml1) then
	    k1=k1+1
	   else
	    l=ml2-k2+1; mr(mlr+1:mlr+l)=m2(k2:ml2); mlr=mlr+l
	    exit mloop
	   endif
	  else
	   mr(mlr)=m2(k2); sign_corr=sign_corr*(-1+2*mod(ml1-k1,2))
	   if(k2.lt.ml2) then
	    k2=k2+1
	   else
	    l=ml1-k1+1; mr(mlr+1:mlr+l)=m1(k1:ml1); mlr=mlr+l
	    exit mloop
	   endif
	  endif
	 enddo mloop
	elseif(ml1.gt.0.and.ml2.le.0) then
	 mlr=ml1; mr(1:ml1)=m1(1:ml1)
	elseif(ml1.le.0.and.ml2.gt.0) then
	 mlr=ml2; mr(1:ml2)=m2(1:ml2)
	endif
!check index repeats:
	do i=1,mlr-1; if(mr(i).eq.mr(i+1)) then; sign_corr=0; exit; endif; enddo
	return
	end subroutine multindx_merge
!------------------------------------------------------------------
	integer function cmp_arrays_int(preorder,ml1,m1,ml2,m2,trn)
!This function compares two integer arrays with an optional preodering.
!INPUT:
! - preorder - if .true., both arrays will be ordered before the comparison;
! - ml1 - length of the 1st array;
! - m1(1:ml1) - 1st array;
! - ml2 - length of the 2nd array;
! - m2(1:ml2) - 2nd array;
!OUTPUT:
! - cmp_arrays_int: -X (1<=X<=ml1,ml1=ml2): first difference in two arrays occurs at position X (left-to-right) and m1(X)<m2(X);
!                   +X (1<=X<=ml1,ml1=ml2): first difference in two arrays occurs at position X (left-to-right) and m1(X)>m2(X);
!                   -X (X>ml1,X>ml2): first array is shorter than the second one (ml1<ml2);
!                   +X (X>ml1,X>ml2): first array is longer than the second one (ml1>ml2);
!                    0 (ml1=ml2): the two arrays are equal.
! - trn(0:) - if preorder=.true. and the arrays are equal (cmp_arrays_int=0),
!             this (optional) output array will contain the permutation matching the two arrays:
!             A new order of old elements of the 2nd array (N2O) that matches the 1st array (both with the original ordering).
	implicit none
	logical, intent(in):: preorder
	integer, intent(in):: ml1,ml2
	integer, intent(in):: m1(1:ml1),m2(1:ml2)
	integer, intent(out), optional:: trn(0:*)
	integer i,j,k,l,m,n,k0,k1,k2,k3,ks,kf,ierr
	integer, allocatable:: prm1(:),prm2(:)

	if(ml1.ge.0.and.ml2.ge.0) then
	 if(ml1.lt.ml2) then
	  cmp_arrays_int=-(max(ml1,ml2)+1)
	 elseif(ml1.gt.ml2) then
	  cmp_arrays_int=(max(ml1,ml2)+1)
	 else !ml1=ml2
	  cmp_arrays_int=0
	  if(preorder) then
	   allocate(prm1(0:ml1),prm2(0:ml2),STAT=ierr)
	   if(ierr.ne.0) then; write(*,*)'ERROR(combinatoric:cmp_arrays_int): memory allocation failed!'; stop; endif
	   prm1(0)=+1; do i=1,ml1; prm1(i)=i; enddo
	   prm2(0)=+1; do i=1,ml2; prm2(i)=i; enddo
	   call merge_sort_key_int(ml1,m1,prm1) !prm1 is N2O
	   call merge_sort_key_int(ml2,m2,prm2) !prm2 is N2O
	   do i=1,ml1
	    if(m1(prm1(i)).ne.m2(prm2(i))) then; cmp_arrays_int=sign(i,m1(prm1(i))-m2(prm2(i))); exit; endif
	   enddo
	   if(cmp_arrays_int.eq.0.and.present(trn)) then
	    trn(0)=prm1(0)*prm2(0); do i=1,ml1; trn(prm1(i))=prm2(i); enddo
	   endif
	   deallocate(prm1,prm2)
	  else
	   do i=1,ml1
	    if(m1(i).ne.m2(i)) then; cmp_arrays_int=sign(i,m1(i)-m2(i)); exit; endif
	   enddo
	  endif
	 endif
	else !invalid arguments
	 write(*,*)'ERROR(combinatoric:cmp_arrays_int): invalid arguments: ',ml1,ml2
	 stop
	endif
	return
	end function cmp_arrays_int
!-------------------------------------------------------------------
	integer function cmp_arrays_int8(preorder,ml1,m1,ml2,m2,trn)
!This function compares two integer(8) arrays with an optional preodering.
!INPUT:
! - preorder - if .true., both arrays will be ordered before the comparison;
! - ml1 - length of the 1st array;
! - m1(1:ml1) - 1st array;
! - ml2 - length of the 2nd array;
! - m2(1:ml2) - 2nd array;
!OUTPUT:
! - cmp_arrays_int8: -X (1<=X<=ml1,ml1=ml2): first difference in two arrays occurs at position X (left-to-right) and m1(X)<m2(X);
!                    +X (1<=X<=ml1,ml1=ml2): first difference in two arrays occurs at position X (left-to-right) and m1(X)>m2(X);
!                    -X (X>ml1,X>ml2): first array is shorter than the second one (ml1<ml2);
!                    +X (X>ml1,X>ml2): first array is longer than the second one (ml1>ml2);
!                     0 (ml1=ml2): the two arrays are equal.
! - trn(0:) - if preorder=.true. and the arrays are equal (cmp_arrays_int8=0),
!             this (optional) output array will contain the permutation matching the two arrays:
!             A new order of old elements of the 2nd array (N2O) that matches the 1st array (both with the original ordering).
	implicit none
	logical, intent(in):: preorder
	integer, intent(in):: ml1,ml2
	integer(8), intent(in):: m1(1:ml1),m2(1:ml2)
	integer, intent(out), optional:: trn(0:*)
	integer i,j,k,l,m,n,k0,k1,k2,k3,ks,kf,ierr
	integer(8), allocatable:: prm1(:),prm2(:)
	integer(8) ml

	if(ml1.ge.0.and.ml2.ge.0) then
	 if(ml1.lt.ml2) then
	  cmp_arrays_int8=-(max(ml1,ml2)+1)
	 elseif(ml1.gt.ml2) then
	  cmp_arrays_int8=(max(ml1,ml2)+1)
	 else !ml1=ml2
	  cmp_arrays_int8=0; ml=int(ml1,8)
	  if(preorder) then
	   allocate(prm1(0:ml),prm2(0:ml),STAT=ierr)
	   if(ierr.ne.0) then; write(*,*)'ERROR(combinatoric:cmp_arrays_int8): memory allocation failed!'; stop; endif
	   prm1(0)=+1_8; do i=1,ml1; prm1(i)=int(i,8); enddo
	   prm2(0)=+1_8; do i=1,ml2; prm2(i)=int(i,8); enddo
	   call merge_sort_key_int8(ml,m1,prm1) !prm1 is N2O
	   call merge_sort_key_int8(ml,m2,prm2) !prm2 is N2O
	   do i=1,ml1
	    if(m1(prm1(i)).ne.m2(prm2(i))) then; cmp_arrays_int8=int(sign(int(i,8),m1(prm1(i))-m2(prm2(i))),4); exit; endif
	   enddo
	   if(cmp_arrays_int8.eq.0.and.present(trn)) then
	    trn(0)=int(prm1(0)*prm2(0),4); do i=1,ml1; trn(prm1(i))=int(prm2(i),4); enddo
	   endif
	   deallocate(prm1,prm2)
	  else
	   do i=1,ml1
	    if(m1(i).ne.m2(i)) then; cmp_arrays_int8=int(sign(int(i,8),m1(i)-m2(i)),4); exit; endif
	   enddo
	  endif
	 endif
	else !invalid arguments
	 write(*,*)'ERROR(combinatoric:cmp_arrays_int8): invalid arguments: ',ml1,ml2
	 stop
	endif
	return
	end function cmp_arrays_int8
!--------------------------------------------------
	subroutine clanal(nt,ctol,dta,ncl,cdta,cmv)
!This subroutine clusterizes a one-dimensional set of points.
!INPUT:
! - nt - number of points;
! - ctol - clusterization tolerance;
! - dta(1:nt) - points;
!OUTPUT:
! - ncl - number of classes;
! - cdta(1:nt) - class of each point;
! - cmv(1:ncl) - class mean values.
!NOTES:
! - if no points has been passed, NCL is set to -1.
	implicit none
	integer i,j,k,l,m,n,k1,k2,k3,k4,k5,k6,ks,kf,ierr
	integer, intent(in):: nt        !number of points
	real(8), intent(in):: ctol      !clusterization tolerance (0..1)
	real(8), intent(in):: dta(*)    !data (points)
	integer, intent(out):: ncl      !number of classes
	integer, intent(out):: cdta(*)  !class# the point belongs to
	real(8), intent(out):: cmv(*)   !class mean values
	real(8) wh,sw,minv,maxv,val
	integer nfup,nfdp

	if(nt.gt.0) then
!calculate range of the values:
	 minv=dta(1); maxv=dta(1)
	 do k=2,nt
	  minv=min(dta(k),minv); maxv=max(dta(k),maxv)
	 enddo
	 wh=maxv-minv   !width of the distribution
	 sw=wh/dble(nt) !specific width
!init cdta:
	 cdta(1:nt)=0   !all points are not classified at the begining
!clusterize:
	 ncl=1         !current class #
	 cdta(1)=1     !element 1 is assigned to class 1
	 nfup=2        !number of the first unclassified point
	 n=1           !will be the number of points in the current class
	 cmv(1)=dta(1)
	 do            !until all the points are classified
	  nfdp=0                     !number of the 1st declined point
	  do k=nfup,nt
	   if(cdta(k).eq.0) then
	    val=dist(k,ncl)
	    if(val.lt.ctol*sw) then  !accepted to the current class
	     cdta(k)=ncl; cmv(ncl)=cmv(ncl)+dta(k)
	     n=n+1
	    else                     !declined
	     if(nfdp.eq.0) nfdp=k
	    endif
	   endif
	  enddo
	  cmv(ncl)=cmv(ncl)/dble(n)  !mean value of the current class
	  if(nfdp.gt.0) then
	   ncl=ncl+1; cdta(nfdp)=ncl
	   nfup=nfdp+1; cmv(ncl)=dta(nfdp)
	   n=1
	  else
	   exit  !all points have been classified
	  endif
	 enddo
	else
	 ncl=-1  !no input data found
	endif   !nt>0: points exist

	return

	contains

	 real(8) function dist(i_point,class_num)
	  integer, intent(in):: i_point,class_num
	  integer npc,lp
	  dist=0d0
	  npc=0      !number of points found for the class#class_num
	  do lp=1,nt
	   if(cdta(lp).eq.class_num) then
	    dist=dist+abs(dta(i_point)-dta(lp))
	    npc=npc+1
	   endif
	  enddo
	  dist=dist/dble(npc)
	  return
	 end function dist

	end subroutine clanal
!-------------------------------------------------------
	subroutine random_permutation(ni,trn,no_trivial)
!This subroutine returns a random permutation of NI items [1..NI].
!INPUT:
! - ni - number of items, range [1:ni], if ni<=0 nothing will be done;
!OUTPUT:
! - trn(0:ni) - generated permutation (trn(0) - sign of the permutation);
	implicit none
	integer, intent(in):: ni
	integer, intent(out):: trn(0:ni)
	logical, intent(in), optional:: no_trivial
!----------------------------------------------
	integer, parameter:: random_chunk=2**10 !size of the chunk of random numbers generated in one call
	integer, parameter:: num_repeats=5      !the bigger the number, the better the generator quality (more expensive)
!----------------------------------------------
	integer i,j,k,l,m,n,nr,ierr
	real(8):: ra(1:random_chunk)

	if(ni.gt.0) then
	 trn(0)=+1; do i=1,ni; trn(i)=i; enddo !initial permutation
	 ploop: do
	  do nr=1,num_repeats
	   do i=1,ni,random_chunk
	    l=min(i+random_chunk-1,ni)-i+1
	    call random_number(ra(1:l))
	    ra(1:l)=ra(1:l)*2d0
	    do k=1,l
	     if(ra(k).lt.1d0) then
	      j=i+k-1
	      n=int(ra(k)*dble(ni))+1; if(n.gt.ni) n=ni
	      if(n.ne.j) then; m=trn(j); trn(j)=trn(n); trn(n)=m; trn(0)=-trn(0); endif
	     endif
	    enddo
	   enddo
	  enddo
	  if(present(no_trivial)) then
	   if(no_trivial) then
	    if(.not.perm_trivial(ni,trn)) exit ploop
	   else
	    exit ploop
	   endif
	  else
	   exit ploop
	  endif
	 enddo ploop
!	else
!	 write(*,*)'ERROR(random_permutation): negative or zero number of items: ',ni
!	 stop
	endif
	return
	end subroutine random_permutation
!------------------------------------------------------------
	subroutine random_permutation_int8(ni,trn,no_trivial)
!This subroutine returns a random permutation of NI items [1..NI].
!INPUT:
! - ni - number of items, range [1:ni], if ni<=0 nothing will be done;
!OUTPUT:
! - trn(0:ni) - generated permutation (trn(0) - sign of the permutation);
	implicit none
	integer(8), intent(in):: ni
	integer(8), intent(out):: trn(0:ni)
	logical, intent(in), optional:: no_trivial
!-------------------------------------------------
	integer(8), parameter:: random_chunk=2**10 !size of the chunk of random numbers generated in one call
	integer(8), parameter:: num_repeats=5_8    !the bigger the number, the better the generator quality (more expensive)
!-------------------------------------------------
	integer(8) i,j,k,l,m,n,nr,ierr
	real(8):: ra(1:random_chunk)

	if(ni.gt.0_8) then
	 trn(0)=+1_8; do i=1_8,ni; trn(i)=i; enddo !initial permutation
	 ploop: do
	  do nr=1_8,num_repeats
	   do i=1_8,ni,random_chunk
	    l=min(i+random_chunk-1_8,ni)-i+1_8
	    call random_number(ra(1_8:l))
	    ra(1_8:l)=ra(1_8:l)*2d0
	    do k=1_8,l
	     if(ra(k).lt.1d0) then
	      j=i+k-1_8
	      n=int(ra(k)*dble(ni),8)+1_8; if(n.gt.ni) n=ni
	      if(n.ne.j) then; m=trn(j); trn(j)=trn(n); trn(n)=m; trn(0)=-trn(0); endif
	     endif
	    enddo
	   enddo
	  enddo
	  if(present(no_trivial)) then
	   if(no_trivial) then
	    if(.not.perm_trivial_int8(ni,trn)) exit ploop
	   else
	    exit ploop
	   endif
	  else
	   exit ploop
	  endif
	 enddo ploop
!	else
!	 write(*,*)'ERROR(random_permutation_int8): negative or zero number of items: ',ni
!	 stop
	endif
	return
	end subroutine random_permutation_int8
!-----------------------------------------------------------
	subroutine random_composition(ordered,irange,ni,trn)
!This subroutine returns a random sequence of ni natural numbers from the range [1..ni] without repeats.
!INPUT:
! - ordered - if .true. the sequence will be ordered;
! - irange - range of natural numbers to be used: [1..irange];
! - ni - number of elements in the sequence (length of the sequence);
!OUTPUT:
! - trn(0:ni) - the sequence generated, trn(0) is the sign of the permutation if unordered.
	implicit none
	logical, intent(in):: ordered
	integer, intent(in):: irange
	integer, intent(in):: ni
	integer, intent(out):: trn(0:ni)
	integer, parameter:: rnd_chunk=2**12
	integer i,j,k,l,m,n,k0,k1,k2,k3,ks,kf,ierr
	integer, allocatable:: prm(:)
	real(8) rnd_buf(1:rnd_chunk),rn,accept_thresh

	accept_thresh(k,l)=dble(k-l)/dble(k) !k>=l, k!=0: k - amount of objects left for selection; l - number of items to select.

	if(ni.ge.0.and.irange.ge.1.and.irange.ge.ni) then
	 if(ni.gt.0) then
	  if(ordered) then
	   k=irange; l=ni; k0=0; n=0
	   do i=1,irange
	    if(k0.eq.0) then; k0=min(rnd_chunk,k); call random_number(rnd_buf(1:k0)); endif
	    if(rnd_buf(k0).ge.accept_thresh(k,l)) then
	     n=n+1; trn(n)=i
	     l=l-1; if(l.eq.0) exit
	    endif
	    k=k-1; k0=k0-1
	   enddo
	   if(n.eq.ni) then
	    trn(0)=+1
	   else
	    write(*,*)'ERROR(combinatoric:random_composition): trap: invalid number of items: ',n,ni,irange,ordered
	    stop
	   endif
	  else
	   allocate(prm(0:ni),STAT=ierr)
	   if(ierr.ne.0) then; write(*,*)'ERROR(combinatoric:random_composition): allocation failed!'; stop; endif
	   prm(0)=+1; do i=1,ni; prm(i)=i; enddo
	   call random_permutation(ni,prm)
	   k=irange; l=ni; k0=0; n=0
	   do i=1,irange
	    if(k0.eq.0) then; k0=min(rnd_chunk,k); call random_number(rnd_buf(1:k0)); endif
	    if(rnd_buf(k0).ge.accept_thresh(k,l)) then
	     n=n+1; trn(prm(n))=i
	     l=l-1; if(l.eq.0) exit
	    endif
	    k=k-1; k0=k0-1
	   enddo
	   deallocate(prm)
	   if(n.eq.ni) then
	    trn(0)=prm(0)
	   else
	    write(*,*)'ERROR(combinatoric:random_composition): trap: invalid number of items: ',n,ni,irange,ordered
	    stop
	   endif
	  endif
	 endif
	else
	 write(*,*)'ERROR(combinatoric:random_composition): incompatible or invalid arguments: ',ni,irange
	 stop
	endif
	return
	end subroutine random_composition
!----------------------------------------
	subroutine merge_sort_int(ni,trn)
!This subroutine sorts an array of NI items in a non-descending order.
!The algorithm was suggested by Johann von Neumann.
!INPUT:
! - ni - number of items;
! - trn(1:ni) - items (an array of arbitrary integers);
!OUTPUT:
! - trn(0:ni) - sorted items and sign in trn(0).
!NOTES:
! - In order to accelerate the procedure use flip/flop for TRN(:)||PRM(:).
	implicit none
	integer, intent(in):: ni
	integer, intent(inout):: trn(0:ni)
	integer i,j,k,l,m,n,k0,k1,k2,k3,k4,k5,k6,k7,ks,kf,ierr
	integer, allocatable:: prm(:)

	trn(0)=+1
	if(ni.gt.1) then
	 allocate(prm(1:ni))
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
	     if(trn(k1)-trn(k3).gt.0) then
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
	 deallocate(prm)
	endif
	return
	end subroutine merge_sort_int
!------------------------------------------------
	subroutine merge_sort_key_int(ni,key,trn)
!This subroutine sorts an array of NI items in a non-descending order according to their keys.
!The algorithm is due to Johann von Neumann.
!INPUT:
! - ni - number of items;
! - key(1:ni) - item keys (retrieved by old item numbers!): arbitrary integers;
! - trn(0:ni) - initial permutation of ni items (a sequence of old numbers), trn(0) is the initial sign;
!OUTPUT:
! - trn(0:ni) - sorted permutation (new sequence of old numbers) of ni items (according to their keys), the sign is in trn(0).
	implicit none
	integer, intent(in):: ni,key(1:ni)
	integer, intent(inout):: trn(0:ni)
	integer, parameter:: max_in_mem=1024
	integer i,j,k,l,m,n,k0,k1,k2,k3,k4,k5,k6,k7,ks,kf,ierr
	integer, target:: prms(1:max_in_mem)
	integer, allocatable,target:: prma(:)
	integer, pointer:: prm(:)

	if(ni.gt.1) then
	 if(ni.le.max_in_mem) then; prm=>prms; else; allocate(prma(1:ni)); prm=>prma; endif
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
	 nullify(prm); if(ni.gt.max_in_mem) deallocate(prma)
	endif
	return
	end subroutine merge_sort_key_int
!-----------------------------------------
	subroutine merge_sort_int8(ni,trn)
!This subroutine sorts an array of NI items in a non-descending order.
!The algorithm was suggested by Johann von Neumann.
!INPUT:
! - ni - number of items;
! - trn(1:ni) - items (arbitrary integer*8 numbers);
!OUTPUT:
! - trn(0:ni) - sorted items and sign in trn(0).
!NOTES:
! - In order to accelerate the procedure use flip/flop for TRN(:)||PRM(:).
	implicit none
	integer(8), intent(in):: ni
	integer(8), intent(inout):: trn(0:ni)
	integer(8) i,j,k,l,m,n,k0,k1,k2,k3,k4,k5,k6,k7,ks,kf,ierr
	integer(8), allocatable:: prm(:)

	trn(0)=+1_8
	if(ni.gt.1_8) then
	 allocate(prm(1_8:ni))
	 n=1_8
	 do while(n.lt.ni)
	  m=n*2_8
	  do i=1_8,ni,m
	   k1=i; k2=i+n
	   if(k2.gt.ni) then
	    k2=ni+1_8; k3=0_8; k4=0_8 !no right block, only left block
	   else
	    k3=i+n; k4=min(ni+1_8,i+m) !right block present
	   endif
	   kf=min(ni+1_8,i+m)-i; l=0_8
	   do while(l.lt.kf)
	    if(k3.ge.k4) then !right block is over
	     prm(i+l:i+kf-1_8)=trn(k1:k2-1_8); l=kf
	    elseif(k1.ge.k2) then !left block is over
	     prm(i+l:i+kf-1_8)=trn(k3:k4-1_8); l=kf
	    else
	     if(trn(k1)-trn(k3).gt.0_8) then
	      prm(i+l)=trn(k3); k3=k3+1_8; trn(0_8)=(1_8-2_8*mod(k2-k1,2_8))*trn(0_8)
	     else
	      prm(i+l)=trn(k1); k1=k1+1_8
	     endif
	     l=l+1_8
	    endif
	   enddo
	  enddo
	  trn(1_8:ni)=prm(1_8:ni)
	  n=m
	 enddo
	 deallocate(prm)
	endif
	return
	end subroutine merge_sort_int8
!-------------------------------------------------
	subroutine merge_sort_key_int8(ni,key,trn)
!This subroutine sorts an array of NI items in a non-descending order according to their keys.
!The algorithm is due to Johann von Neumann.
!INPUT:
! - ni - number of items;
! - key(1:ni) - item keys (retrieved by old item numbers!): arbitrary integer*8 numbers;
! - trn(0:ni) - initial permutation of ni items (a sequence of old numbers), trn(0) is the initial sign;
!OUTPUT:
! - trn(0:ni) - sorted permutation of ni items (according to their keys), the sign is in trn(0).
	implicit none
	integer(8), intent(in):: ni,key(1:ni)
	integer(8), intent(inout):: trn(0:ni)
	integer(8), parameter:: max_in_mem=1024
	integer(8) i,j,k,l,m,n,k0,k1,k2,k3,k4,k5,k6,k7,ks,kf,ierr
	integer(8), target:: prms(1:max_in_mem)
	integer(8), allocatable,target:: prma(:)
	integer(8), pointer:: prm(:)

	if(ni.gt.1_8) then
	 if(ni.le.max_in_mem) then; prm=>prms; else; allocate(prma(1_8:ni)); prm=>prma; endif
	 n=1_8
	 do while(n.lt.ni)
	  m=n*2_8
	  do i=1_8,ni,m
	   k1=i; k2=i+n
	   if(k2.gt.ni) then
	    k2=ni+1_8; k3=0_8; k4=0_8 !no right block, only left block
	   else
	    k3=i+n; k4=min(ni+1_8,i+m) !right block present
	   endif
	   kf=min(ni+1_8,i+m)-i; l=0_8
	   do while(l.lt.kf)
	    if(k3.ge.k4) then !right block is over
	     prm(i+l:i+kf-1_8)=trn(k1:k2-1_8); l=kf
	    elseif(k1.ge.k2) then !left block is over
	     prm(i+l:i+kf-1_8)=trn(k3:k4-1_8); l=kf
	    else
	     if(key(trn(k1))-key(trn(k3)).gt.0_8) then
	      prm(i+l)=trn(k3); k3=k3+1_8; trn(0_8)=(1_8-2_8*mod(k2-k1,2_8))*trn(0_8)
	     else
	      prm(i+l)=trn(k1); k1=k1+1_8
	     endif
	     l=l+1_8
	    endif
	   enddo
	  enddo
	  trn(1_8:ni)=prm(1_8:ni)
	  n=m
	 enddo
	 nullify(prm); if(ni.gt.max_in_mem) deallocate(prma)
	endif
	return
	end subroutine merge_sort_key_int8
!------------------------------------------
	subroutine merge_sort_real8(ni,trn)
!This subroutine sorts an array of NI items in a non-descending order.
!The algorithm was suggested by Johann von Neumann.
!INPUT:
! - ni - number of items;
! - trn(1:ni) - items (arbitrary real*8 numbers);
!OUTPUT:
! - trn(0:ni) - sorted items and sign in trn(0).
	implicit none
	integer, intent(in):: ni
	real(8), intent(inout):: trn(0:ni)
	integer i,j,k,l,m,n,k0,k1,k2,k3,k4,k5,k6,k7,ks,kf,ierr
	real(8), parameter:: ds(0:1)=(/+1d0,-1d0/)
	real(8), allocatable:: prm(:)

	trn(0)=+1d0
	if(ni.gt.1) then
	 allocate(prm(1:ni))
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
	     if(trn(k1)-trn(k3).gt.0d0) then
	      prm(i+l)=trn(k3); k3=k3+1; trn(0)=ds(mod(k2-k1,2))*trn(0)
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
	 deallocate(prm)
	endif
	return
	end subroutine merge_sort_real8
!--------------------------------------------------
	subroutine merge_sort_key_real8(ni,key,trn)
!This subroutine sorts an array of NI items in a non-descending order according to their keys.
!The algorithm is due to Johann von Neumann.
!INPUT:
! - ni - number of items;
! - key(1:ni) - item keys (retrieved by old item numbers!): arbitrary real*8 numbers;
! - trn(0:ni) - initial permutation of ni items (a sequence of old numbers), trn(0) is the initial sign;
!OUTPUT:
! - trn(0:ni) - sorted permutation of ni items (according to their keys), the sign is in trn(0).
	implicit none
	integer, intent(in):: ni
	real(8), intent(in):: key(1:ni)
	integer, intent(inout):: trn(0:ni)
	integer, parameter:: max_in_mem=1024
	integer i,j,k,l,m,n,k0,k1,k2,k3,k4,k5,k6,k7,ks,kf,ierr
	integer, target:: prms(1:max_in_mem)
	integer, allocatable,target:: prma(:)
	integer, pointer:: prm(:)

	if(ni.gt.1) then
	 if(ni.le.max_in_mem) then; prm=>prms; else; allocate(prma(1:ni)); prm=>prma; endif
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
	     if(key(trn(k1))-key(trn(k3)).gt.0d0) then
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
	 nullify(prm); if(ni.gt.max_in_mem) deallocate(prma)
	endif
	return
	end subroutine merge_sort_key_real8
!-------------------------------------------
	subroutine merge_sort_cmplx8(ni,trn)
!This subroutine sorts an array of NI items in a non-descending order.
!The algorithm was suggested by Johann von Neumann.
!COMPLEX(8) comparison (non-standard): (x1,y1)>(x2,y2) iff [[x1>x2].OR.[x1=x2.AND.y1>y2]]
!INPUT:
! - ni - number of items;
! - trn(1:ni) - items (arbitrary complex*8 numbers);
!OUTPUT:
! - trn(0:ni) - sorted items and sign in trn(0).
	implicit none
	integer, intent(in):: ni
	complex(8), intent(inout):: trn(0:ni)
	integer i,j,k,l,m,n,k0,k1,k2,k3,k4,k5,k6,k7,ks,kf,ierr
	real(8), parameter:: ds(0:1)=(/+1d0,-1d0/)
	complex(8), allocatable:: prm(:)
	real(8) sgn

	sgn=+1d0
	if(ni.gt.1) then
	 allocate(prm(1:ni))
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
	     if(dble(trn(k1))-dble(trn(k3)).gt.0d0.or.&
	        &(dble(trn(k1)).eq.dble(trn(k3)).and.dimag(trn(k1))-dimag(trn(k3)).gt.0d0)) then
	      prm(i+l)=trn(k3); k3=k3+1; sgn=ds(mod(k2-k1,2))*sgn
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
	 deallocate(prm)
	endif
	trn(0)=dcmplx(sgn,0d0)
	return
	end subroutine merge_sort_cmplx8
!---------------------------------------------------
	subroutine merge_sort_key_cmplx8(ni,key,trn)
!This subroutine sorts an array of NI items in a non-descending order according to their keys.
!The algorithm is due to Johann von Neumann.
!COMPLEX(8) comparison (non-standard): (x1,y1)>(x2,y2) iff [[x1>x2].OR.[x1=x2.AND.y1>y2]]
!INPUT:
! - ni - number of items;
! - key(1:ni) - item keys (retrieved by old item numbers!): arbitrary complex*8 numbers;
! - trn(0:ni) - initial permutation of ni items (a sequence of old numbers), trn(0) is the initial sign;
!OUTPUT:
! - trn(0:ni) - sorted permutation of ni items (according to their keys), the sign is in trn(0).
	implicit none
	integer, intent(in):: ni
	complex(8), intent(in):: key(1:ni)
	integer, intent(inout):: trn(0:ni)
	integer, parameter:: max_in_mem=1024
	integer i,j,k,l,m,n,k0,k1,k2,k3,k4,k5,k6,k7,ks,kf,ierr
	integer, target:: prms(1:max_in_mem)
	integer, allocatable,target:: prma(:)
	integer, pointer:: prm(:)

	if(ni.gt.1) then
	 if(ni.le.max_in_mem) then; prm=>prms; else; allocate(prma(1:ni)); prm=>prma; endif
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
	     if(dble(key(trn(k1)))-dble(key(trn(k3))).gt.0d0.or.&
	        &(dble(key(trn(k1))).eq.dble(key(trn(k3))).and.dimag(key(trn(k1)))-dimag(key(trn(k3))).gt.0d0)) then
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
	 nullify(prm); if(ni.gt.max_in_mem) deallocate(prma)
	endif
	return
	end subroutine merge_sort_key_cmplx8

       end module combinatoric
