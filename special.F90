!Auxiliary subroutines (GPLv2)
!Author: Dmitry I. Lyakh
	subroutine get_contr_permutations(lrank,rrank,cptrn,dprm,lprm,rprm,ncd,nlu,nru,ierr) bind(c,name='get_contr_permutations') !SERIAL
!This subroutine returns all tensor index permutations necessary for the tensor
!contraction specified by <cptrn> (implemented via a matrix multiplication).
!INPUT:
! - cptrn(1:lrank+rrank) - digital contraction pattern;
!OUTPUT:
! - dprm(0:drank) - index permutation for the destination tensor (N2O, numeration starts from 1);
! - lprm(0:lrank) - index permutation for the left tensor argument (O2N, numeration starts from 1);
! - rprm(0:rrank) - index permutation for the right tensor argument (O2N, numeration starts from 1);
! - ncd - total number of contracted indices;
! - nlu - number of left uncontracted indices;
! - nru - number of right uncontracted indices;
! - ierr - error code (0:success).
	use, intrinsic:: ISO_C_BINDING
	implicit none
!------------------------------------------------
	logical, parameter:: check_pattern=.true.
!------------------------------------------------
	integer(C_INT), intent(in), value:: lrank,rrank
	integer(C_INT), intent(in):: cptrn(1:*)
	integer(C_INT), intent(out):: dprm(0:*),lprm(0:*),rprm(0:*),ncd,nlu,nru
	integer(C_INT), intent(inout):: ierr
	integer(C_INT) i,j,k,drank,jkey(1:lrank+rrank),jtrn0(0:lrank+rrank),jtrn1(0:lrank+rrank)
	logical pattern_ok

	ierr=0
	if(check_pattern) then; pattern_ok=contr_pattern_ok(); else; pattern_ok=.true.; endif
	if(pattern_ok.and.lrank.ge.0.and.rrank.ge.0) then
 !Destination operand:
	 drank=0; dprm(0)=+1;
	 do i=1,lrank+rrank; if(cptrn(i).gt.0) then; drank=drank+1; dprm(drank)=cptrn(i); endif; enddo
 !Right tensor operand:
	 nru=0; ncd=0; rprm(0)=+1; !numbers of the right uncontracted and contracted dimensions
	 if(rrank.gt.0) then
	  j=0; do i=1,rrank; if(cptrn(lrank+i).lt.0) then; j=j+1; rprm(i)=j; endif; enddo; ncd=j !contracted dimensions
	  do i=1,rrank; if(cptrn(lrank+i).gt.0) then; j=j+1; rprm(i)=j; endif; enddo; nru=j-ncd !uncontracted dimensions
	 endif
 !Left tensor operand:
	 nlu=0; lprm(0)=+1; !number of the left uncontracted dimensions
	 if(lrank.gt.0) then
	  j=0
	  do i=1,lrank
	   if(cptrn(i).lt.0) then; j=j+1; jtrn1(j)=i; jkey(j)=abs(cptrn(i)); endif
	  enddo
	  ncd=j !contracted dimensions
	  jtrn0(0:j)=(/+1,(k,k=1,j)/); if(j.ge.2) call merge_sort_key_int(j,jkey,jtrn0)
	  do i=1,j; k=jtrn0(i); lprm(jtrn1(k))=i; enddo !contracted dimensions of the left operand are aligned to the corresponding dimensions of the right operand
	  do i=1,lrank; if(cptrn(i).gt.0) then; j=j+1; lprm(i)=j; endif; enddo; nlu=j-ncd !uncontracted dimensions
	 endif
	else !invalid lrank or rrank or cptrn(:)
	 ierr=1
	endif
	return

	contains

	 logical function contr_pattern_ok()
	 integer(C_INT) j0,j1,jc,jl
	 contr_pattern_ok=.true.; jl=lrank+rrank
	 if(jl.gt.0) then
	  jkey(1:jl)=0; jc=0
	  do j0=1,jl
	   j1=cptrn(j0)
	   if(j1.lt.0) then !contracted index
	    if(j0.le.lrank) then
	     if(abs(j1).gt.rrank) then; contr_pattern_ok=.false.; return; endif
	     if(cptrn(lrank+abs(j1)).ne.-j0) then; contr_pattern_ok=.false.; return; endif
	    else
	     if(abs(j1).gt.lrank) then; contr_pattern_ok=.false.; return; endif
	     if(cptrn(abs(j1)).ne.-(j0-lrank)) then; contr_pattern_ok=.false.; return; endif
	    endif
	   elseif(j1.gt.0.and.j1.le.jl) then !uncontracted index
	    jc=jc+1
	    if(jkey(j1).eq.0) then
	     jkey(j1)=1
	    else
	     contr_pattern_ok=.false.; return
	    endif
	   else
	    contr_pattern_ok=.false.; return
	   endif
	  enddo
	  do j0=1,jc; if(jkey(j0).ne.1) then; contr_pattern_ok=.false.; return; endif; enddo
	 endif
	 return
	 end function contr_pattern_ok

	end subroutine get_contr_permutations
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
