!Standard procedures often used by me.
!AUTHOR: Dmitry I. Lyakh (Liakh): quant4me@gmail.com
!REVISON: 2018/10/03

!Copyright (C) 2005-2017 Dmitry I. Lyakh (Liakh)

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

	module stsubs
	use, intrinsic:: ISO_C_BINDING
	implicit none
	private
!Parameters:
	logical, private:: VERBOSE=.false.                 !verbosity for errors
	real(8), parameter, public:: PI=3.14159265358979d0 !PI constant
	real(8), parameter, public:: BOHR=0.529177249d0    !Bohrs to Angstroms conversion factor
!Procedures:
	public:: alphanumeric!checks if the character is alphanumeric
	public:: alphanumeric_string !checks if the string is alphanumeric_
	public:: array2string!converts a character*1 array into a string
	public:: byte_chksum !returns the raw byte check sum for a given memory segment
	public:: cap_ascii   !makes all English letters capital
	public:: charnum     !converts a number given as a string to real*8 and integer numbers
	public:: clone_object!clones a Fortran object
	public:: crash       !causes a crash
	public:: create_line !creates a table line in .txt format with ; separator
	public:: dumb_work   !performs a dumb work on one or two arrays producing a third one
	public:: dump_bytes  !dumps byte values for a given memory segment
	public:: icharnum    !converts a number given as a string to the integer number
	public:: ifcl        !calculates factorial
	public:: is_it_letter!checks if the character is an ASCII letter
	public:: is_it_number!checks if the character is an ASCII number
	public:: itrsign     !determines a sign of a given transposition
	public:: longnumchar !converts a long integer number to the character representation
	public:: markchf     !counts how many non-blank fields a string contains
	public:: matmat      !multiplies matrix on matrix
	public:: matmatt     !multiplies matrix on transposed matrix
	public:: mattmat     !multiplies transposed matrix on matrix
	public:: mattmatt    !multiplies transposed matrix on transposed matrix
	public:: name_hash   !returns a hash-mask for a given string
	public:: nocomment   !removes comments from a line (!,#)
	public:: normv       !normalizes a vector
	public:: nospace     !removes blanks from a string
	public:: not_a_number!checks whether a given string solely contains a decimal number
	public:: numchar     !converts an integer number to character representation
	public:: printl      !prints a line of characters
	public:: rand_bool   !returns a random logical
	public:: rand_int4   !returns a random integer4 from the given range
	public:: rand_real8  !returns a random real8
	public:: rand_str    !returns a random string
	public:: rots        !rotates an array of points in a 3d-space
	public:: size_of     !scalar data type size in bytes (Fortran 2008)
	public:: small_ascii !makes all English letters small
	public:: str_cmp     !compares two strings
	public:: string2array!converts a string into a character*1 array
	public:: strsearch   !searches a given fragment in a string.
	public:: symbol_type !function which tells you whether the given symbol is a space/tab (0), number (1), letter (2), or other (-1)
	public:: tpause      !pause for a given number of seconds
	public:: valchar     !converts a real number to the string of characters
	public:: wait_delay  !pause for a given time
	public:: wait_press  !subroutine makes pause until user presses a key
	public:: wr_mat_in   !writes a matrix of integers to the screen
	public:: wr_mat_in8  !writes a matrix of integer8's to the screen
	public:: wr_mat_sp   !writes a matrix of single precision elements to the screen
	public:: wr_mat_dp   !writes a matrix of double precision elements to the screen
	public:: wr_mat_dc   !writes a matrix of double complex elements to the screen
	public:: wr_vec_sp   !writes a vector of single precision elements to the screen
	public:: wr_vec_dp   !writes a vector of double precision elements to the screen

	contains
!----------------------------------------
	logical function alphanumeric(ch)
!Returns TRUE if the character is ASCII alphanumeric, FALSE otherwise.
!A space is not considered alphanumeric.
	 implicit none
	 character(1), intent(in):: ch
	 alphanumeric=.FALSE.
	 if(is_it_letter(ch).gt.0) then
	  alphanumeric=.TRUE.
	 else
	  if(is_it_number(ch).ge.0) alphanumeric=.TRUE.
	 endif
	 return
	end function alphanumeric
!------------------------------------------------
	logical function alphanumeric_string(str)
!Returns TRUE if the string only contains ASCII alphanumeric + underscore,
!FALSE otherwise. Spaces are not considered alphanumeric_.
!An empty string is not considered alphanumeric_ either.
	 implicit none
	 character(*), intent(in):: str
	 integer:: i,l
	 alphanumeric_string=.TRUE.
	 l=len(str)
	 if(l.gt.0) then
	  do i=1,l
	   if(.not.(alphanumeric(str(i:i)).or.str(i:i).eq.'_')) then
	    alphanumeric_string=.FALSE.
	    exit
	   endif
	  enddo
	 else
	  alphanumeric_string=.FALSE.
	 endif
	 return
	end function alphanumeric_string
!------------------------------------------------
	subroutine array2string(str,ar1,arl,ierr)
!Converts a CHARACTER(1) array into a Fortran string.
	implicit none
	character(*), intent(out):: str
	integer, intent(in):: arl
	character(1), intent(in):: ar1(1:arl)
	integer i,ierr
	ierr=0
	if(arl.le.len(str)) then
	 do i=1,arl; str(i:i)=ar1(i); enddo
	else
	 ierr=1
	endif
	return
	end subroutine array2string
!----------------------------------------------------
        function byte_chksum(cptr,csize) result(csum)
!Returns the raw byte check sum for a given memory segment.
         implicit none
         integer(8):: csum              !out: check sum
         type(C_PTR), intent(in):: cptr !in: base C pointer
         integer, intent(in):: csize    !in: size in bytes
         integer(1), pointer:: iptr(:)
         integer:: i

         csum=0_8
         call c_f_pointer(cptr,iptr,shape=(/csize/))
         do i=1,csize
          if(iptr(i).ge.0) then
           csum=csum+int(iptr(i),8)
          else
           csum=csum+(256_8+int(iptr(i),8))
          endif
         enddo
         return
        end function byte_chksum
!--------------------------------
	subroutine cap_ascii(str)
!Capitalizes all small English letters in string <str>
	implicit none
	character(*), intent(inout):: str
	integer i,k,l,ia1,ia2,ia3

	ia1=iachar('a')
	ia2=iachar('z')
	ia3=iachar('A')
	l=len_trim(str)
	do k=1,l
	 i=iachar(str(k:k))
	 if(i.ge.ia1.and.i.le.ia2) str(k:k)=achar(ia3+(i-ia1))
	enddo
	return
	end subroutine cap_ascii
!---------------------------------
	subroutine charnum(A,DN,N)
!Converts a number given as a string (<A>) to real and integer forms.
!If <A> is not a number, then it will set {int(DN)=-2, N=0} (not a number tag).
!A real number in <A> can use both "." and "," for the decimal point designation.
!If <A> contains a number at the beginning separated by a blank/tab from the
!following part of the string, only this number will be converted while
!the rest of the string will be ignored (if ALLOW_END_GARBAGE set to TRUE).
!Note that <N> is only guaranteed to be equal to the original (string) number
!when that number is an integer written either as an integer or as a scientific
!number with a bare (D|d|E|e) or with a trvial (D0|d0|E0|e0). In all other cases,
!<N> is not required (or cannot) represent the original (string) number.
	implicit none
	character(*), intent(in):: A !in: string containing a symbolic number (integer or real)
	integer, intent(out):: N     !out: integer number (rounded off for reals)
	real(8), intent(out):: DN    !out: real number
	logical, parameter:: ALLOW_END_GARBAGE=.true.
	real(8):: DSN,DP,DPS
	integer:: I,J,K,K1,K2,L,M,KF,LA,IA0,IA9,ISN,ADP,ierr

        ierr=0; N=0; DN=0d0
	LA=len_trim(A)
        if(LA.gt.0) then
!Check the sign of the number:
         IA0=iachar('0'); IA9=iachar('9')
	 DSN=+1d0; ISN=+1 !default sign is positive
	 DP=0d0; DPS=+1d0 !power (for scientific D and E formats)
	 M=1; KF=0; ADP=0
	 K1=1; do while(K1.lt.LA.and.(A(K1:K1).eq.' '.or.iachar(A(K1:K1)).eq.9)); K1=K1+1; enddo !skip initial blanks
	 if(A(K1:K1).eq.'+') then !positive number
	  if(K1.lt.LA) then; K1=K1+1; else; ierr=1; endif
	 elseif(A(K1:K1).eq.'-') then !negative number
	  if(K1.lt.LA) then; DSN=-1d0; ISN=-1; K1=K1+1; else; ierr=2; endif
	 endif
	else
	 ierr=3
	endif
!Process the number itself:
        if(ierr.eq.0) then
	 cloop: do K=K1,LA
	  I=iachar(A(K:K))
	  if(IA0.le.I.and.I.le.IA9) then !decimal digit
	   KF=max(KF,1) !decimal digit found
	   if(M.eq.1) then !integer part
	    N=N*10+(I-IA0)
	    DN=DN*10d0+real(I-IA0,8)
	   else !fractional part
	    ADP=ADP-1
	    DN=DN+real(I-IA0,8)*(10d0**real(ADP,8))
	    if(I.gt.IA0) KF=2 !non-zero fractional part
	   endif
	  elseif(A(K:K).EQ.'.'.or.A(K:K).EQ.',') then !decimal dot/comma detected
	   if(M.eq.1) then; M=M+1; else; ierr=4; exit cloop; endif
	  elseif(A(K:K).EQ.'D'.or.A(K:K).EQ.'d'.or.A(K:K).EQ.'E'.or.A(K:K).EQ.'e') then !power symbol
	   if(KF.ne.0) then
	    L=K+1
	    if(L.gt.LA) exit cloop !bare D means D0 by default (the same for d,E,e)
	    J=iachar(A(L:L))
	    if(A(L:L).eq.'+') then
	     if(L.lt.LA) then; L=L+1; else; ierr=5; exit cloop; endif
	    elseif(A(L:L).eq.'-') then
	     if(L.lt.LA) then; DPS=-1d0; L=L+1; else; ierr=6; exit cloop; endif
	    elseif(J.lt.IA0.or.J.gt.IA9) then
	     ierr=7; exit cloop !not a number case
	    endif
	    do K2=L,LA
	     J=iachar(A(K2:K2))
	     if(J.ge.IA0.and.J.le.IA9) then
	      if(J.gt.IA0) KF=2 !non-zero power value
	      DP=DP*10d0+real(J-IA0,8)
	     else
	      ierr=8; exit cloop !not a number case
	     endif
	    enddo
	    DP=DPS*DP
	    exit cloop
	   else
	    ierr=9; exit cloop
	   endif
	  elseif(A(K:K).eq.' '.or.I.eq.9) then !blank/tab
	   if(KF.eq.0.or.(.not.ALLOW_END_GARBAGE)) ierr=10
	   exit cloop
	  else !unexpected character
	   ierr=11; exit cloop !not a number case
	  endif
	 enddo cloop !next character A(K:K)
	endif
!Result:
	if(ierr.eq.0.and.KF.ne.0) then !number succesfully translated
	 DN=DSN*(DN*(10d0**DP))
	 if(KF.gt.1) then
	  N=int(DN)
	 else
	  N=ISN*N
	 endif
	else !not a number case
	 DN=-2d0; N=0 !{DN=-2d0 & N=0}: not a number case
	endif
	return
	end subroutine charnum
!-------------------------------------------------------
        function clone_object(object,ierr) result(clone)
!Clones a Fortran object: Deep copy for allocatable components,
!pointer association for pointer components.
         implicit none
         class(*), pointer:: clone             !out: clone
         class(*), intent(in), target:: object !in: object
         integer, intent(out), optional:: ierr !out: error code
         character(256):: errmesg
         integer:: errc

         clone=>NULL(); errc=0
         allocate(clone,SOURCE=object,STAT=errc,ERRMSG=errmesg)
         if(errc.ne.0) then
          write(*,*)'#ERROR(stsubs:clone_object): Fortran sourced allocate() failed: '//errmesg
          if(errmesg(1:39).eq.'Attempt to allocate an allocated object') then !debug
           write(*,*)'If you see status = F below, you are likely experiencing a gfortran bug! Switch to GCC/8.0.0+'
           write(*,*)'Object (pointer) association status = ',associated(clone)
          endif
          clone=>NULL()
         endif
         if(present(ierr)) ierr=errc
         return
        end function clone_object
!-------------------------
        subroutine crash()
         integer:: ax,dx
         write(*,'("Initiating a crash ...")')
         ax=1; dx=ax/0; write(*,*) dx
         return
        end subroutine crash
!------------------------------------------------------------------
	subroutine create_line(talen,tabln,mantl,ctrls,numar,symar)
!The subroutine creates a table line in .txt format with some separator.
!INPUT:
! 1) mantl - number of mantissa digits;
! 2) ctrls - control sequence of commands;
! 3) symar - symbolic array of data (pool of symbolic entries);
! 4) numar - real(8) array of data (pool of numeric entries);
!OUTPT:
! 1) tabln - table line (string of characters);
! 2) talen - length of <tabln>.
	implicit none
!-----------------------------------------------------------
	character(1), parameter:: sep=';' !default separator
!-----------------------------------------------------------
	integer i,j,k,l,m,n,k1,k2,k3,k4,kf
	integer, intent(in):: mantl
	integer, intent(out):: talen
	character(*), intent(out):: tabln
	integer, intent(in):: ctrls(*)
	real(8), intent(in):: numar(*)
	character(*), intent(in):: symar(*)
	character(256):: os

	talen=0
	l=1 !instruction pointer
	do
	 select case(ctrls(l))
	 case(0) !end of control sequence
	  exit
	 case(1) !insert a number of separators
	  do k=1,ctrls(l+1)
	   talen=talen+1
	   tabln(talen:talen)=sep
	  enddo
	  l=l+2
	 case(2) !insert symbolic entries separated by separators
	  do k=ctrls(l+1),ctrls(l+2)
	   m=len_trim(symar(k))
	   tabln(talen+1:talen+m+1)=symar(k)(1:m)//sep
	   talen=talen+m+1
	  enddo
	  l=l+3
	 case(3) !insert numeric entries separated by separators
	  do k=ctrls(l+1),ctrls(l+2)
	   call valchar(numar(k),mantl,m,os)
	   tabln(talen+1:talen+m+1)=os(1:m)//sep
	   talen=talen+m+1
	  enddo
	  l=l+3
	 case default
	  write(*,*)'#ERROR(STSUBS::CREATE_LINE): invalid control command: ',ctrls(l),l
	  stop
	 end select
	enddo
	return
	end subroutine create_line
!-------------------------------------------
        subroutine dumb_work(arr0,arr1,arr2)
!Performs a dumb work on one or two arrays, producing a third one.
        implicit none
        real(8), intent(inout), contiguous:: arr0(0:)
        real(8), intent(in), contiguous:: arr1(0:)
        real(8), intent(in), contiguous, optional:: arr2(0:)
        integer:: vol0,vol1,vol2,i,j,k
        real(8):: val,ax

        vol0=size(arr0); vol1=size(arr1)
        if(vol0.le.0.or.vol1.le.0) return
        if(present(arr2)) then
         vol2=size(arr2)
!$OMP PARALLEL DO DEFAULT(SHARED) SCHEDULE(GUIDED) PRIVATE(i,j,k,val,ax)
         do i=0,vol0-1
          ax=0d0
          do j=mod(i,vol1),vol1-1
           val=arr1(j)
           do k=mod(i,vol2),vol2-1
            ax=ax+arr2(k)*val
           enddo
          enddo
          arr0(i)=ax
         enddo
!$OMP END PARALLEL DO
        else
!$OMP PARALLEL DO DEFAULT(SHARED) SCHEDULE(GUIDED) PRIVATE(i,j)
         do i=0,vol0-1
          arr0(i)=0d0
          do j=mod(i,vol1),vol1-1
           arr0(i)=arr0(i)+arr1(j)*arr1(j)
          enddo
         enddo
!$OMP END PARALLEL DO
        endif
        return
        end subroutine dumb_work
!--------------------------------------------------
        subroutine dump_bytes(cptr,csize,file_name)
!Dumps byte values for a given memory segment.
         implicit none
         type(C_PTR), intent(in):: cptr !in: base C pointer
         integer, intent(in):: csize    !in: size in bytes
         character(*), intent(in), optional:: file_name !in: output file name (defaults to screen)
         integer(1), pointer:: iptr(:)
         integer:: i,devo

         if(present(file_name)) then
          devo=666
          open(devo,file=file_name,FORM='FORMATTED',STATUS='UNKNOWN')
         else
          devo=6
         endif
         write(devo,'()'); write(devo,*) '### Memory dump for address ' !,cptr
         call c_f_pointer(cptr,iptr,shape=(/csize/))
         do i=1,csize
          if(iptr(i).ge.0) then
           write(devo,'("Offset ",i10,": ",i4)') i-1,int(iptr(i),4)
          else
           write(devo,'("Offset ",i10,": ",i4)') i-1,(256+int(iptr(i),4))
          endif
         enddo
         if(devo.ne.6) close(devo)
         return
        end subroutine dump_bytes
!--------------------------------------
	integer function icharnum(L,OS)
!Converts an integer number OS(1:L) given as a string into INTEGER.
!Resets <L> to 0 in case of error.
	 implicit none
	 integer, intent(inout):: L    !inout: length of the string (resets to ZERO in case of error!)
	 character(*), intent(in):: OS !in: OS(1:L): positive or negative number as a string
	 integer:: K,M,N,IA0

	 IA0=iachar('0')
	 if(L.gt.0) then
	  if(OS(1:1).eq.'-') then      !negative
	   M=-2
	  elseif(OS(1:1).eq.'+') then  !explicitly positive
	   M=2
	  else                         !implicitly positive
	   M=1
	  endif
	  icharnum=0
	  do K=abs(M),L
	   N=iachar(OS(K:K))-IA0
	   if(N.ge.0.and.N.le.9) then
	    icharnum=icharnum*10 + N
	   else
	    if(VERBOSE) write(*,*)'#ERROR(STSUBS::ICHARNUM): invalid character number: '//OS(1:L)
	    L=0 !error
	    return
	   endif
	  enddo
	  if(M.eq.-2) icharnum=-icharnum
	 else
	  if(VERBOSE) write(*,*)'#ERROR(STSUBS::ICHARNUM): string of non-positive length: ',L
	  L=0 !error
	 endif
	 return
	end function icharnum
!-------------------------------
	integer function ifcl(N)
!Returns the factorial of a non-negative N.
	 implicit none
	 integer N,K
	 ifcl=1
	  do K=2,N
	   ifcl=ifcl*K
	  enddo
	 return
	end function ifcl
!----------------------------------------
	integer function is_it_letter(ch)
!If character <ch> is a letter, returns 1 for lower-case, 2 for upper-case,
!otherwise returns 0.
	 implicit none
	 character(1), intent(in):: ch
	 integer:: i
	 i=iachar(ch)
	 if(i.ge.iachar('a').and.i.le.iachar('z')) then
	  is_it_letter=1
	 elseif(i.ge.iachar('A').and.i.le.iachar('Z')) then
	  is_it_letter=2
	 else
	  is_it_letter=0
	 endif
	 return
	end function is_it_letter
!----------------------------------------
	integer function is_it_number(ch)
!If character <ch> is a number, returns its value as an integer,
!otherwise a negative integer is returned.
	 implicit none
	 character(1), intent(in):: ch
	 integer:: i
	 i=iachar(ch)
	 if(i.ge.iachar('0').and.i.le.iachar('9')) then
	  is_it_number=i-iachar('0')
	 else
	  is_it_number=-1
	 endif
	 return
	end function is_it_number
!--------------------------------
	subroutine itrsign(N,ITR)
!Reorders a given permutation into an ascending order and returns the permutation sign in ITR(0).
!Bubble algorithm (only for small permutations).
	 implicit none
	 integer, intent(inout):: ITR(0:*) !inout: ITR(1:N) is the permutation, ITR(0) is its initial/final sign
	 integer, intent(in):: N           !in: length of the permutation
	 integer:: K,L
	 K=1
	 do while(K.lt.N)
	  if(ITR(K).gt.ITR(K+1)) then
	   L=ITR(K); ITR(K)=ITR(K+1); ITR(K+1)=L
	   ITR(0)=-ITR(0)
	   if(K.gt.1) then
	    K=K-1
	   else
	    K=2
	   endif
	  else
	   K=K+1
	  endif
	 enddo
	 return !RETURNS an ascending sequence of indices with a sign in ITR(0)
	end subroutine itrsign
!----------------------------------------
	subroutine longnumchar(I,IOSL,OS)
!Converts a long integer into a string.
	implicit none
	integer(8), intent(in):: I       !in: long integer to be converted into a string
	integer, intent(out):: IOSL      !out: string length
	character(*), intent(inout):: OS !out: OS(1:IOSL) is the string
	integer(8) K
	integer L,K1,K2
	character(1) A(0:9),CH
	data A/'0','1','2','3','4','5','6','7','8','9'/

	if(I.lt.0) then
	 OS(1:1)='-'; K=-I; IOSL=1
	elseif(I.eq.0) then
	 OS(1:1)='0'; IOSL=1
	 return
	else
	 K=I; IOSL=0
	endif
	L=IOSL
	do while(K.NE.0)
	 K1=mod(K,10_8); K=K/10_8
	 L=L+1; OS(L:L)=A(K1)
	enddo
	K1=L-IOSL; if(mod(K1,2).eq.1) K1=K1-1; K1=K1/2
	do K2=1,K1
	 CH=OS(IOSL+K2:IOSL+K2)
	 OS(IOSL+K2:IOSL+K2)=OS(L+1-K2:L+1-K2)
	 OS(L+1-K2:L+1-K2)=CH
	enddo
	IOSL=L
	return !OS(1:IOSL) - the number as a string (the sign included if the number is negative)
	end subroutine longnumchar
!-----------------------------------
	subroutine markchf(STR,N,MF)
!Counts the number of fields in the string <STR> (separated by spaces) and returns
!the positions of all fields found (a field is a sequence of characters without blanks).
	 implicit none
	 character(*), intent(in):: STR   !in: string
	 integer, intent(out):: N         !out: N is the number of fields in the string
	 integer, intent(out):: MF(2,1:*) !out: MF(1:2,X) is the beginning and end positions of the field #X(1<=X<=N)
	 integer:: K,LSTR

	 LSTR=len_trim(STR)
	 N=0; K=1
	 do while(K.le.LSTR)
	  if(STR(K:K).ne.' '.and.iachar(STR(K:K)).ne.9) then !skip TABS also (ASCII code 9)
	   N=N+1; MF(1:2,N)=(/K,0/)
	   do while(K.le.LSTR)
	    if(STR(K:K).eq.' '.or.iachar(STR(K:K)).eq.9) exit
	    K=K+1
	   enddo
	   MF(2,N)=K-1
	  else
	   K=K+1
	  endif
	 enddo
	 return
	end subroutine markchf
!---------------------------------------
	subroutine matmat(sa,sb,m,a,b,c)
!Matrix-matrix multiplication: C(sa,sb)=A(sa,m)*B(m,sb)+C(sa,sb)
	 implicit none
	 integer, intent(in):: sa,sb,m
	 real(8), intent(in):: a(1:sa,1:m),b(1:m,1:sb)
	 real(8), intent(inout):: c(1:sa,1:sb)
	 integer j,k

	 do j=1,sb
	  do k=1,m
	   c(1:sa,j)=c(1:sa,j)+a(1:sa,k)*b(k,j)
	  enddo
	 enddo
	 return
	end subroutine matmat
!----------------------------------------
	subroutine matmatt(sa,sb,m,a,b,c)
!Matrix-matrix multiplication (transposed 2nd argument): C(sa,sb)=A(sa,m)*B(sb,m)+C(sa,sb)
	 implicit none
	 integer, intent(in):: sa,sb,m
	 real(8), intent(in):: a(1:sa,1:m),b(1:sb,1:m)
	 real(8), intent(inout):: c(1:sa,1:sb)
	 integer j,k

	 do j=1,sb
	  do k=1,m
	   c(1:sa,j)=c(1:sa,j)+a(1:sa,k)*b(j,k)
	  enddo
	 enddo
	 return
	end subroutine matmatt
!----------------------------------------
	subroutine mattmat(sa,sb,m,a,b,c)
!Matrix-matrix multiplication (transposed 1st argument): C(sa,sb)=A(m,sa)*B(m,sb)+C(sa,sb)
	 implicit none
	 integer, intent(in):: sa,sb,m
	 real(8), intent(in):: a(1:m,1:sa),b(1:m,1:sb)
         real(8), intent(inout):: c(1:sa,1:sb)
	 integer i,j,k
	 real(8) val

	 do j=1,sb
	  do k=1,sa
	   val=c(k,j)
	   do i=1,m; val=val+a(i,k)*b(i,j); enddo !unroll this loop
	   c(k,j)=val
	  enddo
	 enddo
	 return
	end subroutine mattmat
!-----------------------------------------
	subroutine mattmatt(sa,sb,m,a,b,c)
!Matrix-matrix multiplication (both arguments are transposed): C(sa,sb)=A(m,sa)*B(sb,m)+C(sa,sb)
	 implicit none
	 integer, intent(in):: sa,sb,m
	 real(8), intent(in):: a(1:m,1:sa),b(1:sb,1:m)
	 real(8), intent(inout):: c(1:sa,1:sb)
	 integer j,k

	 do j=1,sb
	  do k=1,m
	   c(1:sa,j)=c(1:sa,j)+a(k,1:sa)*b(j,k)
	  enddo
	 enddo
	 return
	end subroutine mattmatt
!--------------------------------------------------
        integer function name_hash(max_hash,sl,str)
!This function returns a hash mask (0..max_hash) for a given string.
!INPUT:
! - max_hash - limit of the hash range (parameter);
! - str(1:sl) - string;
!OUTPUT:
! - name_hash - hash mask [0..max_hash].
        implicit none
        integer, intent(in):: max_hash,sl
        character(*), intent(in):: str
        integer:: i

        name_hash=0
        do i=1,min(sl,len(str))
         name_hash=name_hash+iachar(str(i:i))
        enddo
        name_hash=mod(name_hash,max_hash+1)
        return
        end function name_hash
!----------------------------------
	subroutine nocomment(l,str)
!This subroutine removes comments (!,#) from the line given in str(1:l).
!After that l is a modified length.
	implicit none
	integer, intent(inout):: l
	character(*), intent(in):: str
	integer:: i

	do i=1,min(l,len(str))
	 if(str(i:i).eq.'!'.or.str(i:i).eq.'#') then
	  l=i-1
	  exit
	 endif
	enddo
	return
	end subroutine nocomment
!-------------------------------
	subroutine normv(n,vn,v)
!Substitutes the original vector <v> with its normalized form.
	 implicit none
	 integer, intent(in):: n         !in: dimension of the vector
	 real(8), intent(out):: vn       !out: vector length
	 real(8), intent(inout):: v(1:n) !inout: initial vector will be replaced by the normalized one
	 integer:: i
	 real(8):: val

	 vn=0d0
	 do i=1,n
	  vn=vn+v(i)*v(i)
	 enddo
	 if(vn.ne.0d0) then
	  vn=dsqrt(vn); val=1d0/vn
	  v(1:n)=v(1:n)*val
	 endif
	 return
	end subroutine normv
!------------------------------
	subroutine nospace(A,L)
!Removes spaces/tabs from a string.
	 implicit none
	 character(*), intent(inout):: A !inout: string --> reduced string
	 integer, intent(out):: L        !out: the length of the string with all spaces/tabs removed
	 integer:: J,M

	 M=len_trim(A); L=0
	 do J=1,M
	  if(A(J:J).ne.' '.and.iachar(A(J:J)).ne.9) then
	   L=L+1; A(L:L)=A(J:J)
	  endif
	 enddo
	 return
	end subroutine nospace
!-----------------------------------------
	logical function not_a_number(str)
!Returns TRUE if string <str> is a number, false otherwise.
!Signed integer, decimal, and scientific decimal formats are supported.
!The decimal point is ".". For scientific decimals, the exponent can be either of {D|d|E|e}.
	 implicit none
	 character(*), intent(in):: str
	 integer:: dot_present,exp_present,i,l,n
	 l=len_trim(str)
	 if(l.gt.0) then
	  not_a_number=.false.; dot_present=0; exp_present=0; n=0
	  if(str(1:1).eq.'-'.or.str(1:1).eq.'+') then; i=2; else; i=1; endif
	  do while(i.le.l)
	   if(iachar(str(i:i)).lt.iachar('0').or.iachar(str(i:i)).gt.iachar('9')) then
	    if(str(i:i).eq.'.') then !decimal dot
	     if(dot_present.eq.0.and.exp_present.eq.0) then; dot_present=1; else; not_a_number=.true.; return; endif
	    elseif(str(i:i).eq.'D'.or.str(i:i).eq.'d'.or.str(i:i).eq.'E'.or.str(i:i).eq.'e') then
	     if(exp_present.eq.0.and.n.gt.0) then
	      exp_present=1
	      if(i.lt.l) then; if(str(i+1:i+1).eq.'-'.or.str(i+1:i+1).eq.'+') i=i+1; else; not_a_number=.true.; return; endif
	     else
	      not_a_number=.true.; return
	     endif
	    else
	     not_a_number=.true.; return
	    endif
	   else !decimal digit
	    n=n+1
	   endif
	   i=i+1
	  enddo
	  if(n.le.0) not_a_number=.true.
	 else !empty string is not a number
	  not_a_number=.true.
	 endif
	 return
	end function not_a_number
!------------------------------------
	subroutine numchar(I,IOSL,OS)
!Converts an integer into a string.
	implicit none
	integer, intent(in):: I          !in: integer to be converted into a string
	integer, intent(out):: IOSL      !out: length of the string
	character(*), intent(inout):: OS !out: OS(1:IOSL) is the string
	integer K,K1,L
	character(1) A(0:9),CH
	data A/'0','1','2','3','4','5','6','7','8','9'/

	if(I.lt.0) then
	 OS(1:1)='-'; K=-I; IOSL=1
	elseif(I.eq.0) then
	 OS(1:1)='0'; IOSL=1
	 return
	else
	 K=I; IOSL=0
	endif
	L=IOSL
	do while(K.ne.0)
	 K1=mod(K,10); K=K/10
	 L=L+1; OS(L:L)=A(K1)
	enddo
	K1=L-IOSL; if(mod(K1,2).eq.1) K1=K1-1; K1=K1/2
	do K=1,K1
	 CH=OS(IOSL+K:IOSL+K)
	 OS(IOSL+K:IOSL+K)=OS(L+1-K:L+1-K)
	 OS(L+1-K:L+1-K)=CH
	enddo
	IOSL=L
	return !OS(1:IOSL) - the number as a string (the sign included if the number is negative)
	end subroutine numchar
!------------------------------------
	subroutine printl(FH,STR,ADV)
!Prints a string <str> as a single line into file #FH.
	implicit none
	integer, intent(in):: FH            !in: file handle
	character(*), intent(in):: STR      !in: string to print
	logical, intent(in), optional:: ADV !in: carriage return control (.false. - no carriage return)
	character(32):: FRM
	integer:: l,k

	l=len_trim(STR)
	if(l.gt.0) then
	 FRM(1:2)='(A'
	 call NUMCHAR(l,k,FRM(3:32))
	 k=3+k; FRM(k:k)=')'
	 if(present(ADV)) then
	  if(ADV) then
	   write(FH,FRM(1:k)) STR(1:l)
	  else
	   write(FH,FRM(1:k),advance='no') STR(1:l)
	  endif
	 else
	  write(FH,FRM(1:k)) STR(1:l)
	 endif
	else
	 if(present(ADV)) then
	  if(ADV) write(FH,'()')
	 endif
	endif
	return
	end subroutine printl
!----------------------------------------
        function rand_bool() result(bool)
!Returns a random logical.
         implicit none
         logical:: bool !out: random logical
         real(8):: rn

         bool=.FALSE.
         call random_number(rn)
         if(rn.ge.5d-1) bool=.TRUE.
         return
        end function rand_bool
!---------------------------------------------------
        function rand_int4(lower,upper) result(int4)
!Returns a random integer4 from the given range.
         implicit none
         integer(4):: int4              !out: random number
         integer(4), intent(in):: lower !in: lower bound
         integer(4), intent(in):: upper !in: upper bound

         int4=nint(rand_real8(real(lower,8),real(upper,8)),4)
         return
        end function rand_int4
!-----------------------------------------------------
        function rand_real8(lower,upper) result(real8)
!Returns a random real8 from the given range.
         implicit none
         real(8):: real8             !out: random number
         real(8), intent(in):: lower !in: lower bound
         real(8), intent(in):: upper !in: upper bound

         call random_number(real8)
         real8=lower+real8*(upper-lower)
         return
        end function rand_real8
!----------------------------------
	subroutine rand_str(sl,str)
!Generates a random string.
	implicit none
	integer, intent(out):: sl         !out: string length
	character(*), intent(inout):: str !out: str(1:sl) is the random string
	integer, parameter:: max_str_len=8192
	integer, parameter:: buf_size=512
	integer, parameter:: ascii_b=32,ascii_e=126,ascii_range=ascii_e-ascii_b+1
	real(8) rnds(1:buf_size),rn
	integer i,j,k,m,n

	if(len(str).gt.0) then
	 k=min(len(str),max_str_len)
	 call random_number(rn); sl=int(rn*real(k,8))+1; if(sl.gt.k) sl=k
	 do i=0,sl-1,buf_size
	  n=min(sl-i,buf_size)
	  call random_number(rnds(1:n))
	  do j=1,n
	   m=int(rnds(j)*real(ascii_range,8))+ascii_b; if(m.gt.ascii_e) m=ascii_e
	   str(i+j:i+j)=achar(m)
	  enddo
	 enddo
	else
	 sl=0
	endif
	return
	end subroutine rand_str
!-----------------------------------------
	subroutine rots(axis,alpha,n,pnts)
!This subroutine rotates <n> points in array <pnts>
!around the axis #axis in a 3d-space.
	implicit none
	integer i,j,k,l,m,n,k1,k2,k3,k4,ks,kf
	integer axis
	real(8) alpha,pnts(3,*),q(3),cosa,sina

	sina=sin(alpha)
	cosa=cos(alpha)
	select case(axis)
	case(1)
!X axis:
	 do k=1,n
	  q(1:3)=pnts(1:3,k)
	  pnts(1,k)=q(1)
	  pnts(2,k)=q(2)*cosa-q(3)*sina
	  pnts(3,k)=q(2)*sina+q(3)*cosa
	 enddo
	case(2)
!Y axis:
	 do k=1,n
	  q(1:3)=pnts(1:3,k)
	  pnts(1,k)=q(1)*cosa-q(3)*sina
	  pnts(2,k)=q(2)
	  pnts(3,k)=q(1)*sina+q(3)*cosa
	 enddo
	case(3)
!Z axis:
	 do k=1,n
	  q(1:3)=pnts(1:3,k)
	  pnts(1,k)=q(1)*cosa-q(2)*sina
	  pnts(2,k)=q(1)*sina+q(2)*cosa
	  pnts(3,k)=q(3)
	 enddo
	case default
	 write(*,*)'#ERROR(STSUBS::ROTS): invalid axis number: ',axis
	 stop
	end select
	return
	end subroutine rots
!-------------------------------------
        integer function size_of(uarg) !Fortran 2008
!Returns size in bytes for an arbitrary scalar type (-1 means an error).
        implicit none
        class(*), intent(in), target:: uarg !in: scalar argument of any type/class

        size_of=storage_size(uarg) !in bits
        if(mod(size_of,8).eq.0) then
         size_of=size_of/8 !in bytes
        else
         size_of=-1
        endif
        return
        end function size_of
!----------------------------------
	subroutine small_ascii(STR)
!Lowercases all capital Enlgish letters in string <STR>
	implicit none
	character(*), intent(inout):: STR
	integer:: i,j,k,l,m,n,ia1,ia2,ia3

	ia1=iachar('A')
	ia2=iachar('Z')
	ia3=iachar('a')
	l=len_trim(STR)
	do k=1,l
	 i=iachar(STR(k:k))
	 if(i.ge.ia1.and.i.le.ia2) STR(k:k)=achar(ia3+(i-ia1))
	enddo
	return
	end subroutine small_ascii
!------------------------------------------
        integer function str_cmp(str1,str2)
!Compares two strings:
! -1: str1 < str2;
! +1: str1 > str2;
!  0: str1=str2
         implicit none
         character(*), intent(in):: str1
         character(*), intent(in):: str2
         integer:: l1,l2,i,a1,a2

         l1=len(str1); l2=len(str2)
         if(l1.lt.l2) then
          str_cmp=-1
         elseif(l1.gt.l2) then
          str_cmp=+1
         else
          str_cmp=0
          do i=1,l1
           a1=iachar(str1(i:i)); a2=iachar(str2(i:i))
           if(a1.lt.a2) then
            str_cmp=-1; exit
           elseif(a1.gt.a2) then
            str_cmp=+1; exit
           endif
          enddo
         endif
         return
        end function str_cmp
!------------------------------------------------
	subroutine string2array(str,ar1,arl,ierr)
!Converts a character string into a CHARACTER(1) array.
	implicit none
	character(*), intent(in):: str        !in: character string
	character(1), intent(inout):: ar1(1:) !out: CHARACTER(1) array
	integer, intent(out):: arl            !out: array volume (number of elements)
	integer, intent(inout):: ierr         !out: error code (0:success)
	integer:: i

	ierr=0; arl=len(str)
	if(size(ar1).ge.arl) then
	 do i=1,arl; ar1(i)=str(i:i); enddo
	else
	 ierr=1
	endif
	return
	end subroutine string2array
!--------------------------------------
	subroutine strsearch(L,STR,FRG)
!Searches exactly the first <FRG> fragment in string <STR>.
	 implicit none
	 integer, intent(inout):: L     !inout: initially the length of STR. If <FRG> found, L = position of FRG in STR, otherwise L=-1
	 character(*), intent(in):: STR !in: character string
	 character(*), intent(in):: FRG !in: character substring to be looked for inside the string
	 character(1):: A
	 integer:: LFRG,LSTR,K,M

	 LFRG=len(FRG) !spaces are taken into account
	 if(LFRG.le.0.or.L.le.0) then; L=-1; return; endif
	 LSTR=L; M=1; A=FRG(M:M)
	 do K=1,LSTR
	  if(STR(K:K).eq.A) then
	   if(M.eq.LFRG) then
	    L=K+1-LFRG !the 1st position of the 1st FRG fragment in STR
	    return
	   else
	    M=M+1; A=FRG(M:M)
	   endif
	  else
	   M=1; A=FRG(M:M)
	  endif
	 enddo
	 L=-1 !FRG has not been found in STR
	 return
	end subroutine strsearch
!---------------------------------------
	integer function symbol_type(ch)
!Returns the type of the CHARACTER symbol <ch>:
!  0: space or tab;
!  1: decimal digit;
!  2: English letter;
! -1: otherwise.
	 implicit none
	 character(1), intent(in):: ch
	 integer:: i

	 i=iachar(ch)
	 if(i.eq.9.or.i.eq.32) then !tab or space
	  symbol_type=0
	 elseif(i.ge.iachar('0').and.i.le.iachar('9')) then !number
	  symbol_type=1
	 elseif((i.ge.iachar('a').and.i.le.iachar('z')).or.(i.ge.iachar('A').and.i.le.iachar('Z'))) then
	  symbol_type=2
	 else !other
	  symbol_type=-1
	 endif
	 return
	end function symbol_type
!-------------------------------
	subroutine tpause(ISEC)
!Halts for <ISEC> seconds.
	 implicit none
	 integer, intent(in):: ISEC
	 real(8) TIM1,TIM2

         if(ISEC.gt.0) then
	  call cpu_time(TIM1); TIM1=TIM1+real(ISEC,8)
	  call cpu_time(TIM2)
	  do while(TIM2.lt.TIM1)
	   call cpu_time(TIM2)
	  enddo
	 endif
	 return
	end subroutine tpause
!-----------------------------------------
	subroutine valchar(val,mantl,l,os)
!Prints a decimal number into a string.
	 implicit none
	 real(8), intent(in):: val        !in: decimal number
	 integer, intent(in):: mantl      !in: number of decimal digits after the dot to print
	 integer, intent(out):: l         !out: string length
	 character(*), intent(inout):: os !out: string os(1:l)
	 character(1) A(0:9),ch
	 integer k,m,ls
	 real(8) d
	 data A/'0','1','2','3','4','5','6','7','8','9'/

	 if(val.ge.0d0) then
	  d=val; l=0
	 else
	  d=-val; os(1:1)='-'; l=1
	 endif
	 ls=l+1
	 k=int(d)
	 if(k.gt.0) then
	  do while(k.gt.0)
	   m=mod(k,10); k=k/10
	   l=l+1; os(l:l)=A(m)
	  enddo
	  k=0
	  do while(ls+k.lt.l-k)
	   ch=os(ls+k:ls+k)
	   os(ls+k:ls+k)=os(l-k:l-k)
	   os(l-k:l-k)=ch
	   k=k+1
	  enddo
	  l=l+1; os(l:l)='.'
	 else
	  os(l+1:l+2)='0.'; l=l+2
	 endif
	 d=d-dint(d)
	 k=mantl !max length of the mantissa
	 do while(d.gt.0d0.and.k.gt.0)
	  d=d*10d0; m=mod(int(d),10)
	  l=l+1; os(l:l)=A(m)
	  d=d-dint(d)
	  k=k-1
	 enddo
	 return
	end subroutine valchar
!---------------------------------
	subroutine wait_delay(sec)
!Halts for <sec> seconds.
	implicit none
	real(4), intent(in):: sec
	real(8):: sec0,sec1

        if(sec.gt.0.0) then
	 call cpu_time(sec0); sec0=sec0+real(sec,8)
	 call cpu_time(sec1)
	 do while(sec1.lt.sec0)
	  call cpu_time(sec1)
	 enddo
	endif
	return
	end subroutine wait_delay
!---------------------------------
	subroutine wait_press(msg)
!Halts until the user presses ENTER.
	implicit none
	character(*), intent(in), optional:: msg !in: customized message to print on the screen
	character(1) ch

	if(present(msg)) then
	 call printl(6,msg(1:len_trim(msg)),.false.)
	else
	 write(6,'("Press ENTER to continue ...")',advance='no')
	endif
	read(*,'(A1)') ch
	return
	end subroutine wait_press
!----------------------------------
	subroutine wr_mat_in(m,n,a)
!This subroutine writes the matrix a(m,n) to the screen.
!The elements of matrix are integers.
	 implicit none
	 integer m,n,i,j,a(1:m,1:n)
	 do i=1,m
	  do j=1,n
	   write(*,'((I10,1x))',advance='no') a(i,j)
	  enddo
	  write(*,*)""
	 enddo
	 return
	end subroutine wr_mat_in
!-----------------------------------
	subroutine wr_mat_in8(m,n,a)
!This subroutine writes the matrix a(m,n) to the screen.
!The elements of matrix are integer8.
	 implicit none
	 integer m,n,i,j
	 integer(8) a(1:m,1:n)
	 do i=1,m
	  do j=1,n
	   write(*,'((I20,1x))',advance='no') a(i,j)
	  enddo
	  write(*,*)""
	 enddo
	 return
	end subroutine wr_mat_in8
!----------------------------------
	subroutine wr_mat_sp(m,n,a)
!This subroutine writes the matrix a(m,n) to screen.
!The elements of matrix are of single precision real type.
	 implicit none
	 integer m,n,i,j
	 real(4) a(1:m,1:n)
	 do i=1,m
	  do j=1,n
	   write(*,'((F15.7,1x))',advance='no') a(i,j)
	  enddo
	  write(*,*)""
	 enddo
	 return
	end subroutine wr_mat_sp
!----------------------------------
	subroutine wr_mat_dp(m,n,a)
!This subroutine writes the matrix a(m,n) to screen.
!The elements of matrix are of double precision real type.
	 implicit none
	 integer m,n,i,j
	 real(8) a(1:m,1:n)
	 do i=1,m
	  do j=1,n
	   write(*,'(D22.14,1x)',advance='no') a(i,j)
	  enddo
	  write(*,*)""
	 enddo
	 return
	end subroutine wr_mat_dp
!----------------------------------
	subroutine wr_mat_dc(m,n,a)
!This subroutine writes the matrix a(m,n) to the screen.
!The elements of matrix are of double complex type.
	 implicit none
	 integer m,n,i,j
	 complex(8) a(1:m,1:n)
	 do i=1,m
	  do j=1,n
	   write(*,'(("(",D22.14,",",D22.14,")"))',advance='no') a(i,j)
	  enddo
	  write(*,*)""
	 enddo
	 return
	end subroutine wr_mat_dc
!--------------------------------
	subroutine wr_vec_sp(m,a)
	 implicit none
	 integer m,i
	 real(4) a(1:m)
	 do i=1,m
	  write(*,"(F15.7)") a(i)
	 enddo
	 return
	end subroutine wr_vec_sp
!--------------------------------
	subroutine wr_vec_dp(m,a)
	 implicit none
	 integer m,i
	 real(8) a(1:m)
	 do i=1,m
	  write(*,"(D22.14)") a(i)
	 enddo
	 return
	end subroutine wr_vec_dp

	end module stsubs
