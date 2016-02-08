!This module provides infrastructure for symmetric multi-indexing
!for higher rank tensor algebra.
!AUTHOR: Dmitry I. Lyakh (Liakh): quant4me@gmail.com
!REVISION: 2015/09/23
!DESCRIPTION:
! # Given a symmetric multi-index {I1<=I2<=...<=In}, each subsequent value
!   of the multi-index is consecutively assigned an integer from the range [0..max]
!   via the special addressing table. The lower bound of I1 is always 0.
!   The lower bound of Ik is no less than the lower bound of I(k-1).
!   The upper bound of Ik is no less than the upper bound of I(k-1).
! # For ascending ordered multi-indices, the index numeration is normalized
!   by shifting each index by the lower bound of the first (smallest) index,
!   the latter having the lower bound of zero afterwards.
!   For unordered multi-indices, the index numeration is normalized by shifting
!   each index by its own lower bound, thus having each index start from zero.
!   Only the normalized index values can be used with addressing tables!
!FUNCTIONALITY:
! # v:clean_address_tables();
! # v:info_address_tables(i:num_tables,i:num_elems,i:tables_left);
! # i:find_address_table(i:handle,i:ndim,i:ord,i:mrpt,i[1]:lb,i[1]:ub);
! # i:get_address_table(i:handle,i_p[2]:iba,i_o:ndim,i_o:ord,i_o:mrpt,i_o[1]:lbnd,i_o[1]:ubnd);
! # i:delete_address_table(i:handle);
! # i:test_address_table(i[2]:iba,i:ndim,i:ord,i:mrpt,i[1]:lb,i[1]:ub);
       module symm_index
        implicit none
        private
!PARAMETERS:
        integer, private:: CONS_OUT=6
        logical, private:: VERBOSE=.true.
        logical, private:: DEBUG=.true.
        integer, parameter, private:: MAX_MLNDX_LENGTH=256                      !max allowed multi-index length
        integer, parameter, private:: MAX_BANKS=16                              !max number of table banks
        integer, parameter, private:: TABLES_PER_BANK=16384                     !max number of tables per bank
        integer, parameter, private:: MAX_ADDR_TABLES=MAX_BANKS*TABLES_PER_BANK !max number of addressing tables
!ALIASES:
        integer, parameter, public:: SYMM_INDEX_EMPTY=-1
        integer, parameter, public:: SYMM_INDEX_NO_ORDER=0
        integer, parameter, public:: SYMM_INDEX_LE_ORDER=1
        integer, parameter, public:: SYMM_INDEX_GE_ORDER=2
!TYPES:
 !Address table:
        type, private:: address_table_t
         integer, private:: ordering=SYMM_INDEX_EMPTY      !type of ordering in a multi-index
         integer, private:: max_repeats                    !max allowed number of indices having the same value in an ordered multi-index
         integer, allocatable, private:: lbounds(:)        !lower index bounds in a multi-index
         integer, allocatable, private:: ubounds(:)        !upper index bounds in a multi-index
         integer, allocatable, private:: incr(:,:)         !table of addressing increments (addressing table)
        end type address_table_t
 !Bank of addressing tables:
        type, private:: tables_bank_t
         integer, private:: num_tables=0                           !number of live tables in the bank
         integer, allocatable, private:: free_tab(:)               !free table handles
         type(address_table_t), allocatable, private:: addr_tab(:) !array of addressing tables
        end type tables_bank_t
 !Addressing tables container:
        type, private:: address_tables_t
         integer, private:: banks_in_use=0                    !number of table banks in use
         integer, private:: tables_in_use=0                   !total number of tables in use
         integer, private:: total_tables_size=0               !total number of elements in all allocated addressing tables
         type(tables_bank_t), private:: tab_bank(1:MAX_BANKS) !table banks
        end type address_tables_t
!DATA:
        type(address_tables_t), target, private:: address_tables !addressing tables storage
!FUNCTIONS:
        public clean_address_tables
        public info_address_tables
        public get_address_table
        public delete_address_table
        public test_address_table

       contains
!----------------------------------------
        subroutine clean_address_tables()
!This subroutine cleans (destroys) the addressing table service.
!NOTES:
! # According to the Fortran standard, deallocation of a derived type
!   with allocated components must deallocate the latter as well. This is assumed here.
        implicit none
        integer i
        do i=1,MAX_BANKS
         if(allocated(address_tables%tab_bank(i)%free_tab)) deallocate(address_tables%tab_bank(i)%free_tab)
         if(allocated(address_tables%tab_bank(i)%addr_tab)) deallocate(address_tables%tab_bank(i)%addr_tab) !hierarchical deallocation!
         address_tables%tab_bank(i)%num_tables=0
        enddo
        address_tables%banks_in_use=0; address_tables%tables_in_use=0; address_tables%total_tables_size=0
        return
        end subroutine clean_address_tables
!-----------------------------------------------------------------------
        subroutine info_address_tables(num_tables,num_elems,tables_left)
!This subroutine queries the current state of the addressing table service.
        implicit none
        integer, intent(out):: num_tables  !number of addressing tables in use
        integer, intent(out):: num_elems   !total number of table elements in use
        integer, intent(out):: tables_left !number of free tables
        num_tables=address_tables%tables_in_use
        num_elems=address_tables%total_tables_size
        tables_left=MAX_ADDR_TABLES-max(num_tables,0)
        return
        end subroutine info_address_tables
!----------------------------------------------------------------------
        integer function find_address_table(handle,ndim,ord,mrpt,lb,ub)
!This function looks up a necessary address table in the local storage.
        implicit none
        integer, intent(out):: handle
        integer, intent(in):: ndim
        integer, intent(in):: ord
        integer, intent(in):: mrpt
        integer, intent(in):: lb(1:*)
        integer, intent(in):: ub(1:*)
        integer i,j,k,l,m,n

        find_address_table=0; handle=0
        
        return
        end function find_address_table
!-----------------------------------------------------------------------------
        integer function get_address_table(handle,iba,ndim,ord,mrpt,lbnd,ubnd)
!This function either returns an existing addressing table (by handle) or
!creates a new addressing table and returns it together with its handle.
        implicit none
        integer, intent(inout):: handle            !addressing table handle (<=0: empty, new table will be created)
        integer, pointer, intent(out):: iba(:,:)   !addressing table
        integer, intent(in), optional:: ndim       !length of the multi-index
        integer, intent(in), optional:: ord        !requested index ordering
        integer, intent(in), optional:: mrpt       !max allowed number of indices having the same value in the ordered multi-index
        integer, intent(in), optional:: lbnd(1:*)  !lower bounds of index ranges
        integer, intent(in), optional:: ubnd(1:*)  !upper bounds of index ranges
        integer i,j,k,l,m,n,bank,tab,top_val,ierr
        integer lb(1:MAX_MLNDX_LENGTH),ub(1:MAX_MLNDX_LENGTH),im(0:MAX_MLNDX_LENGTH+1)

        get_address_table=0
        if(handle.gt.0.and.handle.le.MAX_ADDR_TABLES) then !handle --> existing table
         bank=((handle-1)/TABLES_PER_BANK)+1 !bank number
         if(allocated(address_tables%tab_bank(bank)%addr_tab)) then
          i=handle-(bank-1)*TABLES_PER_BANK !entry number
          if(allocated(address_tables%tab_bank(bank)%addr_tab(i)%incr)) then
           iba(0:,1:)=>address_tables%tab_bank(bank)%addr_tab(i)%incr
          else
           get_address_table=-1
          endif
         else
          get_address_table=-2
         endif
        elseif(handle.le.0) then !empty handle: create a new table
         if(present(ndim).and.present(ord).and.present(mrpt).and.present(lbnd).and.present(ubnd)) then
          if(ndim.gt.0.and.ndim.le.MAX_MLNDX_LENGTH.and.mrpt.ge.0) then
           do i=1,ndim; if(lbnd(i).gt.ubnd(i)) then; get_address_table=-3; return; endif; enddo !check
           select case(ord)
           case(SYMM_INDEX_NO_ORDER)
            do i=1,ndim; lb(i)=0; enddo; do i=1,ndim; ub(i)=ubnd(i)-lbnd(i); enddo !normalize
            top_val=0; do i=1,ndim; top_val=max(top_val,ub(i)); enddo !get the max index range
            ierr=get_free_table(bank,tab,handle); if(ierr.ne.0) then; get_address_table=-6; return; endif
            iba(0:,1:)=>address_tables%tab_bank(bank)%addr_tab(tab)%incr
            l=1
            do m=1,ndim
             do i=0,ub(m); iba(i,m)=i*l; enddo
             l=l*(ub(m)+1)
            enddo
           case(SYMM_INDEX_LE_ORDER)
            do i=1,ndim; lb(i)=lbnd(i)-lbnd(1); enddo; do i=1,ndim; ub(i)=ubnd(i)-lbnd(1); enddo !normalize
            do i=1,ndim-1; if(lb(i).gt.lb(i+1)) then; get_address_table=-4; return; endif; enddo !check
            do i=1,ndim-1; if(ub(i).gt.ub(i+1)) then; get_address_table=-5; return; endif; enddo !check
            top_val=0; do i=1,ndim; top_val=max(top_val,ub(i)); enddo !get the max index range
            ierr=get_free_table(bank,tab,handle); if(ierr.ne.0) then; get_address_table=-6; return; endif
            iba(0:,1:)=>address_tables%tab_bank(bank)%addr_tab(tab)%incr
 !1st (minor) position:
            im(0)=lb(1)-1; im(ndim+1)=ub(ndim)+1
            do i=lb(1),ub(1); iba(i,1)=i; enddo !lb(1)=0
 !Subsequent positions:
            do m=2,ndim !index position
  !Get the minimal multi-index of length m:
             n=1
             do i=1,m
              im(i)=max(lb(i),im(i-1))
              if(im(i).eq.im(i-1)) then
               if(n.lt.mrpt) then; n=n+1; else; n=1; im(i)=im(i)+1; if(im(i).gt.ub(i)) then; n=-1; exit; endif; endif
              endif
             enddo
             if(n.lt.0) then; get_address_table=1; return; endif !multi-index range is empty under these restrictions
             if(DEBUG) then
              write(CONS_OUT,'("#DEBUG(symm_index::get_address_table): min: ",64(1x,i4))') im(1:m)
             endif
  !Compute increments for position m:
             l=0
             do while(im(m).le.ub(m)) !index value
              iba(im(m),m)=l
   !Construct the max multi-index of length (m-1):
              n=1
              do j=m-1,1,-1
               im(j)=min(ub(j),im(j+1)); if(im(j).eq.im(j+1)) n=n+1
               if(n.gt.mrpt) then; n=1; im(j)=im(j)-1; endif
              enddo
   !Get the offset for the next index value:
              do j=1,m-1; l=l+iba(im(j),j); enddo; l=l+1
              im(m)=im(m)+1
             enddo
            enddo
           case(SYMM_INDEX_GE_ORDER)
            !`Write
           case default !unknown ordering requested
            get_address_table=-7
           end select
          else !invalid arguments
           get_address_table=-8
          endif
         else !optional arguments missing
          get_address_table=-9
         endif
        else !invalid handle value
         get_address_table=-10
        endif
        return

        contains

         integer function get_free_table(jb,jt,hndl)
         implicit none
         integer, intent(out):: jb,jt,hndl
         integer:: j0,je
         get_free_table=0
         if(address_tables%tables_in_use.lt.MAX_ADDR_TABLES) then
          jb=0; jt=0; hndl=0
          do j0=1,MAX_BANKS
           if(allocated(address_tables%tab_bank(j0)%addr_tab)) then
            if(address_tables%tab_bank(j0)%num_tables.lt.TABLES_PER_BANK) then
             address_tables%tab_bank(j0)%num_tables=address_tables%tab_bank(j0)%num_tables+1
             jt=address_tables%tab_bank(j0)%free_tab(address_tables%tab_bank(j0)%num_tables); jb=j0
             hndl=jt+(jb-1)*TABLES_PER_BANK
             exit
            endif
           else
            allocate(address_tables%tab_bank(j0)%addr_tab(1:TABLES_PER_BANK),&
                     &address_tables%tab_bank(j0)%free_tab(1:TABLES_PER_BANK),STAT=je)
            if(je.ne.0) then; get_free_table=-1; return; endif
            do je=1,TABLES_PER_BANK; address_tables%tab_bank(j0)%free_tab(je)=je; enddo
            address_tables%tab_bank(j0)%num_tables=1; jb=j0; jt=1; hndl=jt+(jb-1)*TABLES_PER_BANK
            address_tables%banks_in_use=address_tables%banks_in_use+1
            exit
           endif
          enddo
          if(jb.gt.0.and.jt.gt.0) then
           allocate(address_tables%tab_bank(jb)%addr_tab(jt)%lbounds(1:ndim),&
                    &address_tables%tab_bank(jb)%addr_tab(jt)%ubounds(1:ndim),&
                    &address_tables%tab_bank(jb)%addr_tab(jt)%incr(0:top_val,1:ndim),STAT=je)
           if(je.eq.0) then
            address_tables%tab_bank(jb)%addr_tab(jt)%ordering=ord
            address_tables%tab_bank(jb)%addr_tab(jt)%max_repeats=mrpt
            address_tables%tab_bank(jb)%addr_tab(jt)%lbounds(1:ndim)=lb(1:ndim)
            address_tables%tab_bank(jb)%addr_tab(jt)%ubounds(1:ndim)=ub(1:ndim)
            address_tables%tables_in_use=address_tables%tables_in_use+1
            address_tables%total_tables_size=address_tables%total_tables_size+(top_val+1)*ndim
           else
            if(allocated(address_tables%tab_bank(jb)%addr_tab(jt)%lbounds))&
             &deallocate(address_tables%tab_bank(jb)%addr_tab(jt)%lbounds)
            if(allocated(address_tables%tab_bank(jb)%addr_tab(jt)%ubounds))&
             &deallocate(address_tables%tab_bank(jb)%addr_tab(jt)%ubounds)
            if(allocated(address_tables%tab_bank(jb)%addr_tab(jt)%incr))&
             &deallocate(address_tables%tab_bank(jb)%addr_tab(jt)%incr)
            jb=0; jt=0; hndl=0; get_free_table=-2
           endif
          else
           get_free_table=1
          endif
         else !no free tables left
          get_free_table=2
         endif
         return
         end function get_free_table

        end function get_address_table
!----------------------------------------------------
        integer function delete_address_table(handle)
!Given a valid addressing table handle, this function destroys that addressing table.
        implicit none
        integer, intent(inout):: handle
        integer i,n
        delete_address_table=0
        if(handle.gt.0.and.handle.le.MAX_ADDR_TABLES) then
         n=((handle-1)/TABLES_PER_BANK)+1; i=handle-(n-1)*TABLES_PER_BANK
         if(allocated(address_tables%tab_bank(n)%addr_tab).and.&
           &allocated(address_tables%tab_bank(n)%free_tab)) then
          if(allocated(address_tables%tab_bank(n)%addr_tab(i)%lbounds))&
            &deallocate(address_tables%tab_bank(n)%addr_tab(i)%lbounds)
          if(allocated(address_tables%tab_bank(n)%addr_tab(i)%ubounds))&
            &deallocate(address_tables%tab_bank(n)%addr_tab(i)%ubounds)
          if(allocated(address_tables%tab_bank(n)%addr_tab(i)%incr)) then
           address_tables%total_tables_size=address_tables%total_tables_size-size(address_tables%tab_bank(n)%addr_tab(i)%incr)
           deallocate(address_tables%tab_bank(n)%addr_tab(i)%incr)
          endif
          address_tables%tab_bank(n)%addr_tab(i)%ordering=SYMM_INDEX_EMPTY
          address_tables%tab_bank(n)%addr_tab(i)%max_repeats=-1
          address_tables%tables_in_use=address_tables%tables_in_use-1
          address_tables%tab_bank(n)%free_tab(address_tables%tab_bank(n)%num_tables)=i
          address_tables%tab_bank(n)%num_tables=address_tables%tab_bank(n)%num_tables-1
          if(address_tables%tab_bank(n)%num_tables.eq.0) then
           deallocate(address_tables%tab_bank(n)%addr_tab,address_tables%tab_bank(n)%free_tab)
           address_tables%banks_in_use=address_tables%banks_in_use-1
          endif
          handle=0
         else
          delete_address_table=-1
         endif
        else
         delete_address_table=-2
        endif
        return
        end function delete_address_table
!-------------------------------------------------------------------
        integer function test_address_table(iba,ndim,ord,mrpt,lb,ub)
!This function tests addressing tables.
        implicit none
        integer, intent(in):: iba(0:,1:) !addressing table (increments)
        integer, intent(in):: ndim       !length of the multi-index
        integer, intent(in):: ord        !multi-index ordering
        integer, intent(in):: mrpt       !max allowed number of index repeats in an ordered multi-index
        integer, intent(in):: lb(1:*)    !index lower bounds
        integer, intent(in):: ub(1:*)    !index upper bounds
        integer i,j,k,l,m,n,im(0:MAX_MLNDX_LENGTH+1),ir(1:MAX_MLNDX_LENGTH)

        test_address_table=0
        if(ndim.gt.0.and.ndim.le.MAX_MLNDX_LENGTH.and.mrpt.ge.0) then
         if(lbound(iba,1).eq.0.and.lbound(iba,2).eq.1.and.lb(1).eq.0) then
          select case(ord)
          case(SYMM_INDEX_NO_ORDER)
           !`Write
          case(SYMM_INDEX_LE_ORDER)
 !Initialize the minimal multi-index:
           im(0)=lb(1)-1; im(ndim+1)=ub(ndim)+1; n=1
           do i=1,ndim
            im(i)=max(lb(i),im(i-1))
            if(im(i).eq.im(i-1)) then
             if(n.lt.mrpt) then; n=n+1; else; n=1; im(i)=im(i)+1; if(im(i).gt.ub(i)) then; n=-1; exit; endif; endif
            endif
           enddo
           ir(ndim)=1; do i=ndim-1,1,-1; if(im(i).eq.im(i+1)) then; ir(i)=ir(i+1)+1; else; ir(i)=1; endif; enddo
 !Test all multi-indices:
           if(n.ge.0) then
            l=-1; k=0; do i=1,ndim; k=k+iba(im(i),i); enddo
            if(DEBUG) then
             write(CONS_OUT,'("#DEBUG(symm_index::test_address_table): ndim, mrpt: ",i4,1x,i4)') ndim,mrpt
             write(CONS_OUT,'("#DEBUG(symm_index::test_address_table): min: ",i10,":",64(1x,i4))') k,im(1:ndim)
             write(CONS_OUT,'("#DEBUG(symm_index::test_address_table): min: ",i10,":",64(1x,i4))') l,ir(1:ndim)
            endif
            tloop: do
             l=l+1
             if(DEBUG) then
              write(CONS_OUT,'("#DEBUG(symm_index::test_address_table):",i10,":",64(1x,i4))') k,im(1:ndim)
!             write(CONS_OUT,'("#DEBUG(symm_index::test_address_table):",i10,":",64(1x,i4))') l,ir(1:ndim)
             endif
             if(k.ne.l) then
              if(VERBOSE) write(CONS_OUT,'("#ERROR(symm_index::test_address_table): mismatch: ",i11,1x,i11)') l,k
              test_address_table=1; return !addressing table is invalid
             endif
             iloop: do i=1,ndim
              k=k-iba(im(i),i)
              if(im(i).lt.min(ub(i),im(i+1))) then
               im(i)=im(i)+1
               if(im(i).eq.im(i+1)) then
                ir(i)=ir(i+1)+1; if(ir(i).gt.mrpt) cycle iloop
               else
                ir(i)=1
               endif
               k=k+iba(im(i),i)
               n=1
               do j=1,i-1
                im(j)=max(lb(j),im(j-1))
                if(im(j).eq.im(j-1)) then; if(n.lt.mrpt) then; n=n+1; else; n=1; im(j)=im(j)+1; endif; endif
               enddo
               do j=i-1,1,-1
                if(im(j).eq.im(j+1)) then; ir(j)=ir(j+1)+1; else; ir(j)=1; endif
                k=k+iba(im(j),j)
               enddo
               cycle tloop
              endif
             enddo iloop
             exit tloop
            enddo tloop
           else
            if(VERBOSE) write(CONS_OUT,'("#ERROR(symm_index::test_address_table): No minimal multi-index exists!")')
            test_address_table=2
           endif
          case(SYMM_INDEX_GE_ORDER)
           !`Write
          case default
           test_address_table=3
          end select
         else
          if(VERBOSE) write(CONS_OUT,'("#ERROR(symm_index::test_address_table): invalid bounds: ",i6,1x,i6,1x,i6)')&
           &lbound(iba,1),lbound(iba,2),lb(1)
          test_address_table=4
         endif
        else
         test_address_table=5
        endif
        return
        end function test_address_table

       end module symm_index
