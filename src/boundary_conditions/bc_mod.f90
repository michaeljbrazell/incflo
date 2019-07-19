!vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv!
!                                                                      !
!  Module name: bc                                                     !
!  Purpose: Global variables for specifying boundary conditions.       !
!                                                                      !
!^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^!
module bc

   use amrex_fort_module, only : rt => amrex_real
   use iso_c_binding , only: c_int

   use constant, only: dim_bc, undefined, zero

   ! Type of boundary:
   character(len=16) :: bc_type(dim_bc)

   ! Flags for periodic boundary conditions
   logical :: cyclic_x = .false.
   logical :: cyclic_y = .false.
   logical :: cyclic_z = .false.

   logical :: bc_defined(1:dim_bc) = .false.

   ! Boundary condition location (EB planes)
   real(rt) :: bc_normal(1:dim_bc,1:3) = undefined
   real(rt) :: bc_center(1:dim_bc,1:3) = undefined

   ! Gas phase BC pressure
   real(rt) :: bc_p(1:dim_bc) = undefined

   ! Velocities at a specified boundary
   real(rt) :: bc_u(1:dim_bc) = zero
   real(rt) :: bc_v(1:dim_bc) = zero
   real(rt) :: bc_w(1:dim_bc) = zero

   ! Character variable to determine the flow plane of a flow cell
   character :: bc_plane(dim_bc)

   ! Cell flag definitions
   integer, parameter :: undef_cell =   0 ! undefined
   integer, parameter :: pinf_      =  10 ! pressure inflow cell
   integer, parameter :: pout_      =  11 ! pressure outflow cell
   integer, parameter :: minf_      =  20 ! mass flux inflow cell
   integer, parameter :: nsw_       = 100 ! wall with no-slip b.c.

contains
!vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv!
!                                                                      !
! Subroutines: getters                                                 !
!                                                                      !
! Purpose: Getters for the boundary conditions values                  !
!                                                                      !
!vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv!
  real(rt) function get_bc_u(pID) bind(C)
    integer(c_int), intent(in) :: pID
    get_bc_u = bc_u(pID)
    return
  end function get_bc_u

  real(rt) function get_bc_v(pID) bind(C)
    integer(c_int), intent(in) :: pID
    get_bc_v = bc_v(pID)
    return
  end function get_bc_v

  real(rt) function get_bc_w(pID) bind(C)
    integer(c_int), intent(in) :: pID
    get_bc_w = bc_w(pID)
    return
  end function get_bc_w

  integer(c_int) function get_minf() bind(C)
    get_minf = minf_
    return
  end function get_minf

  integer(c_int) function get_pinf() bind(C)
    get_pinf = pinf_
    return
  end function get_pinf

  integer(c_int) function get_pout() bind(C)
    get_pout = pout_
    return
  end function get_pout

  integer(c_int) function get_nsw() bind(C)
    get_nsw = nsw_
    return
  end function get_nsw

  subroutine get_domain_bc (domain_bc_out) bind(C)
    integer(c_int), intent(out)  :: domain_bc_out(6)
    integer :: bcv
    ! Default is that we reflect particles off domain boundaries if not periodic
    domain_bc_out(1:6) = 1
    if (cyclic_x) domain_bc_out(1:2) = 0
    if (cyclic_y) domain_bc_out(3:4) = 0
    if (cyclic_z) domain_bc_out(5:6) = 0

    do bcv = 1,6
       select case (trim(bc_type(bcv)))
         case ('P_OUTFLOW','PO')
            domain_bc_out(bcv) = 0
       end select
    end do
  end subroutine get_domain_bc

!vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv!
!                                                                      !
! Subroutine: set_cyclic                                               !
!                                                                      !
! Purpose: Function to set cyclic flags.                               !
!vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv!
  subroutine set_cyclic(cyc_x, cyc_y, cyc_z) &
    bind(C, name="mfix_set_cyclic")

    integer, intent(in) :: cyc_x, cyc_y, cyc_z

    cyclic_x = (cyc_x == 1)
    cyclic_y = (cyc_y == 1)
    cyclic_z = (cyc_z == 1)

  end subroutine set_cyclic


!vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv!
!                                                                      !
!  Subroutine: write_out_bc                                            !
!                                                                      !
!  Purpose: Echo user input for BC regions.                            !
!                                                                      !
!^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^!
  subroutine write_out_bc(unit_out, dx, dy, dz, &
    xlength, ylength, zlength, domlo, domhi)

    use constant, only: is_defined, delp

    implicit none

    integer,        intent(in) :: unit_out
    real(rt)  , intent(in) :: dx, dy, dz
    real(rt)  , intent(in) :: xlength, ylength, zlength
    integer(c_int), intent(in) :: domlo(3), domhi(3)

    integer :: bcv, m

    logical :: flow_bc

!-----------------------------------------------


! Boundary Condition Data
    write (unit_out, 1600)
1600  format(//,3x,'7. BOUNDARY CONDITIONS')

    if (cyclic_x .and. abs(delp(1)) > epsilon(0.0d0)) then
       write (unit_out, 1602) 'X', ' with pressure drop'
       write (unit_out, 1603) 'X', delp(1)
    else if (cyclic_x) then
       write (unit_out, 1602) 'X'
    endif
    if (cyclic_y .and. abs(delp(2)) > epsilon(0.0d0)) then
       write (unit_out, 1602) 'Y', ' with pressure drop'
       write (unit_out, 1603) 'Y', delp(2)
    else if (cyclic_y) then
       write (unit_out, 1602) 'Y'
    endif
    if (cyclic_z .and. abs(delp(3)) > epsilon(0.0d0)) then
       write (unit_out, 1602) 'Z', ' with pressure drop'
       write (unit_out, 1603) 'Z', delp(3)
    else if (cyclic_z) then
       write (unit_out, 1602) 'Z'
    endif

 1602 format(/7X,'Cyclic boundary conditions in ',A,' direction',A)
 1603 format( 7X,'Pressure drop (DELP_',A,') = ',G12.5)

    do bcv = 1, dim_bc
       if (bc_defined(bcv)) then

          write (unit_out, 1610) bcv, bc_type(bcv)

1610  format(/7x,'Boundary condition no : ',I4,2/&
              9X,'Type of boundary condition : ',A16)

          select case (trim(bc_type(bcv)))
          case ('MASS_INFLOW','MI')
             write (unit_out,"(9x,'Inlet with specified mass flux')")
             flow_bc = .true.
          case ('MASS_OUTFLOW','MO')
             write (unit_out,"(9x,'Outlet with specified mass flux')")
          case ('P_INFLOW','PI')
             write (unit_out,"(9x,'Inlet with specified gas pressure')")
             flow_bc = .true.
          case ('P_OUTFLOW','PO')
             write (unit_out,"(9x,'Outlet with specified gas pressure')")
             flow_bc = .true.
          case ('NO_SLIP_WALL','NSW')
             write (unit_out,"(9x,'Velocity is zero at wall')")
             flow_bc = .false.
          end select

          if(flow_bc) then
             write (unit_out, "(' ')")
             if(is_defined(bc_p(bcv))) &
               write (unit_out,1641) bc_p(bcv)
             ! FIXME? think massflow and volflow are just for mfix
             !  and not used/needed here
             ! if(is_defined(bc_massflow(bcv))) &
             !   write (unit_out,1648) bc_massflow(bcv)
             ! if(is_defined(bc_volflow(bcv)))  &
             !   write (unit_out,1649) bc_volflow(bcv)
             write (unit_out, 1650) bc_u(bcv)
             write (unit_out, 1651) bc_v(bcv)
             write (unit_out, 1652) bc_w(bcv)

1641  format(9X,' pressure (BC_P) ..................... ',g12.5)
1648  format(9X,' mass flow rate (BC_MassFlow) ........ ',g12.5)
1649  format(9X,' volumetric flow rate (BC_VOLFLOW) ... ',g12.5)
1650  format(9X,'X-component of velocity (BC_U) ...... ',g12.5)
1651  format(9X,'Y-component of velocity (BC_V) ...... ',g12.5)
1652  format(9X,'Z-component of velocity (BC_W) ...... ',g12.5)

          endif
       endif
    enddo

    return
  end subroutine write_out_bc

end module bc

