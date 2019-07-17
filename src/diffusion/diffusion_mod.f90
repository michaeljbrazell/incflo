!
!  This module contains the subroutines to compute the three components of
!  the explicit update part of the diffusion term  div(tau_exp) / ro, where
!
!      tau_exp = eta (grad u)^T
!
!  Note that the remaining part of the stress tensor, tau_imp = eta grad u,
!  is accounted for through the implicit solve in the DiffusionEquation class.
!
!
module diffusion_mod

   use amrex_fort_module,     only: rt => amrex_real
   use amrex_mempool_module,  only: amrex_allocate, amrex_deallocate
   use iso_c_binding,         only: c_int

   use constant,              only: zero, half, one, two

   implicit none
   private

   real(rt), parameter :: q4 = one / ( two * two )

   public fill_vel_diff_bc

contains
   !
   ! Set the boundary condition for diffusion solve
   !
   ! MLMG expects the BC type to be the uniform on each domain wall.
   ! Since incflo allows for BC patches on each wall, we first check that
   ! the user-provided BCs are uniform, and then return a single BC type for
   ! each domain wall.
   !
   subroutine set_diff_bc ( bc_lo, bc_hi, domlo, domhi, ng, bct_ilo, bct_ihi, &
        & bct_jlo, bct_jhi, bct_klo, bct_khi)  bind(C)

      use amrex_lo_bctypes_module
      use bc

      ! Array of global BC types
      integer(c_int), intent(  out) :: bc_lo(3), bc_hi(3)

      ! Domain bounds
      integer(c_int), intent(in   ) :: domlo(3), domhi(3), ng

      ! Arrays of point-by-point BC types
      integer(c_int), intent(in   )  ::                                 &
           & bct_ilo(domlo(2)-ng:domhi(2)+ng,domlo(3)-ng:domhi(3)+ng,2), &
           & bct_ihi(domlo(2)-ng:domhi(2)+ng,domlo(3)-ng:domhi(3)+ng,2), &
           & bct_jlo(domlo(1)-ng:domhi(1)+ng,domlo(3)-ng:domhi(3)+ng,2), &
           & bct_jhi(domlo(1)-ng:domhi(1)+ng,domlo(3)-ng:domhi(3)+ng,2), &
           & bct_klo(domlo(1)-ng:domhi(1)+ng,domlo(2)-ng:domhi(2)+ng,2), &
           & bct_khi(domlo(1)-ng:domhi(1)+ng,domlo(2)-ng:domhi(2)+ng,2)

      ! Local variables
      integer(c_int)                :: bc_face

      !
      ! By default, all the BCs are Dirichlet
      !
      bc_lo    = amrex_lo_dirichlet
      bc_hi    = amrex_lo_dirichlet

      !
      ! BC -- X direction
      !
      if ( cyclic_x ) then
         bc_lo(1) = amrex_lo_periodic
         bc_hi(1) = amrex_lo_periodic
      else

         ! X at domlo(1)
         bc_face = get_bc_face(bct_ilo, ng)
         if ( (bc_face == pinf_) .or. (bc_face == pout_) ) then
            bc_lo(1) = amrex_lo_neumann
         end if

         ! X at domhi(1)
         bc_face = get_bc_face(bct_ihi, ng)
         if ( (bc_face == pinf_) .or. (bc_face == pout_) ) then
            bc_hi(1) = amrex_lo_neumann
         end if

      end if

      !
      ! BC -- Y direction
      !
      if ( cyclic_y ) then
         bc_lo(2) = amrex_lo_periodic
         bc_hi(2) = amrex_lo_periodic
      else

         ! Y at domlo(2)
         bc_face = get_bc_face(bct_jlo, ng)
         if ( (bc_face == pinf_) .or. (bc_face == pout_) ) then
            bc_lo(2) = amrex_lo_neumann
         end if

         ! Y at domhi(2)
         bc_face = get_bc_face(bct_jhi, ng)
         if ( (bc_face == pinf_) .or. (bc_face == pout_) ) then
            bc_hi(2) = amrex_lo_neumann
         end if

      end if

      !
      ! BC -- Z direction
      !
      if ( cyclic_z ) then
         bc_lo(3) = amrex_lo_periodic
         bc_hi(3) = amrex_lo_periodic
      else

         ! Z at domlo(3)
         bc_face = get_bc_face(bct_klo, ng)
         if ( (bc_face == pinf_) .or. (bc_face == pout_) ) then
            bc_lo(3) = amrex_lo_neumann
         end if

         ! Z at domhi(3)
         bc_face = get_bc_face(bct_khi, ng)
         if ( (bc_face == pinf_) .or. (bc_face == pout_) ) then
            bc_hi(3) = amrex_lo_neumann
         end if

      end if

   contains

      !
      ! Test whether the BC type is the same everywhere on
      ! the face. If BC is uniform on face, it returns its value
      !
      function get_bc_face (bct_array, nghost) result (bc_face)
         integer(c_int), intent(in   ) :: bct_array(:,:,:)
         integer(c_int), intent(in   ) :: nghost
         integer                       :: bc_face
         integer                       :: is, ie, js, je

         ! Do not consider the edges: they may cause problems
         is = nghost+1
         ie = size(bct_array,1) - nghost
         js = nghost+1
         je = size(bct_array,2) - nghost

         bc_face = bct_array(is,js,1)

         if ( .not. all (bct_array(is:ie,js:je,1) == bc_face) ) then
            stop "BC type must be uniform on each face of the domain"
         end if

      end function get_bc_face

   end subroutine set_diff_bc

!-----------------------------------------------------------------------!
!-----------------------------------------------------------------------!
!-----------------------------------------------------------------------!
!-----------------------------------------------------------------------!

   subroutine fill_vel_diff_bc(vel_in, vinlo, vinhi, vel, lo, hi, domlo, domhi, ng, &
                               bc_ilo_type, bc_ihi_type, &
                               bc_jlo_type, bc_jhi_type, &
                               bc_klo_type, bc_khi_type)

      use bc, only: minf_, nsw_

      integer,  intent(in   ) ::   vinlo(3),   vinhi(3)
      integer,  intent(in   ) ::    lo(3),    hi(3)
      integer,  intent(in   ) :: domlo(3), domhi(3)

      real(rt), intent(in   ) :: vel_in(vinlo(1):vinhi(1),vinlo(2):vinhi(2),vinlo(3):vinhi(3),3)
      real(rt), intent(  out) ::    vel(lo(1)-ng:hi(1)+ng,lo(2)-ng:hi(2)+ng,lo(3)-ng:hi(3)+ng,3)

      ! BC types
      integer(c_int), intent(in   ) :: ng,  &
           & bc_ilo_type(domlo(2)-ng:domhi(2)+ng,domlo(3)-ng:domhi(3)+ng,2), &
           & bc_ihi_type(domlo(2)-ng:domhi(2)+ng,domlo(3)-ng:domhi(3)+ng,2), &
           & bc_jlo_type(domlo(1)-ng:domhi(1)+ng,domlo(3)-ng:domhi(3)+ng,2), &
           & bc_jhi_type(domlo(1)-ng:domhi(1)+ng,domlo(3)-ng:domhi(3)+ng,2), &
           & bc_klo_type(domlo(1)-ng:domhi(1)+ng,domlo(2)-ng:domhi(2)+ng,2), &
           & bc_khi_type(domlo(1)-ng:domhi(1)+ng,domlo(2)-ng:domhi(2)+ng,2)

      integer :: i,j,k,n

      do k = lo(3)-ng, hi(3)+ng
         do j = lo(2)-ng, hi(2)+ng
            do i = lo(1)-ng, hi(1)+ng
               vel(i,j,k,:) = vel_in(i,j,k,:)
            end do
         end do
      end do

      if ( lo(1) == domlo(1) ) then
         i = lo(1)
         do n = 1, 3
            do k = lo(3), hi(3)
               do j = lo(2), hi(2)

                  if ( ( bc_ilo_type(j,k,1) == MINF_) .or. &
                       ( bc_ilo_type(j,k,1) == NSW_ ) ) then

                     vel(:lo(1)-1,j,k,n) = 2.d0*vel_in(lo(1)-1,j,k,n) - vel_in(lo(1),j,k,n)

                  end if
               end do
            end do
         end do
      end if

      if ( hi(1) == domhi(1) ) then

         i = hi(1)

         do n = 1, 3
            do k = lo(3), hi(3)
               do j = lo(2), hi(2)

                  if ( ( bc_ihi_type(j,k,1) == MINF_) .or. &
                       ( bc_ihi_type(j,k,1) == NSW_ ) ) then

                     vel(hi(1)+1:,j,k,n) = 2.d0*vel_in(hi(1)+1,j,k,n) - vel_in(hi(1),j,k,n)

                  end if
               end do
            end do
         end do
      end if

      if ( lo(2) == domlo(2) ) then

         j = lo(2)

         do n = 1, 3
            do k = lo(3), hi(3)
               do i = lo(1)-1, hi(1)+1

                  if ( ( bc_jlo_type(i,k,1) == MINF_) .or. &
                       ( bc_jlo_type(i,k,1) == NSW_ ) ) then

                     vel(i,:lo(2)-1,k,n) = 2.d0*vel_in(i,lo(2)-1,k,n) - vel_in(i,lo(2),k,n)

                  end if
               end do
            end do
         end do
      end if

      if ( hi(2) == domhi(2) ) then

         j = hi(2)

         do n = 1, 3
            do k = lo(3), hi(3)
               do i = lo(1)-1, hi(1)+1

                  if ( ( bc_jhi_type(i,k,1) == MINF_) .or. &
                       ( bc_jhi_type(i,k,1) == NSW_ ) ) then

                     vel(i,hi(2)+1:,k,n) = 2.d0*vel_in(i,hi(2)+1,k,n) - vel_in(i,hi(2),k,n)

                  end if
               end do
            end do
         end do
      end if

      if ( lo(3) == domlo(3) ) then

         k = lo(3)

         do n = 1, 3
            do j = lo(2)-1, hi(2)+1
               do i = lo(1)-1, hi(1)+1

                  if ( ( bc_klo_type(i,j,1) == MINF_) .or. &
                       ( bc_klo_type(i,j,1) == NSW_ ) ) then

                     vel(i,j,:lo(3)-1,n) = 2.d0*vel_in(i,j,lo(3)-1,n) - vel_in(i,j,lo(3),n)

                  end if
               end do
            end do
         end do
      end if

      if ( hi(3) == domhi(3) ) then

         k = hi(3)

         do n = 1, 3
            do j = lo(2)-1, hi(2)+1
               do i = lo(1)-1, hi(1)+1

                  if ( ( bc_khi_type(i,j,1) == MINF_) .or. &
                       ( bc_khi_type(i,j,1) == NSW_ ) ) then

                     vel(i,j,hi(3)+1:,n) = 2.d0*vel_in(i,j,hi(3)+1,n) - vel_in(i,j,hi(3),n)

                  end if
               end do
            end do
         end do
      end if

      ! Revisit these because when we did these before we didn't have the side values filled
      if ( lo(1) == domlo(1) ) then

         i = lo(1)

         do n = 1, 3
            do k = lo(3)-1, hi(3)+1
               do j = lo(2)-1, hi(2)+1

                  if ( ( bc_ilo_type(j,k,1) == MINF_) .or. &
                       ( bc_ilo_type(j,k,1) == NSW_ ) ) then

                     vel(lo(1)-1,j,k,n) = 2.d0*vel_in(lo(1)-1,j,k,n) - vel_in(lo(1),j,k,n)

                  end if
               end do
            end do
         end do
      end if

      ! Revisit these because when we did these before we didn't have the side values filled
      if ( hi(1) == domhi(1) ) then

         i = hi(1)

         do n = 1, 3
            do k = lo(3)-1, hi(3)+1
               do j = lo(2)-1, hi(2)+1

                  if ( ( bc_ihi_type(j,k,1) == MINF_) .or. &
                       ( bc_ihi_type(j,k,1) == NSW_ ) ) then

                     vel(hi(1)+1,j,k,n) = 2.d0*vel_in(hi(1)+1,j,k,n) - vel_in(hi(1),j,k,n)

                  end if
               end do
            end do
         end do
      end if

      ! Revisit these because when we did these before we didn't have the side values filled
      if ( lo(2) == domlo(2) ) then

         j = lo(2)

         do n = 1, 3
            do k = lo(3)-1, hi(3)+1
               do i = lo(1)-1, hi(1)+1

                  if ( ( bc_jlo_type(i,k,1) == MINF_) .or. &
                       ( bc_jlo_type(i,k,1) == NSW_ ) ) then

                     vel(i,lo(2)-1,k,n) = 2.d0*vel_in(i,lo(2)-1,k,n) - vel_in(i,lo(2),k,n)

                  end if
               end do
            end do
         end do
      end if

      ! Revisit these because when we did these before we didn't have the side values filled
      if ( hi(2) == domhi(2) ) then

         j = hi(2)

         do n = 1, 3
            do k = lo(3)-1, hi(3)+1
               do i = lo(1)-1, hi(1)+1

                  if ( ( bc_jhi_type(i,k,1) == MINF_) .or. &
                       ( bc_jhi_type(i,k,1) == NSW_ ) ) then

                     vel(i,hi(2)+1,k,n) = 2.d0*vel_in(i,hi(2)+1,k,n) - vel_in(i,hi(2),k,n)

                  end if
               end do
            end do
         end do
      end if

   end subroutine fill_vel_diff_bc

end module diffusion_mod

