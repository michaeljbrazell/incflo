!
!
!  This module contains the subroutines to compute the three components
!  of the convection term (U.grad)U
!
!
module ugradu_mod

   use amrex_fort_module, only: ar => amrex_real
   use iso_c_binding ,    only: c_int

   use bc,                only: minf_, nsw_, pinf_, pout_
   use constant,          only: zero, half, one, my_huge

   implicit none
   public upwind
   private

contains

   !#####################################################

   !  MAC VERSION

   !#####################################################
   subroutine compute_ugradu(lo, hi, &
                             ugradu, glo, ghi, &
                             vel, vello, velhi, &
                             u, ulo, uhi, &
                             v, vlo, vhi, &
                             w, wlo, whi, &
                             xslopes, yslopes, zslopes, slo, shi, &
                             domlo, domhi, &
                             bc_ilo_type, bc_ihi_type, &
                             bc_jlo_type, bc_jhi_type, &
                             bc_klo_type, bc_khi_type, dx, ng) bind(C)

      ! Tile bounds
      integer(c_int),  intent(in   ) :: lo(3),  hi(3)

      ! Array Bounds
      integer(c_int),  intent(in   ) :: slo(3), shi(3)
      integer(c_int),  intent(in   ) :: glo(3), ghi(3)
      integer(c_int),  intent(in   ) :: vello(3), velhi(3)
      integer(c_int),  intent(in   ) :: ulo(3), uhi(3)
      integer(c_int),  intent(in   ) :: vlo(3), vhi(3)
      integer(c_int),  intent(in   ) :: wlo(3), whi(3)
      integer(c_int),  intent(in   ) :: domlo(3), domhi(3), ng

      ! Grid
      real(ar),        intent(in   ) :: dx(3)

      ! Velocity Array
      real(ar),        intent(in   ) ::                            &
           & vel(vello(1):velhi(1),vello(2):velhi(2),vello(3):velhi(3),3)    , &
           & xslopes(slo(1):shi(1),slo(2):shi(2),slo(3):shi(3),3), &
           & yslopes(slo(1):shi(1),slo(2):shi(2),slo(3):shi(3),3), &
           & zslopes(slo(1):shi(1),slo(2):shi(2),slo(3):shi(3),3), &
           & u(ulo(1):uhi(1),ulo(2):uhi(2),ulo(3):uhi(3)), &
           & v(vlo(1):vhi(1),vlo(2):vhi(2),vlo(3):vhi(3)), &
           & w(wlo(1):whi(1),wlo(2):whi(2),wlo(3):whi(3))

      real(ar),        intent(  out) ::                           &
           & ugradu(glo(1):ghi(1),glo(2):ghi(2),glo(3):ghi(3),3)

      ! BC types
      integer(c_int), intent(in   ) ::  &
           & bc_ilo_type(domlo(2)-ng:domhi(2)+ng,domlo(3)-ng:domhi(3)+ng,2), &
           & bc_ihi_type(domlo(2)-ng:domhi(2)+ng,domlo(3)-ng:domhi(3)+ng,2), &
           & bc_jlo_type(domlo(1)-ng:domhi(1)+ng,domlo(3)-ng:domhi(3)+ng,2), &
           & bc_jhi_type(domlo(1)-ng:domhi(1)+ng,domlo(3)-ng:domhi(3)+ng,2), &
           & bc_klo_type(domlo(1)-ng:domhi(1)+ng,domlo(2)-ng:domhi(2)+ng,2), &
           & bc_khi_type(domlo(1)-ng:domhi(1)+ng,domlo(2)-ng:domhi(2)+ng,2)

      ! Local variables
      integer(c_int)                 :: i, j, k
      real(ar)                       :: idx, idy, idz
      real(ar)                       :: upls, umns, vpls, vmns, wpls, wmns
      real(ar)                       :: u_e, u_w, u_s, u_n, u_b, u_t
      real(ar)                       :: v_e, v_w, v_s, v_n, v_b, v_t
      real(ar)                       :: w_e, w_w, w_s, w_n, w_b, w_t
      real(ar)                       :: divumac
      integer, parameter             :: bc_list(4) = [MINF_, NSW_, PINF_, POUT_]

      idx = one / dx(1)
      idy = one / dx(2)
      idz = one / dx(3)

      do k = lo(3), hi(3)
         do j = lo(2), hi(2)
            do i = lo(1), hi(1)

               ! ****************************************************
               ! West face
               ! ****************************************************

               ! In the case of MINF, NSW, we are using the prescribed Dirichlet value
               ! In the case of PINF, POUT          we are using the upwind value
               if (i.eq.domlo(1) .and. any(bc_ilo_type(j,k,1) == bc_list ) ) then
                  u_w =  vel(i-1,j,k,1)
                  v_w =  vel(i-1,j,k,2)
                  w_w =  vel(i-1,j,k,3)
               else
                  upls  = vel(i  ,j,k,1) - half * xslopes(i  ,j,k,1)
                  umns  = vel(i-1,j,k,1) + half * xslopes(i-1,j,k,1)
                  vpls  = vel(i  ,j,k,2) - half * xslopes(i  ,j,k,2)
                  vmns  = vel(i-1,j,k,2) + half * xslopes(i-1,j,k,2)
                  wpls  = vel(i  ,j,k,3) - half * xslopes(i  ,j,k,3)
                  wmns  = vel(i-1,j,k,3) + half * xslopes(i-1,j,k,3)

                  u_w   = upwind( umns, upls, u(i,j,k) )
                  v_w   = upwind( vmns, vpls, u(i,j,k) )
                  w_w   = upwind( wmns, wpls, u(i,j,k) )
               endif

               ! ****************************************************
               ! East face
               ! ****************************************************

               ! In the case of MINF, NSW, we are using the prescribed Dirichlet value
               ! In the case of PINF, POUT          we are using the upwind value
               if (i.eq.domhi(1) .and. any(bc_ihi_type(j,k,1) == bc_list ) ) then
                  u_e =  vel(i+1,j,k,1)
                  v_e =  vel(i+1,j,k,2)
                  w_e =  vel(i+1,j,k,3)
               else
                  upls  = vel(i+1,j,k,1) - half * xslopes(i+1,j,k,1)
                  umns  = vel(i  ,j,k,1) + half * xslopes(i  ,j,k,1)
                  vpls  = vel(i+1,j,k,2) - half * xslopes(i+1,j,k,2)
                  vmns  = vel(i  ,j,k,2) + half * xslopes(i  ,j,k,2)
                  wpls  = vel(i+1,j,k,3) - half * xslopes(i+1,j,k,3)
                  wmns  = vel(i  ,j,k,3) + half * xslopes(i  ,j,k,3)

                  u_e   = upwind( umns, upls, u(i+1,j,k) )
                  v_e   = upwind( vmns, vpls, u(i+1,j,k) )
                  w_e   = upwind( wmns, wpls, u(i+1,j,k) )
               endif

               ! ****************************************************
               ! South face
               ! ****************************************************

               ! In the case of MINF, NSW, we are using the prescribed Dirichlet value
               ! In the case of PINF, POUT          we are using the upwind value
               if (j.eq.domlo(2) .and. any(bc_jlo_type(i,k,1) == bc_list ) ) then
                  u_s =  vel(i,j-1,k,1)
                  v_s =  vel(i,j-1,k,2)
                  w_s =  vel(i,j-1,k,3)
               else
                  upls  = vel(i,j  ,k,1) - half * yslopes(i,j  ,k,1)
                  umns  = vel(i,j-1,k,1) + half * yslopes(i,j-1,k,1)
                  vpls  = vel(i,j  ,k,2) - half * yslopes(i,j  ,k,2)
                  vmns  = vel(i,j-1,k,2) + half * yslopes(i,j-1,k,2)
                  wpls  = vel(i,j  ,k,3) - half * yslopes(i,j  ,k,3)
                  wmns  = vel(i,j-1,k,3) + half * yslopes(i,j-1,k,3)

                  v_s   = upwind( vmns, vpls, v(i,j,k) )
                  u_s   = upwind( umns, upls, v(i,j,k) )
                  w_s   = upwind( wmns, wpls, v(i,j,k) )
               endif

               ! ****************************************************
               ! North face
               ! ****************************************************

               ! In the case of MINF, NSW, we are using the prescribed Dirichlet value
               ! In the case of PINF, POUT          we are using the upwind value
               if (j.eq.domhi(2) .and.  any(bc_jhi_type(i,k,1) == bc_list ) ) then
                  u_n =  vel(i,j+1,k,1)
                  v_n =  vel(i,j+1,k,2)
                  w_n =  vel(i,j+1,k,3)
               else
                  upls  = vel(i,j+1,k,1) - half * yslopes(i,j+1,k,1)
                  umns  = vel(i,j  ,k,1) + half * yslopes(i,j  ,k,1)
                  vpls  = vel(i,j+1,k,2) - half * yslopes(i,j+1,k,2)
                  vmns  = vel(i,j  ,k,2) + half * yslopes(i,j  ,k,2)
                  wpls  = vel(i,j+1,k,3) - half * yslopes(i,j+1,k,3)
                  wmns  = vel(i,j  ,k,3) + half * yslopes(i,j  ,k,3)

                  v_n   = upwind( vmns, vpls, v(i,j+1,k) )
                  u_n   = upwind( umns, upls, v(i,j+1,k) )
                  w_n   = upwind( wmns, wpls, v(i,j+1,k) )
               endif

               ! ****************************************************
               ! Bottom face
               ! ****************************************************

               ! In the case of MINF, NSW, we are using the prescribed Dirichlet value
               ! In the case of PINF, POUT          we are using the upwind value
               if (k.eq.domlo(3) .and. any( bc_klo_type(i,j,1) == bc_list ) ) then
                  u_b =  vel(i,j,k-1,1)
                  v_b =  vel(i,j,k-1,2)
                  w_b =  vel(i,j,k-1,3)
               else
                  upls  = vel(i,j,k  ,1) - half * zslopes(i,j,k  ,1)
                  umns  = vel(i,j,k-1,1) + half * zslopes(i,j,k-1,1)
                  vpls  = vel(i,j,k  ,2) - half * zslopes(i,j,k  ,2)
                  vmns  = vel(i,j,k-1,2) + half * zslopes(i,j,k-1,2)
                  wpls  = vel(i,j,k  ,3) - half * zslopes(i,j,k  ,3)
                  wmns  = vel(i,j,k-1,3) + half * zslopes(i,j,k-1,3)

                  w_b   = upwind( wmns, wpls, w(i,j,k) )
                  u_b   = upwind( umns, upls, w(i,j,k) )
                  v_b   = upwind( vmns, vpls, w(i,j,k) )
               endif

               ! ****************************************************
               ! Top face
               ! ****************************************************

               ! In the case of MINF, NSW, we are using the prescribed Dirichlet value
               ! In the case of PINF, POUT          we are using the upwind value
               if (k.eq.domhi(3) .and. any( bc_khi_type(i,j,1) == bc_list ) ) then
                  u_t =  vel(i,j,k+1,1)
                  v_t =  vel(i,j,k+1,2)
                  w_t =  vel(i,j,k+1,3)
               else
                  upls  = vel(i,j,k+1,1) - half * zslopes(i,j,k+1,1)
                  umns  = vel(i,j,k  ,1) + half * zslopes(i,j,k  ,1)
                  vpls  = vel(i,j,k+1,2) - half * zslopes(i,j,k+1,2)
                  vmns  = vel(i,j,k  ,2) + half * zslopes(i,j,k  ,2)
                  wpls  = vel(i,j,k+1,3) - half * zslopes(i,j,k+1,3)
                  wmns  = vel(i,j,k  ,3) + half * zslopes(i,j,  k,3)

                  w_t   = upwind( wmns, wpls, w(i,j,k+1) )
                  u_t   = upwind( umns, upls, w(i,j,k+1) )
                  v_t   = upwind( vmns, vpls, w(i,j,k+1) )
               endif

               ! ****************************************************
               ! Define convective terms -- conservatively
               !   ugradu = ( div(u^MAC u^cc) - u^cc div(u^MAC) )
               ! ****************************************************

               divumac = (u(i+1,j,k) - u(i,j,k)) * idx + &
                         (v(i,j+1,k) - v(i,j,k)) * idy + &
                         (w(i,j,k+1) - w(i,j,k)) * idz

               ugradu(i,j,k,1) = (u(i+1,j,k) * u_e - u(i,j,k) * u_w) * idx + &
                                 (v(i,j+1,k) * u_n - v(i,j,k) * u_s) * idy + &
                                 (w(i,j,k+1) * u_t - w(i,j,k) * u_b) * idz - &
                                 vel(i,j,k,1) * divumac
               ugradu(i,j,k,2) = (u(i+1,j,k) * v_e - u(i,j,k) * v_w) * idx + &
                                 (v(i,j+1,k) * v_n - v(i,j,k) * v_s) * idy + &
                                 (w(i,j,k+1) * v_t - w(i,j,k) * v_b) * idz - &
                                 vel(i,j,k,2) * divumac
               ugradu(i,j,k,3) = (u(i+1,j,k) * w_e - u(i,j,k) * w_w) * idx + &
                                 (v(i,j+1,k) * w_n - v(i,j,k) * w_s) * idy + &
                                 (w(i,j,k+1) * w_t - w(i,j,k) * w_b) * idz - &
                                 vel(i,j,k,3) * divumac

               ! ****************************************************
               ! Return the negative
               ! ****************************************************

               ugradu(i,j,k,1) = -ugradu(i,j,k,1)
               ugradu(i,j,k,2) = -ugradu(i,j,k,2)
               ugradu(i,j,k,3) = -ugradu(i,j,k,3)

            end do
         end do
      end do

   end subroutine compute_ugradu

   ! Upwind non-normal velocity
   function upwind ( velmns, velpls, uedge ) result (ev)

      ! Small value to protect against tiny velocities used in upwinding
      real(ar), parameter  :: small_vel = 1.0d-10

      real(ar), intent(in) :: velmns, velpls, uedge
      real(ar)             :: ev

      if ( abs(uedge) .lt. small_vel) then
         ev = half * ( velpls + velmns )
      else
         ev = merge ( velmns, velpls, uedge >= zero )
      end if

   end function upwind

end module ugradu_mod
