#include <incflo.H>
#include <divop_conv.hpp>
namespace ugradu_aux {

//
// Compute upwind non-normal velocity
//
AMREX_GPU_HOST_DEVICE
Real
upwind(const Real velocity_minus,
       const Real velocity_plus,
       const Real u_edge)
{
  // Small value to protect against tiny velocities used in upwinding
  const Real small_velocity(1.e-10);

  if(std::abs(u_edge) < small_velocity)
    return .5*(velocity_minus+velocity_plus);

  return u_edge > 0 ? velocity_minus : velocity_plus;
}

AMREX_GPU_HOST_DEVICE
bool
is_equal_to_any(const int bc,
                const int* bc_types,
                const int size)
{
  for(int i(0); i < size; ++i)
  {
    if(bc == bc_types[i])
      return true;
  }
  return false;
}

} // end namespace ugradu_aux

using namespace ugradu_aux;

//
// Compute the three components of the convection term
//
void
incflo::incflo_compute_ugradu( Box& bx,
                           Vector< std::unique_ptr<MultiFab> >& conv, 
                           Vector< std::unique_ptr<MultiFab> >& vel,
                           Vector< std::unique_ptr<MultiFab> >& u_mac,
                           Vector< std::unique_ptr<MultiFab> >& v_mac,
                           Vector< std::unique_ptr<MultiFab> >& w_mac,
                           MFIter* mfi,
                           Box& domain,
                           const int lev)
{
  const Real* dx = geom[lev].CellSize();
  const amrex::Dim3 dom_low = amrex::lbound(domain);
  const amrex::Dim3 dom_high = amrex::ubound(domain);

  Array4<Real> const& ugradu = conv[lev]->array(*mfi); 
  
  Array4<Real> const& velocity = vel[lev]->array(*mfi);
  
  Array4<Real> const& u = u_mac[lev]->array(*mfi);
  Array4<Real> const& v = v_mac[lev]->array(*mfi);
  Array4<Real> const& w = w_mac[lev]->array(*mfi);

  Array4<Real> const& x_slopes = xslopes[lev]->array(*mfi);
  Array4<Real> const& y_slopes = yslopes[lev]->array(*mfi);
  Array4<Real> const& z_slopes = zslopes[lev]->array(*mfi);

  Array4<int> const& bc_ilo_type = bc_ilo[lev]->array();
  Array4<int> const& bc_ihi_type = bc_ihi[lev]->array();
  Array4<int> const& bc_jlo_type = bc_jlo[lev]->array();
  Array4<int> const& bc_jhi_type = bc_jhi[lev]->array();
  Array4<int> const& bc_klo_type = bc_klo[lev]->array();
  Array4<int> const& bc_khi_type = bc_khi[lev]->array();

  const Real idx(1/dx[0]), idy(1/dx[1]), idz(1/dx[2]);

  // Vectorize the boundary conditions list in order to use it in lambda
  // functions
  const GpuArray<int, 4> bc_types =
    {bc_list.get_minf(), bc_list.get_nsw(), bc_list.get_pinf(), bc_list.get_pout()};

  AMREX_HOST_DEVICE_FOR_3D(bx, i, j, k,
  {
    Real u_w(0); Real v_w(0); Real w_w(0);
    Real u_e(0); Real v_e(0); Real w_e(0);
    Real u_s(0); Real v_s(0); Real w_s(0);
    Real u_n(0); Real v_n(0); Real w_n(0);
    Real u_b(0); Real v_b(0); Real w_b(0);
    Real u_t(0); Real v_t(0); Real w_t(0);
    Real umns(0); Real vmns(0); Real wmns(0);
    Real upls(0); Real vpls(0); Real wpls(0);

    //
    // West face
    //
    // In the case of MINF, NSW  we are using the prescribed Dirichlet value
    // In the case of PINF, POUT we are using the upwind value
    if((i == dom_low.x) and
     ugradu_aux::is_equal_to_any(bc_ilo_type(dom_low.x-1,j,k,0),
                                 bc_types.data(), bc_types.size()))
    {
      u_w = velocity(i-1,j,k,0);
      v_w = velocity(i-1,j,k,1);
      w_w = velocity(i-1,j,k,2);
    }
    else {
      upls = velocity(i  ,j,k,0) - .5*x_slopes(i  ,j,k,0);
      umns = velocity(i-1,j,k,0) + .5*x_slopes(i-1,j,k,0);
      vpls = velocity(i  ,j,k,1) - .5*x_slopes(i  ,j,k,1);
      vmns = velocity(i-1,j,k,1) + .5*x_slopes(i-1,j,k,1);
      wpls = velocity(i  ,j,k,2) - .5*x_slopes(i  ,j,k,2);
      wmns = velocity(i-1,j,k,2) + .5*x_slopes(i-1,j,k,2);

      u_w = upwind( umns, upls, u(i,j,k) );
      v_w = upwind( vmns, vpls, u(i,j,k) );
      w_w = upwind( wmns, wpls, u(i,j,k) );
    }

    //
    // East face
    //
    // In the case of MINF, NSW  we are using the prescribed Dirichlet value
    // In the case of PINF, POUT we are using the upwind value
    if((i == dom_high.x) and
     ugradu_aux::is_equal_to_any(bc_ihi_type(dom_high.x+1,j,k,0),
                                 bc_types.data(), bc_types.size()))
    {
      u_e = velocity(i+1,j,k,0);
      v_e = velocity(i+1,j,k,1);
      w_e = velocity(i+1,j,k,2);
    }
    else {
      upls = velocity(i+1,j,k,0) - .5*x_slopes(i+1,j,k,0);
      umns = velocity(i  ,j,k,0) + .5*x_slopes(i  ,j,k,0);
      vpls = velocity(i+1,j,k,1) - .5*x_slopes(i+1,j,k,1);
      vmns = velocity(i  ,j,k,1) + .5*x_slopes(i  ,j,k,1);
      wpls = velocity(i+1,j,k,2) - .5*x_slopes(i+1,j,k,2);
      wmns = velocity(i  ,j,k,2) + .5*x_slopes(i  ,j,k,2);

      u_e = upwind( umns, upls, u(i+1,j,k) );
      v_e = upwind( vmns, vpls, u(i+1,j,k) );
      w_e = upwind( wmns, wpls, u(i+1,j,k) );
    }

    //
    // South face
    //
    // In the case of MINF       we are using the prescribed Dirichlet value
    // In the case of PINF, POUT we are using the upwind value
    if((j == dom_low.y) and
     ugradu_aux::is_equal_to_any(bc_jlo_type(i,dom_low.y-1,k,0),
                                 bc_types.data(), bc_types.size()))
    {
      u_s = velocity(i,j-1,k,0);
      v_s = velocity(i,j-1,k,1);
      w_s = velocity(i,j-1,k,2);
    }
    else {
      upls = velocity(i,j  ,k,0) - .5*y_slopes(i,j  ,k,0);
      umns = velocity(i,j-1,k,0) + .5*y_slopes(i,j-1,k,0);
      vpls = velocity(i,j  ,k,1) - .5*y_slopes(i,j  ,k,1);
      vmns = velocity(i,j-1,k,1) + .5*y_slopes(i,j-1,k,1);
      wpls = velocity(i,j  ,k,2) - .5*y_slopes(i,j  ,k,2);
      wmns = velocity(i,j-1,k,2) + .5*y_slopes(i,j-1,k,2);

      u_s = upwind( umns, upls, v(i,j,k) );
      v_s = upwind( vmns, vpls, v(i,j,k) );
      w_s = upwind( wmns, wpls, v(i,j,k) );
    }

    //
    // North face
    //
    // In the case of MINF       we are using the prescribed Dirichlet value
    // In the case of PINF, POUT we are using the upwind value
    if((j == dom_high.y) and
     ugradu_aux::is_equal_to_any(bc_jhi_type(i,dom_high.y+1,k,0),
                                 bc_types.data(), bc_types.size()))
    {
      u_n = velocity(i,j+1,k,0);
      v_n = velocity(i,j+1,k,1);
      w_n = velocity(i,j+1,k,2);
    }
    else {
      upls = velocity(i,j+1,k,0) - .5*y_slopes(i,j+1,k,0);
      umns = velocity(i,j  ,k,0) + .5*y_slopes(i,j  ,k,0);
      vpls = velocity(i,j+1,k,1) - .5*y_slopes(i,j+1,k,1);
      vmns = velocity(i,j  ,k,1) + .5*y_slopes(i,j  ,k,1);
      wpls = velocity(i,j+1,k,2) - .5*y_slopes(i,j+1,k,2);
      wmns = velocity(i,j  ,k,2) + .5*y_slopes(i,j  ,k,2);

      u_n = upwind( umns, upls, v(i,j+1,k) );
      v_n = upwind( vmns, vpls, v(i,j+1,k) );
      w_n = upwind( wmns, wpls, v(i,j+1,k) );
    }

    //
    // Bottom face
    //
    // In the case of MINF       we are using the prescribed Dirichlet value
    // In the case of PINF, POUT we are using the upwind value
    if((k == dom_low.z) and
     ugradu_aux::is_equal_to_any(bc_klo_type(i,j,dom_low.z-1,0),
                                 bc_types.data(), bc_types.size()))
    {
      u_b = velocity(i,j,k-1,0);
      v_b = velocity(i,j,k-1,1);
      w_b = velocity(i,j,k-1,2);
    }
    else {
      upls = velocity(i,j,k  ,0) - .5*z_slopes(i,j,k  ,0);
      umns = velocity(i,j,k-1,0) + .5*z_slopes(i,j,k-1,0);
      vpls = velocity(i,j,k  ,1) - .5*z_slopes(i,j,k  ,1);
      vmns = velocity(i,j,k-1,1) + .5*z_slopes(i,j,k-1,1);
      wpls = velocity(i,j,k  ,2) - .5*z_slopes(i,j,k  ,2);
      wmns = velocity(i,j,k-1,2) + .5*z_slopes(i,j,k-1,2);

      u_b = upwind( umns, upls, w(i,j,k) );
      v_b = upwind( vmns, vpls, w(i,j,k) );
      w_b = upwind( wmns, wpls, w(i,j,k) );
    }

    //
    // Top face
    //
    // In the case of MINF       we are using the prescribed Dirichlet value
    // In the case of PINF, POUT we are using the upwind value
    if((k == dom_high.z) and
     ugradu_aux::is_equal_to_any(bc_khi_type(i,j,dom_high.z+1,0),
                                 bc_types.data(), bc_types.size()))
    {
      u_t = velocity(i,j,k+1,0);
      v_t = velocity(i,j,k+1,1);
      w_t = velocity(i,j,k+1,2);
    }
    else {
      upls = velocity(i,j,k+1,0) - .5*z_slopes(i,j,k+1,0);
      umns = velocity(i,j,k  ,0) + .5*z_slopes(i,j,k  ,0);
      vpls = velocity(i,j,k+1,1) - .5*z_slopes(i,j,k+1,1);
      vmns = velocity(i,j,k  ,1) + .5*z_slopes(i,j,k  ,1);
      wpls = velocity(i,j,k+1,2) - .5*z_slopes(i,j,k+1,2);
      wmns = velocity(i,j,k  ,2) + .5*z_slopes(i,j,k  ,2);

      u_t = upwind( umns, upls, w(i,j,k+1) );
      v_t = upwind( vmns, vpls, w(i,j,k+1) );
      w_t = upwind( wmns, wpls, w(i,j,k+1) );
    }

    // ****************************************************
    // Define convective terms -- conservatively
    // ugradu = ( div(u^MAC u^cc) - u^cc div(u^MAC) )
    // ****************************************************
    Real divumac = (u(i+1,j,k) - u(i,j,k)) * idx + 
      (v(i,j+1,k) - v(i,j,k)) * idy + 
      (w(i,j,k+1) - w(i,j,k)) * idz;
    
    ugradu(i,j,k,0) = (u(i+1,j,k) * u_e - u(i,j,k) * u_w) * idx + 
      (v(i,j+1,k) * u_n - v(i,j,k) * u_s) * idy + 
      (w(i,j,k+1) * u_t - w(i,j,k) * u_b) * idz - 
      velocity(i,j,k,0) * divumac;
    ugradu(i,j,k,1) = (u(i+1,j,k) * v_e - u(i,j,k) * v_w) * idx + 
      (v(i,j+1,k) * v_n - v(i,j,k) * v_s) * idy + 
      (w(i,j,k+1) * v_t - w(i,j,k) * v_b) * idz - 
      velocity(i,j,k,1) * divumac;
    ugradu(i,j,k,2) = (u(i+1,j,k) * w_e - u(i,j,k) * w_w) * idx + 
      (v(i,j+1,k) * w_n - v(i,j,k) * w_s) * idy + 
      (w(i,j,k+1) * w_t - w(i,j,k) * w_b) * idz - 
      velocity(i,j,k,2) * divumac;

    // // ****************************************************
    // // Return the negative
    // // ****************************************************
    
    ugradu(i,j,k,0) = -ugradu(i,j,k,0);
    ugradu(i,j,k,1) = -ugradu(i,j,k,1);
    ugradu(i,j,k,2) = -ugradu(i,j,k,2);
    
  });

#ifdef AMREX_USE_CUDA
  Gpu::Device::synchronize();
#endif
}


//
// Compute [TODO description]
//
void
incflo::incflo_compute_ugradu_eb(Box& bx,
                             Vector< std::unique_ptr<MultiFab> >& conv, 
                             Vector< std::unique_ptr<MultiFab> >& vel,
                             Vector< std::unique_ptr<MultiFab> >& u_mac,
                             Vector< std::unique_ptr<MultiFab> >& v_mac,
                             Vector< std::unique_ptr<MultiFab> >& w_mac,
                             MFIter* mfi,
                             Array<const MultiCutFab*,AMREX_SPACEDIM>& areafrac,
                             Array<const MultiCutFab*,AMREX_SPACEDIM>& facecent,
                             const MultiFab* volfrac,
                             const MultiCutFab* bndrycent,
                             Box& domain,
                             const EBCellFlagFab& flags,
                             const int lev)
{
  AMREX_ASSERT_WITH_MESSAGE(nghost >= 4, "Compute divop_conv(): ng must be >= 4");

  const Real* dx = geom[lev].CellSize();
  const amrex::Dim3 dom_low = amrex::lbound(domain);
  const amrex::Dim3 dom_high = amrex::ubound(domain);

  Array4<Real> const& ugradu = conv[lev]->array(*mfi);

  Array4<Real> const& velocity = vel[lev]->array(*mfi);

  Array4<const Real> const& areafrac_x = areafrac[0]->array(*mfi);
  Array4<const Real> const& areafrac_y = areafrac[1]->array(*mfi);
  Array4<const Real> const& areafrac_z = areafrac[2]->array(*mfi);

  Array4<Real> const& u = u_mac[lev]->array(*mfi);
  Array4<Real> const& v = v_mac[lev]->array(*mfi);
  Array4<Real> const& w = w_mac[lev]->array(*mfi);

  Array4<Real> const& x_slopes = xslopes[lev]->array(*mfi);
  Array4<Real> const& y_slopes = yslopes[lev]->array(*mfi);
  Array4<Real> const& z_slopes = zslopes[lev]->array(*mfi);

  Array4<int> const& bc_ilo_type = bc_ilo[lev]->array();
  Array4<int> const& bc_ihi_type = bc_ihi[lev]->array();
  Array4<int> const& bc_jlo_type = bc_jlo[lev]->array();
  Array4<int> const& bc_jhi_type = bc_jhi[lev]->array();
  Array4<int> const& bc_klo_type = bc_klo[lev]->array();
  Array4<int> const& bc_khi_type = bc_khi[lev]->array();

  // Number of Halo layers
  const int nh(3);

  const Box ubx = amrex::surroundingNodes(amrex::grow(bx,nh),0);
  const Box vbx = amrex::surroundingNodes(amrex::grow(bx,nh),1);
  const Box wbx = amrex::surroundingNodes(amrex::grow(bx,nh),2);

  const int ncomp(3);
  
  FArrayBox fxfab(ubx, ncomp);
  FArrayBox fyfab(vbx, ncomp);
  FArrayBox fzfab(wbx, ncomp);

  Array4<Real> const& fx = fxfab.array();
  Array4<Real> const& fy = fyfab.array();
  Array4<Real> const& fz = fzfab.array();

  const GpuArray<int, 3> bc_types =
    {bc_list.get_minf(), bc_list.get_pinf(), bc_list.get_pout()};

  const Real my_huge = get_my_huge();
  //
  // First compute the convective fluxes at the face center
  // Do this on ALL faces on the tile, i.e. INCLUDE as many ghost faces as
  // possible
  //

  //
  // ===================== X =====================
  //
  AMREX_HOST_DEVICE_FOR_4D(ubx, ncomp, i, j, k, n,
  {
    Real u_face(0);
    Real upls(0); Real umns(0);

    if( areafrac_x(i,j,k) > 0 ) {
      if( i <= dom_low.x and
       ugradu_aux::is_equal_to_any(bc_ilo_type(dom_low.x-1,j,k,0),
                                   bc_types.data(), bc_types.size()))
      {
        u_face = velocity(dom_low.x-1,j,k,n);
      }
      else if( i >= dom_high.x+1 and
       ugradu_aux::is_equal_to_any(bc_ihi_type(dom_high.x+1,j,k,0),
                                   bc_types.data(), bc_types.size()))
      {
        u_face = velocity(dom_high.x+1,j,k,n);
      }
      else {
        upls = velocity(i  ,j,k,n) - .5*x_slopes(i  ,j,k,n);
        umns = velocity(i-1,j,k,n) + .5*x_slopes(i-1,j,k,n);

        u_face = upwind( umns, upls, u(i,j,k) );
      }
    }
    else {
      u_face = my_huge; 
    }
    fx(i,j,k,n) = u(i,j,k) * u_face;
  });

  //
  // ===================== Y =====================
  //
  AMREX_HOST_DEVICE_FOR_4D(vbx, ncomp, i, j, k, n,
  {
    Real v_face(0);
    Real vpls(0); Real vmns(0);

    if( areafrac_y(i,j,k) > 0 ) {
      if( j <= dom_low.y and
       ugradu_aux::is_equal_to_any(bc_jlo_type(i,dom_low.y-1,k,0),
                                   bc_types.data(), bc_types.size()))
      {
        v_face = velocity(i,dom_low.y-1,k,n);
      }
      else if( j >= dom_high.y+1 and
       ugradu_aux::is_equal_to_any(bc_jhi_type(i,dom_high.y+1,k,0),
                                   bc_types.data(), bc_types.size()))
      {
        v_face = velocity(i,dom_high.y+1,k,n);
      }
      else {
        vpls = velocity(i,j  ,k,n) - .5*y_slopes(i,j  ,k,n);
        vmns = velocity(i,j-1,k,n) + .5*y_slopes(i,j-1,k,n);

        v_face = upwind( vmns, vpls, v(i,j,k) );
      }
    }
    else {
      v_face = my_huge;
    }
    fy(i,j,k,n) = v(i,j,k) * v_face;
  });

  //
  // ===================== Z =====================
  //
  AMREX_HOST_DEVICE_FOR_4D(wbx, ncomp, i, j, k, n,
  {
    Real w_face(0);
    Real wpls(0); Real wmns(0);

    if( areafrac_z(i,j,k) > 0 ) {
      if( k <= dom_low.z and
       ugradu_aux::is_equal_to_any(bc_klo_type(i,j,dom_low.z-1,0),
                                   bc_types.data(), bc_types.size()))
      {
        w_face = velocity(i,j,dom_low.z-1,n);
      }
      else if( k >= dom_high.z+1 and
       ugradu_aux::is_equal_to_any(bc_khi_type(i,j,dom_high.z+1,0),
                                   bc_types.data(), bc_types.size()))
      {
        w_face = velocity(i,j,dom_high.z+1,n);
      }
      else {
        wpls = velocity(i,j,k  ,n) - .5*z_slopes(i,j,k  ,n);
        wmns = velocity(i,j,k-1,n) + .5*z_slopes(i,j,k-1,n);

        w_face = upwind( wmns, wpls, w(i,j,k) );
      }
    }
    else {
      w_face = my_huge;
    }
    fz(i,j,k,n) = w(i,j,k) * w_face;
  });

#ifdef AMREX_USE_CUDA
  Gpu::Device::synchronize();
#endif

  const int cyclic_x = geom[0].isPeriodic(0) ? 1 : 0;
  const int cyclic_y = geom[0].isPeriodic(1) ? 1 : 0;
  const int cyclic_z = geom[0].isPeriodic(2) ? 1 : 0;

  // Compute div(tau) with EB algorithm
  compute_divop_conv(bx, *conv[lev], mfi, fxfab, fyfab, fzfab, 
                     areafrac, facecent, flags, volfrac, bndrycent, domain,
                     cyclic_x, cyclic_y, cyclic_z, dx);

  AMREX_HOST_DEVICE_FOR_3D(bx, i, j, k,
  {
    const Real coefficient(-1.);
    ugradu(i,j,k,0) *= coefficient; 
    ugradu(i,j,k,1) *= coefficient; 
    ugradu(i,j,k,2) *= coefficient; 
  });

#ifdef AMREX_USE_CUDA
  Gpu::Device::synchronize();
#endif
}
