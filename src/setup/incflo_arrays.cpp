#include <incflo.H>

void incflo::AllocateArrays(int lev)
{
  UpdateEBFactory(lev);

  // ********************************************************************************
  // Cell-based arrays
  // ********************************************************************************

  // Gas density
  ro[lev].reset(new MultiFab(grids[lev], dmap[lev], 1, nghost, MFInfo(), *ebfactory[lev]));
  ro[lev]->setVal(0.);

  // Current velocity
  vel[lev].reset(new MultiFab(grids[lev], dmap[lev], 3, nghost, MFInfo(), *ebfactory[lev]));
  vel[lev]->setVal(0.);

  // Old velocity
  vel_old[lev].reset(new MultiFab(grids[lev], dmap[lev], 3, nghost, MFInfo(), *ebfactory[lev]));
  vel_old[lev]->setVal(0.);

  // Pressure gradients
  gp[lev].reset(new MultiFab(grids[lev], dmap[lev], 3, nghost, MFInfo(), *ebfactory[lev]));
  gp[lev]->setVal(0.);

  // Viscosity
  eta[lev].reset(new MultiFab(grids[lev], dmap[lev], 1, nghost, MFInfo(), *ebfactory[lev]));
  eta[lev]->setVal(0.);

  // Strain-rate magnitude
  strainrate[lev].reset(new MultiFab(grids[lev], dmap[lev], 1, nghost, MFInfo(), *ebfactory[lev]));
  strainrate[lev]->setVal(0.);

  // Vorticity
  vort[lev].reset(new MultiFab(grids[lev], dmap[lev], 1, nghost, MFInfo(), *ebfactory[lev]));
  vort[lev]->setVal(0.);

  // Convective terms for diffusion equation
  conv[lev].reset(new MultiFab(grids[lev], dmap[lev], 3, 0, MFInfo(), *ebfactory[lev]));
  conv_old[lev].reset(new MultiFab(grids[lev], dmap[lev], 3, 0, MFInfo(), *ebfactory[lev]));
  conv[lev]->setVal(0.);
  conv_old[lev]->setVal(0.);

  // Divergence of stress tensor terms for diffusion equation
  divtau_old[lev].reset(new MultiFab(grids[lev], dmap[lev], 3, 0, MFInfo(), *ebfactory[lev]));
  divtau_old[lev]->setVal(0.);

  // Slopes in x-direction
  xslopes[lev].reset(new MultiFab(grids[lev], dmap[lev], 3, nghost, MFInfo(), *ebfactory[lev]));
  xslopes[lev]->setVal(0.);

  // Slopes in y-direction
  yslopes[lev].reset(new MultiFab(grids[lev], dmap[lev], 3, nghost, MFInfo(), *ebfactory[lev]));
  yslopes[lev]->setVal(0.);

  // Slopes in z-direction
  zslopes[lev].reset(new MultiFab(grids[lev], dmap[lev], 3, nghost, MFInfo(), *ebfactory[lev]));
  zslopes[lev]->setVal(0.);

  // ********************************************************************************
  // Node-based arrays
  // ********************************************************************************

  const BoxArray & nd_grids = amrex::convert(grids[lev], IntVect{1,1,1});

  // Pressure
  p0[lev].reset(new MultiFab(nd_grids, dmap[lev], 1, nghost, MFInfo(), *ebfactory[lev]));
  p0[lev]->setVal(0.);
  p[lev].reset(new MultiFab(nd_grids, dmap[lev], 1, nghost, MFInfo(), *ebfactory[lev]));
  p[lev]->setVal(0.);

  // Divergence of velocity field
  divu[lev].reset(new MultiFab(nd_grids, dmap[lev], 1, nghost, MFInfo(), *ebfactory[lev]));
  divu[lev]->setVal(0.);

  // ********************************************************************************
  // Face-based arrays
  // ********************************************************************************

  // Create a BoxArray on x-faces.
  BoxArray x_edge_ba = grids[lev];
  x_edge_ba.surroundingNodes(0);
  m_u_mac[lev].reset(new MultiFab(x_edge_ba, dmap[lev], 1, nghost, MFInfo(), *ebfactory[lev]));
  m_u_mac[lev]->setVal(0.);

  // Create a BoxArray on y-faces.
  BoxArray y_edge_ba = grids[lev];
  y_edge_ba.surroundingNodes(1);
  m_v_mac[lev].reset(new MultiFab(y_edge_ba, dmap[lev], 1, nghost, MFInfo(), *ebfactory[lev]));
  m_v_mac[lev]->setVal(0.);

  // Create a BoxArray on y-faces.
  BoxArray z_edge_ba = grids[lev];
  z_edge_ba.surroundingNodes(2);
  m_w_mac[lev].reset(new MultiFab(z_edge_ba, dmap[lev], 1, nghost, MFInfo(), *ebfactory[lev]));
  m_w_mac[lev]->setVal(0.);
}

void incflo::RegridArrays(int lev)
{
  bool need_regrid = UpdateEBFactory(lev);

  // exit this function is ebfactory has not been updated because that means
  // that dm and ba haven't changed
  if (!need_regrid)
    return;

  // ********************************************************************************
  // Cell-based arrays
  // ********************************************************************************
  //
  // After calling copy() with dst_ngrow set to ng, we do not need to call
  // FillBoundary().
  //

  // Gas density
  std::unique_ptr<MultiFab> ro_tmp(new MultiFab(grids[lev], dmap[lev], 1, nghost, 
						MFInfo(), *ebfactory[lev]));
  ro_tmp->setVal(0.0);
  ro_tmp->copy(*ro[lev], 0, 0, 1, 0, nghost);
  ro[lev] = std::move(ro_tmp);

  // Gas velocity
  std::unique_ptr<MultiFab> vel_tmp(new MultiFab(grids[lev], dmap[lev], 3, nghost,
						 MFInfo(), *ebfactory[lev]));
  vel_tmp->setVal(0.);
  vel_tmp->copy(*vel[lev], 0, 0, vel[lev]->nComp(), 0, nghost);
  vel[lev] = std::move(vel_tmp);

  // Old gas velocity
  std::unique_ptr<MultiFab> vel_old_tmp(new MultiFab(grids[lev], dmap[lev], 3, nghost,
						     MFInfo(), *ebfactory[lev]));
  vel_old_tmp->setVal(0.);
  vel_old_tmp->copy(*vel_old[lev], 0, 0, vel_old[lev]->nComp(), 0, nghost);
  vel_old[lev] = std::move(vel_old_tmp);

  // Pressure gradients
  std::unique_ptr<MultiFab> gp_tmp(new MultiFab(grids[lev], dmap[lev], 3, nghost, 
						MFInfo(), *ebfactory[lev]));
  gp_tmp->setVal(0.);
  gp_tmp->copy(*gp[lev], 0, 0, gp[lev]->nComp(), 0, nghost);
  gp[lev] = std::move(gp_tmp);

  // Apparent viscosity
  std::unique_ptr<MultiFab> eta_tmp(new MultiFab(grids[lev], dmap[lev], 1, nghost,
						 MFInfo(), *ebfactory[lev]));
  eta_tmp->setVal(0.);
  eta_tmp->copy(*eta[lev], 0, 0, 1, 0, nghost);
  eta[lev] = std::move(eta_tmp);

  // Strain-rate magnitude
  std::unique_ptr<MultiFab> strainrate_tmp(new MultiFab(grids[lev], dmap[lev], 1, nghost,
							MFInfo(), *ebfactory[lev]));
  strainrate[lev] = std::move(strainrate_tmp);
  strainrate[lev]->setVal(0.);

  // Vorticity
  std::unique_ptr<MultiFab> vort_tmp(new MultiFab(grids[lev], dmap[lev], 1, nghost,
						  MFInfo(), *ebfactory[lev]));
  vort[lev] = std::move(vort_tmp);
  vort[lev]->setVal(0.);

  // Convective terms
  std::unique_ptr<MultiFab> conv_tmp(new MultiFab(grids[lev], dmap[lev], 3, nghost,
						  MFInfo(), *ebfactory[lev]));
  conv[lev] = std::move(conv_tmp);
  conv[lev]->setVal(0.);

  std::unique_ptr<MultiFab> conv_old_tmp(new MultiFab(grids[lev], dmap[lev], 3, nghost,
						      MFInfo(), *ebfactory[lev]));
  conv_old[lev] = std::move(conv_old_tmp);
  conv_old[lev]->setVal(0.);

  // Divergence of stress tensor terms 
  std::unique_ptr<MultiFab> divtau_tmp(new MultiFab(grids[lev], dmap[lev], 3, nghost,
						    MFInfo(), *ebfactory[lev]));
  divtau_old[lev] = std::move(divtau_tmp);
  divtau_old[lev]->setVal(0.);

  // Slopes in x-direction
  std::unique_ptr<MultiFab> xslopes_tmp(new MultiFab(grids[lev], dmap[lev], 3, nghost, 
						     MFInfo(), *ebfactory[lev]));
  xslopes[lev] = std::move(xslopes_tmp);
  xslopes[lev] -> setVal(0.);

  // Slopes in y-direction
  std::unique_ptr<MultiFab> yslopes_tmp(new MultiFab(grids[lev], dmap[lev], 3, nghost, 
						     MFInfo(), *ebfactory[lev]));
  yslopes[lev] = std::move(yslopes_tmp);
  yslopes[lev] -> setVal(0.);

  // Slopes in z-direction
  std::unique_ptr<MultiFab> zslopes_tmp(new MultiFab(grids[lev], dmap[lev], 3, nghost, 
						     MFInfo(), *ebfactory[lev]));
  zslopes[lev] = std::move(zslopes_tmp);
  zslopes[lev] -> setVal(0.);

  /****************************************************************************
   * Node-based Arrays                                                        *
   ****************************************************************************/

  // Pressures, projection vars
  const BoxArray & nd_grids = amrex::convert(grids[lev], IntVect{1,1,1});

  std::unique_ptr<MultiFab> p_tmp(new MultiFab(nd_grids, dmap[lev], 1, nghost, 
					       MFInfo(), *ebfactory[lev]));
  p_tmp->setVal(0.0);
  p_tmp->copy(*p[lev],0,0,1,0,nghost);
  p[lev] = std::move(p_tmp);

  std::unique_ptr<MultiFab> p0_tmp(new MultiFab(nd_grids, dmap[lev], 1, nghost, 
						MFInfo(), *ebfactory[lev]));
  p0_tmp->setVal(0.0);
  p0_tmp->copy(*p0[lev],0,0,1,0,nghost);
  p0[lev] = std::move(p0_tmp);

  std::unique_ptr<MultiFab> divu_tmp(new MultiFab(nd_grids, dmap[lev], 1, nghost, 
						  MFInfo(), *ebfactory[lev]));
  divu[lev] = std::move(divu_tmp);
  divu[lev]->setVal(0.);

  /****************************************************************************
   * Face-based Arrays                                                        *
   ****************************************************************************/

  BoxArray x_ba = grids[lev];
  x_ba = x_ba.surroundingNodes(0);

  // MAC velocity
  std::unique_ptr<MultiFab> u_mac_tmp(new MultiFab(x_ba, dmap[lev], 1, nghost, 
						   MFInfo(), *ebfactory[lev]));
  m_u_mac[lev] = std::move(u_mac_tmp);
  m_u_mac[lev]->setVal(0.0);

  BoxArray y_ba = grids[lev];
  y_ba = y_ba.surroundingNodes(1);

  // MAC velocity
  std::unique_ptr<MultiFab> v_mac_tmp(new MultiFab(y_ba, dmap[lev], 1, nghost, 
						   MFInfo(), *ebfactory[lev]));
  m_v_mac[lev] = std::move(v_mac_tmp);
  m_v_mac[lev] -> setVal(0.0);

  BoxArray z_ba = grids[lev];
  z_ba = z_ba.surroundingNodes(2);

  // MAC velocity
  std::unique_ptr<MultiFab> w_mac_tmp(new MultiFab(z_ba, dmap[lev], 1, nghost, 
						   MFInfo(), *ebfactory[lev]));
  m_w_mac[lev] = std::move(w_mac_tmp);
  m_w_mac[lev] -> setVal(0.0);
}

// Resize all arrays when instance of incflo class is constructed.
// This is only done at the very start of the simulation. 
void incflo::ResizeArrays()
{
  // Time holders for fillpatch stuff
  t_new.resize(max_level + 1);
  t_old.resize(max_level + 1);

  // Density 
  ro.resize(max_level + 1);

  // Current (vel) and old (vel_old) velocities
  vel.resize(max_level + 1);
  vel_old.resize(max_level + 1);

  // Pressure
  p.resize(max_level + 1);
  p0.resize(max_level + 1);

  // Pressure gradients
  gp.resize(max_level + 1);

  // Derived quantities: viscosity, strainrate, vorticity, div(u)
  eta.resize(max_level + 1);
  strainrate.resize(max_level + 1);
  vort.resize(max_level + 1);
  divu.resize(max_level + 1);

  // Convective terms u grad u 
  conv.resize(max_level + 1);
  conv_old.resize(max_level + 1);
  divtau_old.resize(max_level + 1);

  // MAC velocities used for defining convective term
  m_u_mac.resize(max_level + 1);
  m_v_mac.resize(max_level + 1);
  m_w_mac.resize(max_level + 1);

  // Slopes used for upwinding convective terms
  xslopes.resize(max_level + 1);
  yslopes.resize(max_level + 1);
  zslopes.resize(max_level + 1);

  // BCs
  bc_ilo.resize(max_level + 1);
  bc_ihi.resize(max_level + 1);
  bc_jlo.resize(max_level + 1);
  bc_jhi.resize(max_level + 1);
  bc_klo.resize(max_level + 1);
  bc_khi.resize(max_level + 1);

  // EB factory
  ebfactory.resize(max_level + 1);
}

void incflo::MakeBCArrays()
{
  for(int lev = 0; lev <= max_level; lev++)
    {
      // Define and allocate the integer MultiFab that is the outside adjacent cells of the
      // problem domain.
      Box domainx(geom[lev].Domain());
      domainx.grow(1, nghost);
      domainx.grow(2, nghost);
      Box box_ilo = amrex::adjCellLo(domainx, 0, 1);
      Box box_ihi = amrex::adjCellHi(domainx, 0, 1);

      Box domainy(geom[lev].Domain());
      domainy.grow(0, nghost);
      domainy.grow(2, nghost);
      Box box_jlo = amrex::adjCellLo(domainy, 1, 1);
      Box box_jhi = amrex::adjCellHi(domainy, 1, 1);

      Box domainz(geom[lev].Domain());
      domainz.grow(0, nghost);
      domainz.grow(1, nghost);
      Box box_klo = amrex::adjCellLo(domainz, 2, 1);
      Box box_khi = amrex::adjCellHi(domainz, 2, 1);

      // Note that each of these is a single IArrayBox so every process has a copy of them
      bc_ilo[lev].reset(new IArrayBox(box_ilo, 2));
      bc_ihi[lev].reset(new IArrayBox(box_ihi, 2));
      bc_jlo[lev].reset(new IArrayBox(box_jlo, 2));
      bc_jhi[lev].reset(new IArrayBox(box_jhi, 2));
      bc_klo[lev].reset(new IArrayBox(box_klo, 2));
      bc_khi[lev].reset(new IArrayBox(box_khi, 2));
    }
}

