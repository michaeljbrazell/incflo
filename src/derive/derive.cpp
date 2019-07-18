#include <AMReX_Box.H>
#include <AMReX_EBMultiFabUtil.H>

#include <incflo.H>
#include <derive_F.H>
#include <projection_F.H>

void incflo::UpdateDerivedQuantities()
{
    BL_PROFILE("incflo::UpdateDerivedQuantities()");

    //FIXME don't need to compute divu every time this fn is called
    ComputeDivU(cur_time);
    ComputeStrainrate();
    ComputeViscosity();
    ComputeVorticity();
}

void incflo::ComputeDivU(Real time)
{
    int extrap_dir_bcs = 0;
    FillVelocityBC(time, extrap_dir_bcs);

    // Define the operator in order to compute the multi-level divergence
    //
    //        (del dot b sigma grad)) phi
    //
    LPInfo info;
    MLNodeLaplacian matrix(geom, grids, dmap, info, amrex::GetVecOfConstPtrs(ebfactory));

    // Set domain BCs for Poisson's solver
    // The domain BCs refer to level 0 only
    int bc_lo[3], bc_hi[3];
    Box domain(geom[0].Domain());

    set_ppe_bc(bc_lo, bc_hi,
               domain.loVect(), domain.hiVect(),
               &nghost,
               bc_ilo[0]->dataPtr(), bc_ihi[0]->dataPtr(),
               bc_jlo[0]->dataPtr(), bc_jhi[0]->dataPtr(),
               bc_klo[0]->dataPtr(), bc_khi[0]->dataPtr());

    matrix.setDomainBC({(LinOpBCType)bc_lo[0], (LinOpBCType)bc_lo[1], (LinOpBCType)bc_lo[2]},
                       {(LinOpBCType)bc_hi[0], (LinOpBCType)bc_hi[1], (LinOpBCType)bc_hi[2]});

    matrix.compDivergence(GetVecOfPtrs(divu), GetVecOfPtrs(vel)); 
}

// Compute the magnitude of the rate-of-strain tensor
void incflo::ComputeStrainrate()
{
    BL_PROFILE("incflo::ComputeStrainrate");

    for(int lev = 0; lev <= finest_level; lev++)
    {
	Real idx = 1. / geom[lev].CellSize()[0];
	Real idy = 1. / geom[lev].CellSize()[1];
	Real idz = 1. / geom[lev].CellSize()[2];

        // fill ghost cells
        FillPatchVel(lev, cur_time, *vel[lev], 0, (*vel[lev]).nComp());

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for(MFIter mfi(*vel[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            // Tilebox
            Box bx = mfi.tilebox();

            // This is to check efficiently if this tile contains any eb stuff
            const EBFArrayBox& veltemp_fab = static_cast<EBFArrayBox const&>((*vel[lev])[mfi]);
            const EBCellFlagFab& flags = veltemp_fab.getEBCellFlagFab();

	    const auto& strainrate_fab = (strainrate[lev])->array(mfi);
	    const auto& vel_fab = (vel[lev])->array(mfi);
	    
            if (flags.getType(bx) == FabType::covered)
            {
                (*strainrate[lev])[mfi].setVal(1.2345e200, bx);
            }
            else
            {
                if(flags.getType(amrex::grow(bx, 0)) == FabType::regular)
                {
		  AMREX_HOST_DEVICE_FOR_3D(bx, i, j, k,
		  {
		    Real ux = (vel_fab(i+1,j  ,k  ,0) - vel_fab(i-1,j  ,k  ,0)) * idx;
		    Real vx = (vel_fab(i+1,j  ,k  ,1) - vel_fab(i-1,j  ,k  ,1)) * idx;
		    Real wx = (vel_fab(i+1,j  ,k  ,2) - vel_fab(i-1,j  ,k  ,2)) * idx;
                                                                
		    Real uy = (vel_fab(i  ,j+1,k  ,0) - vel_fab(i  ,j-1,k  ,0)) * idy;
		    Real vy = (vel_fab(i  ,j+1,k  ,1) - vel_fab(i  ,j-1,k  ,1)) * idy;
		    Real wy = (vel_fab(i  ,j+1,k  ,2) - vel_fab(i  ,j-1,k  ,2)) * idy;
                                                                
		    Real uz = (vel_fab(i  ,j  ,k+1,0) - vel_fab(i  ,j  ,k-1,0)) * idz;
		    Real vz = (vel_fab(i  ,j  ,k+1,1) - vel_fab(i  ,j  ,k-1,1)) * idz;
		    Real wz = (vel_fab(i  ,j  ,k+1,2) - vel_fab(i  ,j  ,k-1,2)) * idz;
               
		    // The factor half is included here instead of in each of the above
		    strainrate_fab(i,j,k) = 0.5 * 
		      sqrt(2. * pow(ux,2) + 2. * pow(vy,2) + 2. * pow(wz,2) + 
			   pow(uy + vx,2) + pow(vz + wy,2) + pow(wx + uz,2));
		  });
                }
                else
                {
                    compute_strainrate_eb(BL_TO_FORTRAN_BOX(bx),
                                          BL_TO_FORTRAN_ANYD((*strainrate[lev])[mfi]),
                                          BL_TO_FORTRAN_ANYD((*vel[lev])[mfi]),
                                          BL_TO_FORTRAN_ANYD(flags),
                                          geom[lev].CellSize());
                }
            }
        }
    }
    
#ifdef AMREX_USE_CUDA
    Gpu::Device::synchronize();
#endif

}

void incflo::ComputeVorticity()
{
  BL_PROFILE("incflo::ComputeVorticity");

    for(int lev = 0; lev <= finest_level; lev++)
    {
        Real idx = 1.0 / geom[lev].CellSize()[0];
        Real idy = 1.0 / geom[lev].CellSize()[1];
        Real idz = 1.0 / geom[lev].CellSize()[2];
	
        // fill ghost cells
        FillPatchVel(lev, cur_time, *vel[lev], 0, vel[lev]->nComp());

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for(MFIter mfi(*vel[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            // Tilebox
            Box bx = mfi.tilebox();

            // This is to check efficiently if this tile contains any eb stuff
            const EBFArrayBox& veltemp_fab = static_cast<EBFArrayBox const&>((*vel[lev])[mfi]);
            const EBCellFlagFab& flags = veltemp_fab.getEBCellFlagFab();

            if (flags.getType(bx) == FabType::covered)
            {
                (*vort[lev])[mfi].setVal(1.2345e200, bx);
            }
            else
            {
                if(flags.getType(amrex::grow(bx, 0)) == FabType::regular)
                {
                    const auto& vel_arr = vel[lev]->array(mfi);
                    const auto& vort_arr = vort[lev]->array(mfi);

		    AMREX_HOST_DEVICE_FOR_3D(bx, i, j, k,
		    {
		      Real vx = (vel_arr(i+1, j  , k  , 1) - vel_arr(i-1, j  , k  , 1)) * idx;
		      Real wx = (vel_arr(i+1, j  , k  , 2) - vel_arr(i-1, j  , k  , 2)) * idx;
		      Real uy = (vel_arr(i  , j+1, k  , 0) - vel_arr(i  , j-1, k  , 0)) * idy;
		      Real wy = (vel_arr(i  , j+1, k  , 2) - vel_arr(i  , j-1, k  , 2)) * idy;
		      Real uz = (vel_arr(i  , j  , k+1, 0) - vel_arr(i  , j  , k-1, 0)) * idz;
		      Real vz = (vel_arr(i  , j  , k+1, 1) - vel_arr(i  , j  , k-1, 1)) * idz;
                        
		      // The factor half is included here instead of in each of the above
		      vort_arr(i,j,k) = 0.5 * sqrt(pow(wy - vz, 2) + pow(uz - wx, 2) + pow(vx - uy, 2));
                    });
                }
                else
                {
                    compute_vort_eb(BL_TO_FORTRAN_BOX(bx),
                                    BL_TO_FORTRAN_ANYD((*vort[lev])[mfi]),
                                    BL_TO_FORTRAN_ANYD((*vel[lev])[mfi]),
                                    BL_TO_FORTRAN_ANYD(flags),
                                    geom[lev].CellSize());
                }
            }
        }
    }
    
#ifdef AMREX_USE_CUDA
    Gpu::Device::synchronize();
#endif
}
