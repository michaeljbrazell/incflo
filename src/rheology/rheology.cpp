#include <AMReX_Box.H>

#include <incflo.H>

AMREX_GPU_HOST_DEVICE Real expterm(Real nu)
{
   // 
   // Compute the exponential term:
   //
   //  ( 1 - exp(-nu) ) / nu ,
   //
   // making sure to avoid overflow for small nu by using the exponential Taylor series
   //
  Real result; 
  if (nu < 1.0e-14) {
    result = 1. - 0.5 * nu + nu*nu / 6.0 - nu*nu*nu / 24.0;
  }  
  else {
    result = (1. - exp(-nu)) / nu;
  }   
  return result;
}

void incflo::ComputeViscosity()
{
  BL_PROFILE("incflo::ComputeViscosity");

    for(int lev = 0; lev <= finest_level; lev++)
    {
        Box domain(geom[lev].Domain());

#ifdef _OPENMP
#pragma omp parallel
#endif
        for(MFIter mfi(*vel[lev], true); mfi.isValid(); ++mfi)
        {
            // Tilebox
            Box bx = mfi.tilebox();

            const auto& strainrate_arr = strainrate[lev]->array(mfi);
            const auto& viscosity_arr = eta[lev]->array(mfi);

	    AMREX_HOST_DEVICE_FOR_3D(bx, i, j, k, 
            {
	      auto& viscosity=viscosity_arr(i,j,k);
	      auto sr = strainrate_arr(i,j,k);
	      Real nu;

	      if (fluid_model_num == _newtonian) {
		// Viscosity is constant
		viscosity = mu;
	      }
	      else if (fluid_model_num == _powerlaw) {
		// Power-law fluid: 
		//   eta = mu dot(gamma)^(n-1)
		viscosity = mu * pow(sr,(n - 1.));
	      }
	      else if (fluid_model_num == _bingham) {
		// Papanastasiou-regularised Bingham fluid: 
		//   eta = mu + tau_0 (1 - exp(-dot(gamma) / eps)) / dot(gamma)
		nu = sr / papa_reg;
		viscosity = mu + tau_0 * expterm(nu) / papa_reg;
	      }
	      else if (fluid_model_num == _hb) {
		// Papanastasiou-regularised Herschel-Bulkley fluid: 
		//   eta = (mu dot(gamma)^n + tau_0) (1 - exp(-dot(gamma) / eps)) / dot(gamma)
		nu = sr / papa_reg;
		viscosity = (mu * pow(sr,n) + tau_0) * expterm(nu) / papa_reg;
	      }
	      else if (fluid_model_num == _smd) {
		// de Souza Mendes - Dutra fluid: 
		//   eta = (mu dot(gamma)^n + tau_0) (1 - exp(-eta_0 dot(gamma) / tau_0)) / dot(gamma)
		nu = eta_0 * sr / tau_0;
		viscosity = (mu * pow(sr,n) + tau_0) * expterm(nu) * eta_0 / tau_0;
	      }
	      else {		
		// This should have been caught earlier, but doesn't hurt to double check
		printf("\n\n Unknown fluid_model! Choose either newtonian, powerlaw, bingham, hb, smd \n\n");
	      }

            } );
        }
    }
}
