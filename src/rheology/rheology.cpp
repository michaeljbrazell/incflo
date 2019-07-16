#include <AMReX_Box.H>

#include <incflo.H>
#include <rheology_F.H>

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
		viscosity = mu * sr**(n - 1.);
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
		viscosity = (mu * sr**n + tau_0) * expterm(nu) / papa_reg;
	      }
	      else if (fluid_model_num == _smd) {
		// de Souza Mendes - Dutra fluid: 
		//   eta = (mu dot(gamma)^n + tau_0) (1 - exp(-eta_0 dot(gamma) / tau_0)) / dot(gamma)
		nu = eta_0 * sr / tau_0;
		viscosity = (mu * sr**n + tau_0) * expterm(nu) * eta_0 / tau_0;
	      }
	      else {		
		// This should have been caught earlier, but doesn't hurt to double check
		printf("\n\n Unknown fluid_model! Choose either newtonian, powerlaw, bingham, hb, smd \n\n");
	      }

            } );
        }
    }
}
