#include <AMReX_REAL.H>
#include <AMReX_BLFort.H>
#include <AMReX_SPACE.H>

#ifdef __cplusplus
extern "C"
{
#endif
  void set_bc_type(const int* bc_ilo, const int* bc_ihi,
		   const int* bc_jlo, const int* bc_jhi,
		   const int* bc_klo, const int* bc_khi,
		   const int* domlo , const int* domhi,
		   amrex::Real* dx, amrex::Real* dy, amrex::Real* dz,
		   amrex::Real* xlen, amrex::Real* ylen, amrex::Real* zlen,
		   const int* ng);
  
  void fill_bc0(amrex::Real* s, const int* slo, const int* shi, 
		const int* bc_ilo, const int* bc_ihi, 
		const int* bc_jlo, const int* bc_jhi, 
		const int* bc_klo, const int* bc_khi, 
		const int* domlo, const int* domhi, 
		const int* ng);
  
  amrex::Real get_bc_u_g(const int& pID);
  amrex::Real get_bc_v_g(const int& pID);
  amrex::Real get_bc_w_g(const int& pID);
  
  amrex::Real get_bc_t_g(const int& pID);
  
  amrex::Real get_bc_ep_g(const int& pID);
  
  int get_minf();
  int get_pinf();
  int get_pout();
  int get_nsw();
  
  void get_domain_bc (int* domain_bc_out);
  
#ifdef __cplusplus
}
#endif
