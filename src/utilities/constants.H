#ifndef CONSTANTS_H_
#define CONSTANTS_H_

#include <AMReX_REAL.H>
#include <AMReX_BLFort.H>

// Define unit vectors to easily convert indices
extern const amrex::IntVect e_x;
extern const amrex::IntVect e_y;
extern const amrex::IntVect e_z;

#ifdef __cplusplus
extern "C"
{
#endif

amrex::Real get_my_huge();

#ifdef __cplusplus
}
#endif

#endif
