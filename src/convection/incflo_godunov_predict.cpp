#include "incflo_godunov_ppm.H" 
#include "incflo.H"
#include <AMReX_BCRec.H>
#include <iomanip> 

using namespace amrex;

void incflo::make_ppm_integrals (int lev, Box const& bx, int ncomp,
                                 Array4<Real> const& Imx,
                                 Array4<Real> const& Ipx,
                                 Array4<Real> const& Imy,
                                 Array4<Real> const& Ipy,
                                 Array4<Real> const& Imz,
                                 Array4<Real> const& Ipz,
                                 Array4<Real const> const& q,
                                 Array4<Real const> const& vel)
{
    Real l_dt = this->dt;
    const auto dx = Geom(lev).CellSize();
    const Box& domain = Geom(lev).Domain();
    const Dim3 dlo = amrex::lbound(domain);
    const Dim3 dhi = amrex::ubound(domain);

    BCRec const* pbc = get_velocity_bcrec_device_ptr();

    amrex::ParallelFor(bx, AMREX_SPACEDIM,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        Godunov_ppm_pred_x(i,j,k,n,l_dt,dx[0],q,vel,Imx,Ipx,pbc[0],dlo.x,dhi.x);
        Godunov_ppm_pred_y(i,j,k,n,l_dt,dx[1],q,vel,Imy,Ipy,pbc[1],dlo.y,dhi.y);
        Godunov_ppm_pred_z(i,j,k,n,l_dt,dx[2],q,vel,Imz,Ipz,pbc[2],dlo.z,dhi.z);
    });
}

void incflo::make_trans_velocities (int lev, Box const& xbx, Box const& ybx, Box const& zbx,
                                    Array4<Real> const& u_ad,
                                    Array4<Real> const& v_ad,
                                    Array4<Real> const& w_ad,
                                    Array4<Real const> const& Imx,
                                    Array4<Real const> const& Ipx,
                                    Array4<Real const> const& Imy,
                                    Array4<Real const> const& Ipy,
                                    Array4<Real const> const& Imz,
                                    Array4<Real const> const& Ipz,
                                    Array4<Real const> const& vel,
                                    Array4<Real const> const& f)
{
    Real l_dt = this->dt;
    bool l_use_forces_in_trans = this->use_forces_in_trans;

    const Box& domain = Geom(lev).Domain();

    BCRec const* pbc = get_velocity_bcrec_device_ptr();

    amrex::ParallelFor(xbx, ybx, zbx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        // We only care about x-velocity on x-faces here
        constexpr int n = 0;

        Real lo, hi;
        if (l_use_forces_in_trans) {
            lo = Ipx(i-1,j,k,n) + 0.5*l_dt*f(i-1,j,k,n);
            hi = Imx(i  ,j,k,n) + 0.5*l_dt*f(i  ,j,k,n);
        } else {
            lo = Ipx(i-1,j,k,n);
            hi = Imx(i  ,j,k,n);
        }

        auto bc = pbc[n];
        Godunov_trans_xbc_lo(i, j, k, n, vel, lo, hi, lo, bc.lo(0),
                             domain.loVect(), domain.hiVect(), false, false);
        Godunov_trans_xbc_hi(i, j, k, n, vel, lo, hi, lo, bc.hi(0), 
                             domain.loVect(), domain.hiVect(), false, false);
        constexpr Real eps = 1e-6;

        Real st = ( (lo+hi) >= 0.) ? lo : hi;
        bool ltm = ( (lo <= 0. && hi >= 0.) || (std::abs(lo+hi) < eps) );
        u_ad(i,j,k) = ltm ? 0. : st;
    },
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        // We only care about y-velocity on y-faces here
        constexpr int n = 1;

        Real lo, hi;
        if (l_use_forces_in_trans) {
            lo = Ipy(i,j-1,k,n) + 0.5*l_dt*f(i,j-1,k,n);
            hi = Imy(i,j  ,k,n) + 0.5*l_dt*f(i,j  ,k,n);
        } else {
            lo = Ipy(i,j-1,k,n);
            hi = Imy(i,j  ,k,n);
        }

        auto bc = pbc[n];
        Godunov_trans_ybc_lo(i, j, k, n, vel, lo, hi, lo, bc.lo(1),
                             domain.loVect(), domain.hiVect(), false, false);
        Godunov_trans_ybc_hi(i, j, k, n, vel, lo, hi, lo, bc.hi(1),
                             domain.loVect(), domain.hiVect(), false, false);
        constexpr Real eps = 1e-6;

        Real st = ( (lo+hi) >= 0.) ? lo : hi;
        bool ltm = ( (lo <= 0. && hi >= 0.) || (std::abs(lo+hi) < eps) );
        v_ad(i,j,k) = ltm ? 0. : st;
    },
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        // We only care about z-velocity on z-faces here
        constexpr int n = 2;

        Real lo, hi;
        if (l_use_forces_in_trans) {
            lo = Ipz(i,j,k-1,n) + 0.5*l_dt*f(i,j,k-1,n);
            hi = Imz(i,j,k  ,n) + 0.5*l_dt*f(i,j,k  ,n);
        } else {
            lo = Ipz(i,j,k-1,n);
            hi = Imz(i,j,k  ,n);
        }

        auto bc = pbc[n];
        Godunov_trans_zbc_lo(i, j, k, n, vel, lo, hi, lo, bc.lo(2),
                             domain.loVect(), domain.hiVect(), false, false);
        Godunov_trans_zbc_hi(i, j, k, n, vel, lo, hi, lo, bc.hi(2),
                             domain.loVect(), domain.hiVect(), false, false);

        constexpr Real eps = 1e-6;

        Real st = ( (lo+hi) >= 0.) ? lo : hi;
        bool ltm = ( (lo <= 0. && hi >= 0.) || (std::abs(lo+hi) < eps) );
        w_ad(i,j,k) = ltm ? 0. : st;
    });
}

void incflo::predict_godunov_on_box (int lev, Box const& bx, int ncomp,
                                     Box const& xbx, Box const& ybx, Box const& zbx,
                                     Array4<Real> const& qx,
                                     Array4<Real> const& qy,
                                     Array4<Real> const& qz,
                                     Array4<Real const> const& q,
                                     Array4<Real const> const& u_ad,
                                     Array4<Real const> const& v_ad,
                                     Array4<Real const> const& w_ad,
                                     Array4<Real> const& Imx,
                                     Array4<Real> const& Ipx,
                                     Array4<Real> const& Imy,
                                     Array4<Real> const& Ipy,
                                     Array4<Real> const& Imz,
                                     Array4<Real> const& Ipz,
                                     Array4<Real const> const& f,
                                     Real* p)
{
    Real l_dt = this->dt;
    bool l_use_forces_in_trans = this->use_forces_in_trans;

    const Box& domain = Geom(lev).Domain();
    const Dim3 dlo = amrex::lbound(domain);
    const Dim3 dhi = amrex::ubound(domain);
    Real dx = Geom(lev).CellSize(0);
    Real dy = Geom(lev).CellSize(1);
    Real dz = Geom(lev).CellSize(2);

    // We only predict velocity in this routine, which is convectively not conservatively, updated
    GpuArray<int,3> iconserv{0,0,0};

    BCRec const* pbc = get_velocity_bcrec_device_ptr();

    Box xebox = Box(bx).grow(1,1).grow(2,1).surroundingNodes(0);
    Box yebox = Box(bx).grow(0,1).grow(2,1).surroundingNodes(1);
    Box zebox = Box(bx).grow(0,1).grow(1,1).surroundingNodes(2);
    Array4<Real> xlo = makeArray4(p, xebox, ncomp);
    p += xlo.size();
    Array4<Real> xhi = makeArray4(p, xebox, ncomp);
    p += xhi.size();
    Array4<Real> ylo = makeArray4(p, yebox, ncomp);
    p += ylo.size();
    Array4<Real> yhi = makeArray4(p, yebox, ncomp);
    p += yhi.size();
    Array4<Real> zlo = makeArray4(p, zebox, ncomp);
    p += zlo.size();
    Array4<Real> zhi = makeArray4(p, zebox, ncomp);
    p += zhi.size();

    amrex::ParallelFor(
        xebox, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            Real lo, hi;
            if (l_use_forces_in_trans) {
                lo = Ipx(i-1,j,k,n) + 0.5*l_dt*f(i-1,j,k,n);
                hi = Imx(i  ,j,k,n) + 0.5*l_dt*f(i  ,j,k,n);
            } else {
                lo = Ipx(i-1,j,k,n);
                hi = Imx(i  ,j,k,n);
            }

            Real uad = u_ad(i,j,k);
            auto bc = pbc[n];

            Godunov_trans_xbc_lo(i, j, k, n, q, lo, hi, uad, bc.lo(0),
                                 domain.loVect(), domain.hiVect(), false, false);
            Godunov_trans_xbc_hi(i, j, k, n, q, lo, hi, uad, bc.hi(0),
                                 domain.loVect(), domain.hiVect(), false, false);
            xlo(i,j,k,n) = lo;
            xhi(i,j,k,n) = hi;

            constexpr Real eps = 1e-6;

            Real st = (uad >= 0.) ? lo : hi;
            Real fu = (std::abs(uad) < eps) ? 0.0 : 1.0;
            Imx(i, j, k, n) = fu*st + (1.0 - fu) *0.5 * (hi + lo); // store xedge
        },
        yebox, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            Real lo, hi;
            if (l_use_forces_in_trans) {
                lo = Ipy(i,j-1,k,n) + 0.5*l_dt*f(i,j-1,k,n);
                hi = Imy(i,j  ,k,n) + 0.5*l_dt*f(i,j  ,k,n);
            } else {
                lo = Ipy(i,j-1,k,n);
                hi = Imy(i,j  ,k,n);
            }

            Real vad = v_ad(i,j,k);
            auto bc = pbc[n];

            Godunov_trans_ybc_lo(i, j, k, n, q, lo, hi, vad, bc.lo(1),
                                 domain.loVect(), domain.hiVect(), false, false);
            Godunov_trans_ybc_hi(i, j, k, n, q, lo, hi, vad, bc.hi(1),
                                 domain.loVect(), domain.hiVect(), false, false);
            ylo(i,j,k,n) = lo;
            yhi(i,j,k,n) = hi;

            constexpr Real eps = 1e-6;

            Real st = (vad >= 0.) ? lo : hi;
            Real fu = (std::abs(vad) < eps) ? 0.0 : 1.0;
            Imy(i, j, k, n) = fu*st + (1.0 - fu)*0.5*(hi + lo); // store yedge
        },
        zebox, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            Real lo, hi;
            if (l_use_forces_in_trans) {
                lo = Ipz(i,j,k-1,n) + 0.5*l_dt*f(i,j,k-1,n);
                hi = Imz(i,j,k  ,n) + 0.5*l_dt*f(i,j,k  ,n);
            } else {
                lo = Ipz(i,j,k-1,n);
                hi = Imz(i,j,k  ,n);
            }

            Real wad = w_ad(i,j,k);
            auto bc = pbc[n];
            Godunov_trans_zbc_lo(i, j, k, n, q, lo, hi, wad, bc.lo(2),
                                 domain.loVect(), domain.hiVect(), false, false);
            Godunov_trans_zbc_hi(i, j, k, n, q, lo, hi, wad, bc.hi(2),
                                 domain.loVect(), domain.hiVect(), false, false);

            zlo(i,j,k,n) = lo;
            zhi(i,j,k,n) = hi;

            constexpr Real eps = 1e-6;

            Real st = (wad >= 0.) ? lo : hi;
            Real fu = (std::abs(wad) < eps) ? 0.0 : 1.0;
            Imz(i, j, k, n) = fu*st + (1.0 - fu)*0.5*(hi + lo); // store zedge
        });

    Array4<Real> xedge = Imx;
    Array4<Real> yedge = Imy;
    Array4<Real> zedge = Imz;

    Array4<Real> divu = makeArray4(Ipx.dataPtr(), grow(bx,1), 1);
    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
        divu(i,j,k) = 0.0;
    });

    // We can reuse the space in Ipy and Ipz.

    //
    // X-Flux
    //
    Box xbxtmp = Box(xbx).enclosedCells().grow(0,1);
    Array4<Real> yzlo = makeArray4(Ipy.dataPtr(), amrex::surroundingNodes(xbxtmp,1),ncomp);
    Array4<Real> zylo = makeArray4(Ipz.dataPtr(), amrex::surroundingNodes(xbxtmp,2),ncomp);
    // Add d/dy term to z-faces
    // Start with {zlo,zhi} --> {zylo, zyhi} and upwind using w_ad to {zylo}
    // Add d/dz to y-faces
    // Start with {ylo,yhi} --> {yzlo, yzhi} and upwind using v_ad to {yzlo}
    amrex::ParallelFor(
    Box(zylo), ncomp,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        const auto bc = pbc[n]; 
        Real l_zylo, l_zyhi;
        Godunov_corner_couple_zy(l_zylo, l_zyhi,
                                 i, j, k, n, l_dt, dy, iconserv[n],
                                 zlo(i,j,k,n), zhi(i,j,k,n),
                                 q, divu, v_ad, yedge);

        Real wad = w_ad(i,j,k);
        Godunov_trans_zbc_lo(i, j, k, n, q, l_zylo, l_zyhi, wad, bc.lo(2),
                             domain.loVect(), domain.hiVect(), false, true);  
        Godunov_trans_zbc_hi(i, j, k, n, q, l_zylo, l_zyhi, wad, bc.hi(2),
                             domain.loVect(), domain.hiVect(), false, true);  

        constexpr Real eps = 1e-6;

        Real st = (wad >= 0.) ? l_zylo : l_zyhi;
        Real fu = (std::abs(wad) < eps) ? 0.0 : 1.0;
        zylo(i,j,k,n) = fu*st + (1.0 - fu) *0.5 * (l_zyhi + l_zylo);
    },
    Box(yzlo), ncomp,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        const auto bc = pbc[n];
        Real l_yzlo, l_yzhi;
        Godunov_corner_couple_yz(l_yzlo, l_yzhi,
                                 i, j, k, n, l_dt, dz, iconserv[n],
                                 ylo(i,j,k,n), yhi(i,j,k,n),
                                 q, divu, w_ad, zedge);

        Real vad = v_ad(i,j,k);
        Godunov_trans_ybc_lo(i, j, k, n, q, l_yzlo, l_yzhi, vad, bc.lo(1),
                             domain.loVect(), domain.hiVect(), false, true);
        Godunov_trans_ybc_hi(i, j, k, n, q, l_yzlo, l_yzhi, vad, bc.hi(1),
                             domain.loVect(), domain.hiVect(), false, true);

        constexpr Real eps = 1e-6;

        Real st = (vad >= 0.) ? l_yzlo : l_yzhi;
        Real fu = (std::abs(vad) < eps) ? 0.0 : 1.0;
        yzlo(i,j,k,n) = fu*st + (1.0 - fu) *0.5 * (l_yzhi + l_yzlo);
    });
    //
    amrex::ParallelFor(xbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        constexpr int n = 0;
        auto bc = pbc[n];
        Real stl = xlo(i,j,k,n) - (0.25*l_dt/dy)*(v_ad(i-1,j+1,k)+v_ad(i-1,j,k))*
                                  (yzlo(i-1,j+1,k,n) - yzlo(i-1,j,k,n))
                                - (0.25*l_dt/dz)*(w_ad(i-1,j,k+1)+w_ad(i-1,j,k))*
                                  (zylo(i-1,j,k+1,n) - zylo(i-1,j,k,n));
        Real sth = xhi(i,j,k,n) - (0.25*l_dt/dy)*(v_ad(i,j+1,k)+v_ad(i,j,k))*
                                  (yzlo(i,j+1,k,n) - yzlo(i,j,k,n))
                                - (0.25*l_dt/dz)*(w_ad(i,j,k+1)+w_ad(i,j,k))*
                                  (zylo(i,j,k+1,n) - zylo(i,j,k,n));

        if (!l_use_forces_in_trans)
        {
            stl += 0.5 * l_dt * f(i-1,j,k,n);
            sth += 0.5 * l_dt * f(i  ,j,k,n);
        }

        constexpr Real eps = 1e-6;

        Godunov_cc_xbc(i, j, k, n, q, stl, sth, u_ad, bc.lo(0), bc.hi(0), dlo.x, dhi.x);

        Real st = ( (stl+sth) >= 0.) ? stl : sth;
        bool ltm = ( (stl <= 0. && sth >= 0.) || (std::abs(stl+sth) < eps) );
        qx(i,j,k) = ltm ? 0. : st;
    });
}

void incflo::predict_godunov (int lev, Real time, MultiFab& u_mac, MultiFab& v_mac,
                              MultiFab& w_mac, MultiFab const& vel, MultiFab const& rho,
                              MultiFab const& vel_forces)
{
    const int ncomp = AMREX_SPACEDIM;
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        FArrayBox scratch;
        for (MFIter mfi(vel,TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const& bx = mfi.tilebox();
            Box const& bxg1 = amrex::grow(bx,1);
            Box const& xbx = mfi.nodaltilebox(0);
            Box const& ybx = mfi.nodaltilebox(1);
            Box const& zbx = mfi.nodaltilebox(2);

            Array4<Real> const& a_umac = u_mac.array(mfi);
            Array4<Real> const& a_vmac = v_mac.array(mfi);
            Array4<Real> const& a_wmac = w_mac.array(mfi);
            Array4<Real const> const& a_vel = vel.const_array(mfi);
            Array4<Real const> const& a_f = vel_forces.const_array(mfi);

            scratch.resize(bxg1, ncomp*12+3);
//            Elixir eli = scratch.elixir(); // not needed because of streamSynchronize later
            Real* p = scratch.dataPtr();

            Array4<Real> Imx = makeArray4(p,bxg1,ncomp);
            p +=         Imx.size();
            Array4<Real> Ipx = makeArray4(p,bxg1,ncomp);
            p +=         Ipx.size();
            Array4<Real> Imy = makeArray4(p,bxg1,ncomp);
            p +=         Imy.size();
            Array4<Real> Ipy = makeArray4(p,bxg1,ncomp);
            p +=         Ipy.size();
            Array4<Real> Imz = makeArray4(p,bxg1,ncomp);
            p +=         Imz.size();
            Array4<Real> Ipz = makeArray4(p,bxg1,ncomp);
            p +=         Ipz.size();
            Array4<Real> u_ad = makeArray4(p,Box(bx).grow(1,1).grow(2,1).surroundingNodes(0),1);
            p +=         u_ad.size();
            Array4<Real> v_ad = makeArray4(p,Box(bx).grow(0,1).grow(2,1).surroundingNodes(1),1);
            p +=         v_ad.size();
            Array4<Real> w_ad = makeArray4(p,Box(bx).grow(0,1).grow(1,1).surroundingNodes(2),1);
            p +=         w_ad.size();

            make_ppm_integrals (lev, bxg1, AMREX_SPACEDIM, Imx, Ipx, Imy, Ipy, Imz, Ipz, a_vel, a_vel);

            make_trans_velocities(lev, Box(u_ad), Box(v_ad), Box(w_ad),
                                  u_ad, v_ad, w_ad,
                                  Imx, Ipx, Imy, Ipy, Imz, Ipz, a_vel, a_f);

            predict_godunov_on_box(lev, bx, ncomp, xbx, ybx, zbx, a_umac, a_vmac, a_wmac,
                                   a_vel, u_ad, v_ad, w_ad, Imx, Ipx, Imy, Ipy, Imz, Ipz, a_f, p);

            Gpu::streamSynchronize();  // otherwise we might be using too much memory
        }
    }

    VisMF::Write(u_mac, "u_mac");

    amrex::Abort("end of predict_godunov");
}