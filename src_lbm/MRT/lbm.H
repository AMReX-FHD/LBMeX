#ifndef LBM_H_
#define LBM_H_

#include "d3q19.H"

AMREX_GPU_MANAGED Real density = 1.0;
AMREX_GPU_MANAGED Real temperature = 0.0;
AMREX_GPU_MANAGED Real tau = 0.9;

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Array1D<Real,0,ncomp> fequilibrium(const Real& rho, const RealVect& u) {
  Array1D<Real,0,ncomp> fEq;
  for (int i=0; i<ncomp; ++i) {
    Real uc = u.dotProduct(c[i]);
    fEq(i) = w[i] * rho * ( 1. + uc/cs2 + (uc*uc - u.dotProduct(u)*cs2)/(2.*cs2*cs2) );
  }
  return fEq;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Array1D<Real,0,ncomp> relax_moments(int x, int y, int z, const Array4<Real>& m) {
  const Real gamma = 1. - 1./tau;
  const Array1D<Real,0,ncomp> mEq = mequilibrium(x, y, z, m);
  Array1D<Real,0,ncomp> mom;
  for (int i=0; i<ncomp; ++i) {
    mom(i) = m(x,y,z,i) = mEq(i) + gamma * (m(x,y,z,i) - mEq(i));
  }
  return mom;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Array1D<Real,0,ncomp> thermalize_moments(int x, int y, int z, const Array4<Real>& m, RandomEngine const& engine) {
  const Real gamma = 1. - 1./tau;
  const Real rho = m(x,y,z,0);
  const Real phi = sqrt(rho*temperature/cs2*(1.-gamma*gamma));
  Array1D<Real,0,ncomp> r = {}; // {} to value initialize r
  for (int i=4; i<ncomp; ++i) {
    r(i) = sqrt(b[i])*phi*RandomNormal(0., 1., engine);
    m(x,y,z,i) += r(i);
  }
  return r;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void collide(int x, int y, int z, const Array4<Real>& f, const Array4<Real>& msite, const Array4<Real>& hsite, const Array4<Real>& hEqsite, RandomEngine const& engine) {
  const Real gamma = 1. - 1./tau;
  Array1D<Real,0,ncomp> m, h, hEq, r;
  // collision step
  m = moments(x, y, z, f, msite);
  relax_moments(x, y, z, msite);
  r = thermalize_moments(x, y, z, msite, engine);
  // record hydrodynamic variables
  h = hydrovars(m);
  r = hydrovars(r);
  hEq = hydrovars(mequilibrium(h(0), { h(1), h(2), h(3) }));
  for (int i=0; i<=AMREX_SPACEDIM; ++i) hsite(x,y,z,i) = h(i);
  for (int i=AMREX_SPACEDIM+1; i<(AMREX_SPACEDIM+1)*(AMREX_SPACEDIM+2)/2; ++i) {
    // note that h is pre-collision
    hEqsite(x,y,z,i) = hEq(i);
    hsite(x,y,z,i) = hEq(i) + 0.5*(1.+gamma)*(h(i)-hEq(i)) + 0.5*r(i);
  }
  populations(x, y, z, f, msite);
}

// pull scheme
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void stream_collide(int x, int y, int z, const Array4<Real>& fOld, const Array4<Real>& fNew,  const Array4<Real>& m, const Array4<Real>& h, const Array4<Real>& hEq, RandomEngine const& engine) {
  for (int i=0; i<ncomp; ++i) { // pull populations from neighbors
    int xm = x - c[i][0];
    int ym = y - c[i][1];
    int zm = z - c[i][2];
    fNew(x,y,z,i) = fOld(xm,ym,zm,i);
  }
  collide(x, y, z, fNew, m, h, hEq, engine); // collide populations on current site
}

#endif
