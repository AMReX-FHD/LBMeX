#ifndef LBM_H_
#define LBM_H_

#include "d3q19.H"

Real density = 1.0;
Real temperature = 0.0;
Real tau = 0.9;

inline Array<Real,ncomp> fequilibrium(const Real& rho, const RealVect& u) {
  Array<Real,ncomp> fEq;
  for (int i=0; i<ncomp; ++i) {
    Real uc = u.dotProduct(c[i]);
    fEq[i] = w[i] * rho * ( 1. + uc/cs2 + (uc*uc - u.dotProduct(u)*cs2)/(2.*cs2*cs2) );
  }
  return fEq;
}

inline Array<Real,ncomp> relax_moments(int x, int y, int z, const Array4<Real>& m) {
  const Real gamma = 1. - 1./tau;
  const Array<Real,ncomp> mEq = mequilibrium(x, y, z, m);
  Array<Real,ncomp> mom;
  for (int i=0; i<ncomp; ++i) {
    mom[i] = m(x,y,z,i) = mEq[i] + gamma * (m(x,y,z,i) - mEq[i]);
  }
  return mom;
}

inline Array<Real,ncomp> thermalize_moments(int x, int y, int z, const Array4<Real>& m, RandomEngine const& engine) {
  const Real gamma = 1. - 1./tau;
  const Real rho = m(x,y,z,0);
  const Real phi = sqrt(rho*temperature/cs2*(1.-gamma*gamma));
  Array<Real,ncomp> r = {}; // {} to value initialize r
  for (int i=4; i<ncomp; ++i) {
    r[i] = sqrt(b[i])*phi*RandomNormal(0., 1., engine);
    m(x,y,z,i) += r[i];
  }
  return r;
}

inline void collide(int x, int y, int z, const Array4<Real>& f, const Array4<Real>& msite, const Array4<Real>& hsite, RandomEngine const& engine) {
  const Real gamma = 1. - 1./tau;
  Array<Real,ncomp> m, h, r;
  moments(x, y, z, f, msite);
  m = relax_moments(x, y, z, msite);
  r = thermalize_moments(x, y, z, msite, engine);
  h = hydrovars(m);
  r = hydrovars(r);
  for (int i=0; i<=AMREX_SPACEDIM; ++i) hsite(x,y,z,i) = h[i];
  for (int i=AMREX_SPACEDIM+1; i<(AMREX_SPACEDIM+1)*(AMREX_SPACEDIM+2)/2; ++i) {
    // note that r was not added to m and h
    // => + 0.5*r[i] instead of - 0.5/gamma*r[i]
    hsite(x,y,z,i) = 0.5*(1.+1./gamma)*h[i] + 0.5*r[i];
  }
  populations(x, y, z, f, msite);
}

// pull scheme
inline void stream_collide(int x, int y, int z, const Array4<Real>& m, const Array4<Real>& fOld, const Array4<Real>& fNew, const Array4<Real>& h, RandomEngine const& engine) {
  for (int i=0; i<ncomp; ++i) { // pull populations from neighbors
    int xm = x - c[i][0];
    int ym = y - c[i][1];
    int zm = z - c[i][2];
    fNew(x,y,z,i) = fOld(xm,ym,zm,i);
  }
  collide(x, y, z, fNew, m, h, engine); // collide populations on current site
}

#endif
