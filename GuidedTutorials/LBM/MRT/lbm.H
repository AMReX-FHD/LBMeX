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

inline void relax_moments(int x, int y, int z, const Array4<Real>& m) {
  const Real gamma = 1. - 1./tau;
  const Array<Real,ncomp> mEq = mequilibrium(x, y, z, m);
  for (int i=0; i<ncomp; ++i) {
    m(x,y,z,i) = mEq[i] + gamma * (m(x,y,z,i) - mEq[i]);
  }
}

inline void thermalize_moments(int x, int y, int z, const Array4<Real>& m, RandomEngine const& engine) {
  const Real gamma = 1. - 1./tau;
  const Real rho = m(x,y,z,0);
  const Real phi = sqrt(rho*temperature/cs2*(1.-gamma*gamma));
  for (int i=4; i<ncomp; ++i) {
    m(x,y,z,i) += sqrt(b[i])*phi*RandomNormal(0., 1., engine);
  }
}

inline void collide(int x, int y, int z, const Array4<Real>& f, const Array4<Real>& m, RandomEngine const& engine) {
  moments(x, y, z, f, m);
  relax_moments(x, y, z, m);
  thermalize_moments(x, y, z, m, engine);
  populations(x, y, z, f, m);
}

// pull scheme
inline void stream_collide(int x, int y, int z, const Array4<Real>& mom, const Array4<Real>& fOld, const Array4<Real>& fNew, RandomEngine const& engine) {
  for (int i=0; i<ncomp; ++i) { // pull populations from neighbors
    int xm = x - c[i][0];
    int ym = y - c[i][1];
    int zm = z - c[i][2];
    fNew(x,y,z,i) = fOld(xm,ym,zm,i);
  }
  collide(x, y, z, fNew, mom, engine); // collide populations on current site
}

#endif
