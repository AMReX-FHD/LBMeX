#ifndef LBM_H_
#define LBM_H_

#include "LBM_d3q19.H"

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

inline void collide(int x, int y, int z, const Array4<Real>& f, const Array4<Real>& msite, const Array4<Real>& hsite, RandomEngine const& engine) {
  moments(x, y, z, f, msite);
  relax_moments(x, y, z, msite);
  populations(x, y, z, f, msite);
}

// stream from \vec{x}-\vec{c}_i to \vec{x} (pull scheme)
inline void stream_pull(int x, int y, int z, const Array4<Real>& fOld, const Array4<Real>& fNew) {
  for (int i=0; i<ncomp; ++i) { // pull populations from neighbors
    int xm = x - c[i][0];
    int ym = y - c[i][1];
    int zm = z - c[i][2];
    fNew(x,y,z,i) = fOld(xm,ym,zm,i);
  }
}

// pull scheme
inline void stream_collide(int x, int y, int z, const Array4<Real>& m, const Array4<Real>& fOld, const Array4<Real>& fNew, const Array4<Real>& h, RandomEngine const& engine) {
  stream_pull(x, y, z, fOld, fNew);     // pull populations from neighbors
  collide(x, y, z, fNew, m, h, engine); // collide populations on current site
}

#endif
