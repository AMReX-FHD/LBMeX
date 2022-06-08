#ifndef LBM_H_
#define LBM_H_

#include "d3q19.H"

Real tau = 0.9;

inline Array<Real,ncomp> fequilibrium(const Real& rho, const RealVect& u) {
  Array<Real,ncomp> fEq;
  for (int i=0; i<ncomp; ++i) {
    Real uc = u.dotProduct(c[i]);
    fEq[i] = w[i] * rho * ( 1. + uc/cs2 + (uc*uc - u.dotProduct(u)*cs2)/(2.*cs2*cs2) );
  }
  return fEq;
}

inline Array<Real,ncomp>& relax_moments(Array<Real,ncomp>& m) {
  Array<Real,ncomp> mEq;
  Real gamma = 1. - 1./tau;
  mEq = mequilibrium(m);
  for (int i=0; i<ncomp; ++i) {
    m[i] = mEq[i] + gamma * (m[i] - mEq[i]);
  }
  return m;
}

// push scheme
inline void collide_stream(int x, int y, int z, const Array4<Real>& fOld, const Array4<Real>& fNew) {
  Array<Real,ncomp> f, m;
  for (int i=0; i<ncomp; ++i) { // read populations from current site
    f[i] = fOld(x,y,z,i);
  }
  m = moments(f);
  m = relax_moments(m);
  f = populations(m);
  for (int i=0; i<ncomp; ++i) { // push populations to neighbors
    int xp = x + c[i][0];
    int yp = y + c[i][1];
    int zp = z + c[i][2];
    fNew(xp,yp,zp,i) = f[i];
  }
}

// pull scheme
inline void stream_collide(int x, int y, int z, const Array4<Real>& fOld, const Array4<Real>& fNew) {
  Array<Real,ncomp> f, m;
  for (int i=0; i<ncomp; ++i) { // pull populations from neighbors
    int xm = x - c[i][0];
    int ym = y - c[i][1];
    int zm = z - c[i][2];
    f[i] = fOld(xm,ym,zm,i);
  }
  m = moments(f);
  m = relax_moments(m);
  f = populations(m);
  for (int i=0; i<ncomp; ++i) { // write populations to current site
    fNew(x,y,z,i) = f[i];
  }
}

#endif
