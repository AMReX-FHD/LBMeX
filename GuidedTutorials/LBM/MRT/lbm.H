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

inline Array<Real,ncomp>& relax_moments(Array<Real,ncomp>& m) {
  const Real gamma = 1. - 1./tau;
  Array<Real,ncomp> mEq;
  mEq = mequilibrium(m);
  for (int i=0; i<ncomp; ++i) {
    m[i] = mEq[i] + gamma * (m[i] - mEq[i]);
  }
  return m;
}

inline Array<Real,ncomp>& thermalize_moments(Array<Real,ncomp>& m, RandomEngine const& engine) {
  const Real gamma = 1. - 1./tau;
  const Real mu = temperature/cs2;
  const Real phi = sqrt(m[0]*mu*(1.-gamma*gamma));

  /* stress modes */
  m[4] += sqrt(b[4])*phi*RandomNormal(0.,1.,engine);
  m[5] += sqrt(b[5])*phi*RandomNormal(0.,1.,engine);
  m[6] += sqrt(b[6])*phi*RandomNormal(0.,1.,engine);
  m[7] += sqrt(b[7])*phi*RandomNormal(0.,1.,engine);
  m[8] += sqrt(b[8])*phi*RandomNormal(0.,1.,engine);
  m[9] += sqrt(b[9])*phi*RandomNormal(0.,1.,engine);

  /* kinetic modes */
  m[10] += sqrt(b[10])*phi*RandomNormal(0.,1.,engine);
  m[11] += sqrt(b[11])*phi*RandomNormal(0.,1.,engine);
  m[12] += sqrt(b[12])*phi*RandomNormal(0.,1.,engine);
  m[13] += sqrt(b[13])*phi*RandomNormal(0.,1.,engine);
  m[14] += sqrt(b[14])*phi*RandomNormal(0.,1.,engine);
  m[15] += sqrt(b[15])*phi*RandomNormal(0.,1.,engine);
  m[16] += sqrt(b[16])*phi*RandomNormal(0.,1.,engine);
  m[17] += sqrt(b[17])*phi*RandomNormal(0.,1.,engine);
  m[18] += sqrt(b[18])*phi*RandomNormal(0.,1.,engine);

  return m;
}

inline Array<Real,ncomp> collide(Array<Real,ncomp>& f, RandomEngine const& engine) {
  Array<Real,ncomp> m;
  m = moments(f);
  m = relax_moments(m);
  m = thermalize_moments(m, engine);
  f = populations(m);
  return m;
}

// push scheme
inline void collide_stream(int x, int y, int z, const Array4<Real>& fOld, const Array4<Real>& fNew, RandomEngine const& engine) {
  Array<Real,ncomp> f;
  for (int i=0; i<ncomp; ++i) { // read populations from current site
    f[i] = fOld(x,y,z,i);
  }
  collide(f, engine);
  for (int i=0; i<ncomp; ++i) { // push populations to neighbors
    int xp = x + c[i][0];
    int yp = y + c[i][1];
    int zp = z + c[i][2];
    fNew(xp,yp,zp,i) = f[i];
  }
}

// pull scheme
inline void stream_collide(int x, int y, int z, const Array4<Real>& mom, const Array4<Real>& fOld, const Array4<Real>& fNew, RandomEngine const& engine) {
  Array<Real,ncomp> f, m;
  for (int i=0; i<ncomp; ++i) { // pull populations from neighbors
    int xm = x - c[i][0];
    int ym = y - c[i][1];
    int zm = z - c[i][2];
    f[i] = fOld(xm,ym,zm,i);
  }
  m = collide(f, engine);
  for (int i=0; i<ncomp; ++i) { // write populations to current site
    fNew(x,y,z,i) = f[i];
    mom(x,y,z,i) = m[i];
  }
}

#endif
