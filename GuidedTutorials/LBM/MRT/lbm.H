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
  Real rho, u[3], m4eq, m5eq, m6eq, m7eq, m8eq, m9eq;
  Real gamma = 1. - 1./tau;

  /* mass density */
  rho = m[0];

  /* momentum density */
  u[0] = m[1]/rho;
  u[1] = m[2]/rho;
  u[2] = m[3]/rho;

  /* equilibrium part of the stress modes */
  m4eq = (u[0]*u[0] + u[1]*u[1] + u[2]*u[2])*rho;
  m5eq = (u[0]*u[0] - u[1]*u[1])*rho;
  m6eq = (u[0]*u[0] + u[1]*u[1] - 2.*u[2]*u[2])*rho;
  m7eq = u[0]*u[1]*rho;
  m8eq = u[0]*u[2]*rho;
  m9eq = u[1]*u[2]*rho;
  
  /* relax stress modes */  
  m[4] = m4eq + gamma * (m[4] - m4eq);
  m[5] = m5eq + gamma * (m[5] - m5eq);
  m[6] = m6eq + gamma * (m[6] - m6eq);
  m[7] = m7eq + gamma * (m[7] - m7eq);
  m[8] = m8eq + gamma * (m[8] - m8eq);
  m[9] = m9eq + gamma * (m[9] - m9eq);
  
  /* ghost modes have no equilibrium part due to orthogonality */
  m[10] = gamma * m[10];
  m[11] = gamma * m[11];
  m[12] = gamma * m[12];
  m[13] = gamma * m[13];
  m[14] = gamma * m[14];
  m[15] = gamma * m[15];
  m[16] = gamma * m[16];
  m[17] = gamma * m[17];
  m[18] = gamma * m[18];

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
