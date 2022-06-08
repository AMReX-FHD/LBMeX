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

inline Array<Real,ncomp> moments(const Array<Real,ncomp>& fsite) {
  Array<Real,ncomp> m;

  Real f;
  Real mc0, mc1, mc2;
  Real mx1, my1, mz1, mx2, my2, mz2, mx3, my3, mz3;
  Real mxy, mxz, myz, mxx1, myy1, mzz1, mxx2, myy2, mzz2;

  f = fsite[ 0]; mc0  = f;
  f = fsite[ 1]; mx1  = f; mxx1  = f;
  f = fsite[ 2]; mx1 -= f; mxx1 += f;
  f = fsite[ 3]; my1  = f; myy1  = f;
  f = fsite[ 4]; my1 -= f; myy1 += f;
  f = fsite[ 5]; mz1  = f; mzz1  = f;
  f = fsite[ 6]; mz1 -= f; mzz1 += f;
  f = fsite[ 7]; mx2  = f; my3  = f; mxy  = f; mzz2  = f;
  f = fsite[ 8]; mx2 -= f; my3 -= f; mxy += f; mzz2 += f;
  f = fsite[ 9]; mx2 += f; my3 -= f; mxy -= f; mzz2 += f;
  f = fsite[10]; mx2 -= f; my3 += f; mxy -= f; mzz2 += f;
  f = fsite[11]; mz2  = f; mx3  = f; mxz  = f; mxx2  = f;
  f = fsite[12]; mz2 -= f; mx3 -= f; mxz += f; mxx2 += f;
  f = fsite[13]; mz2 -= f; mx3 += f; mxz -= f; mxx2 += f;
  f = fsite[14]; mz2 += f; mx3 -= f; mxz -= f; mxx2 += f;
  f = fsite[15]; my2  = f; mz3  = f; myz  = f; myy2  = f;
  f = fsite[16]; my2 -= f; mz3 -= f; myz += f; myy2 += f;
  f = fsite[17]; my2 += f; mz3 -= f; myz -= f; myy2 += f;
  f = fsite[18]; my2 -= f; mz3 += f; myz -= f; myy2 += f;

  mc1 = mxx1 + myy1 + mzz1;
  mc2 = mxx2 + myy2 + mzz2;

  m[ 0] = mc0 + mc1 + mc2;
  m[ 1] = mx1 + mx2 + mx3;
  m[ 2] = my1 + my2 + my3;
  m[ 3] = mz1 + mz2 + mz3;
  m[ 4] = mc2 - mc0;
  m[ 5] = mxx1 - myy1 + mxx2 - myy2;
  m[ 6] = mc1 - 3.*mzz1 - mc2 + 3.*mzz2;
  m[ 7] = mxy;
  m[ 8] = mxz;
  m[ 9] = myz;
  m[10] = m[1] - 3.*mx1;
  m[11] = m[2] - 3.*my1;
  m[12] = m[3] - 3.*mz1;
  m[13] = mx2 - mx3;
  m[14] = my3 - my2;
  m[15] = mz2 - mz3;
  m[16] = m[0] - 3.*mc1;
  m[17] = myy1 - mxx1 + mxx2 - myy2;
  m[18] = 3.*mzz1 - mc1 + 3.*mzz2 - mc2;

  return m;
}

inline Array<Real,ncomp> populations(const Array<Real,ncomp>& msite) {
  Array<Real,ncomp> m, f;

  Real mc0, mc1, mc2;
  Real mx1, my1, mz1, mx2, my2, mz2, mx3, my3, mz3;
  Real mxx1, myy1, mzz1, mxy, mxz, myz, mxy2, mxz2, myz2;

  m[ 0] = msite[ 0] / 36.;
  m[ 1] = msite[ 1] / 12.;
  m[ 2] = msite[ 2] / 12.;
  m[ 3] = msite[ 3] / 12.;
  m[ 4] = msite[ 4] / 24.;
  m[ 5] = msite[ 5] / 16.;
  m[ 6] = msite[ 6] / 48.;
  m[ 7] = msite[ 7] / 4.;
  m[ 8] = msite[ 8] / 4.;
  m[ 9] = msite[ 9] / 4.;
  m[10] = msite[10] / 24.;
  m[11] = msite[11] / 24.;
  m[12] = msite[12] / 24.;
  m[13] = msite[13] / 8.;
  m[14] = msite[14] / 8.;
  m[15] = msite[15] / 8.;
  m[16] = msite[16] / 72.;
  m[17] = msite[17] / 16.;
  m[18] = msite[18] / 48.;

  mc0 = 12.*(m[0] - m[4] + m[16]);
  mc1 =  2.*(m[0] - 2.*m[16]);
  mc2 = m[0] + m[4] + m[16];

  mx1 = 2.*(m[1] - 2.*m[10]);
  my1 = 2.*(m[2] - 2.*m[11]);
  mz1 = 2.*(m[3] - 2.*m[12]);

  mx2 = m[1] + m[10] + m[13];
  my2 = m[2] + m[11] - m[14];
  mz2 = m[3] + m[12] + m[15];

  mx3 = m[1] + m[10] - m[13];
  my3 = m[2] + m[11] + m[14];
  mz3 = m[3] + m[12] - m[15];

  mxx1 = mc1 + 2.*(m[5] + m[6]) - 2.*(m[17] + m[18]);
  myy1 = mc1 - 2.*(m[5] - m[6]) + 2.*(m[17] - m[18]);
  mzz1 = mc1 - 4.*(m[6] - m[18]);
  
  mxy2 = mc2 + 2.*(m[6] + m[18]);
  mxz2 = mc2 + (m[5] + m[17]) - (m[6] + m[18]);
  myz2 = mc2 - (m[5] + m[17]) - (m[6] + m[18]);

  mxy = m[7];
  mxz = m[8];
  myz = m[9];

  f[ 0] = mc0;
  f[ 1] = mxx1 + mx1;
  f[ 2] = mxx1 - mx1;
  f[ 3] = myy1 + my1;
  f[ 4] = myy1 - my1;
  f[ 5] = mzz1 + mz1;
  f[ 6] = mzz1 - mz1;
  f[ 7] = mxy2 + mx2 + my3 + mxy;
  f[ 8] = mxy2 - mx2 - my3 + mxy;
  f[ 9] = mxy2 + mx2 - my3 - mxy;
  f[10] = mxy2 - mx2 + my3 - mxy;
  f[11] = mxz2 + mz2 + mx3 + mxz;
  f[12] = mxz2 - mz2 - mx3 + mxz;
  f[13] = mxz2 - mz2 + mx3 - mxz;
  f[14] = mxz2 + mz2 - mx3 - mxz;
  f[15] = myz2 + my2 + mz3 + myz;
  f[16] = myz2 - my2 - mz3 + myz;
  f[17] = myz2 + my2 - mz3 - myz;
  f[18] = myz2 - my2 + mz3 - myz;

  return f;
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
