#ifndef LBM_D3Q19_H_
#define LBM_D3Q19_H_

const int ncomp = 19;

const Real cs2 = 1./3.;

const IntVect c[19] = {
  {  0,  0,  0 },
  {  1,  0,  0 },
  { -1,  0,  0 },
  {  0,  1,  0 },
  {  0, -1,  0 },
  {  0,  0,  1 },
  {  0,  0, -1 },
  {  1,  1,  0 },
  { -1, -1,  0 },
  {  1, -1,  0 },
  { -1,  1,  0 },
  {  1,  0,  1 },
  { -1,  0, -1 },
  {  1,  0, -1 },
  { -1,  0,  1 },
  {  0,  1,  1 },
  {  0, -1, -1 },
  {  0,  1, -1 },
  {  0, -1,  1 }
};

const Real w[19] = {
  1./3.,
  1./18.,
  1./18.,
  1./18.,
  1./18.,
  1./18.,
  1./18.,
  1./36.,
  1./36.,
  1./36.,
  1./36.,
  1./36.,
  1./36.,
  1./36.,
  1./36.,
  1./36.,
  1./36.,
  1./36.,
  1./36.
};

const Real b[19] = {
  1.0,
  1./3.,
  1./3.,
  1./3.,
  2./3.,
  4./9.,
  4./3.,
  1./9.,
  1./9.,
  1./9.,
  2./3.,
  2./3.,
  2./3.,
  2./9.,
  2./9.,
  2./9.,
  2.0,
  4./9.,
  4./3.
};

inline void moments(int x, int y, int z, const Array4<Real>& fsite, const Array4<Real>& m) {
  Real f;
  Real mc0, mc1, mc2;
  Real mx1, my1, mz1, mx2, my2, mz2, mx3, my3, mz3;
  Real mxy, mxz, myz, mxx1, myy1, mzz1, mxx2, myy2, mzz2;

  f = fsite(x,y,z, 0); mc0  = f;
  f = fsite(x,y,z, 1); mx1  = f; mxx1  = f;
  f = fsite(x,y,z, 2); mx1 -= f; mxx1 += f;
  f = fsite(x,y,z, 3); my1  = f; myy1  = f;
  f = fsite(x,y,z, 4); my1 -= f; myy1 += f;
  f = fsite(x,y,z, 5); mz1  = f; mzz1  = f;
  f = fsite(x,y,z, 6); mz1 -= f; mzz1 += f;
  f = fsite(x,y,z, 7); mx2  = f; my3  = f; mxy  = f; mzz2  = f;
  f = fsite(x,y,z, 8); mx2 -= f; my3 -= f; mxy += f; mzz2 += f;
  f = fsite(x,y,z, 9); mx2 += f; my3 -= f; mxy -= f; mzz2 += f;
  f = fsite(x,y,z,10); mx2 -= f; my3 += f; mxy -= f; mzz2 += f;
  f = fsite(x,y,z,11); mz2  = f; mx3  = f; mxz  = f; mxx2  = f;
  f = fsite(x,y,z,12); mz2 -= f; mx3 -= f; mxz += f; mxx2 += f;
  f = fsite(x,y,z,13); mz2 -= f; mx3 += f; mxz -= f; mxx2 += f;
  f = fsite(x,y,z,14); mz2 += f; mx3 -= f; mxz -= f; mxx2 += f;
  f = fsite(x,y,z,15); my2  = f; mz3  = f; myz  = f; myy2  = f;
  f = fsite(x,y,z,16); my2 -= f; mz3 -= f; myz += f; myy2 += f;
  f = fsite(x,y,z,17); my2 += f; mz3 -= f; myz -= f; myy2 += f;
  f = fsite(x,y,z,18); my2 -= f; mz3 += f; myz -= f; myy2 += f;

  mc1 = mxx1 + myy1 + mzz1;
  mc2 = mxx2 + myy2 + mzz2;

  m(x,y,z, 0) = mc0 + mc1 + mc2;
  m(x,y,z, 1) = mx1 + mx2 + mx3;
  m(x,y,z, 2) = my1 + my2 + my3;
  m(x,y,z, 3) = mz1 + mz2 + mz3;
  m(x,y,z, 4) = mc2 - mc0;
  m(x,y,z, 5) = mxx1 - myy1 + mxx2 - myy2;
  m(x,y,z, 6) = mc1 - 3.*mzz1 - mc2 + 3.*mzz2;
  m(x,y,z, 7) = mxy;
  m(x,y,z, 8) = mxz;
  m(x,y,z, 9) = myz;
  m(x,y,z,10) = m(x,y,z,1) - 3.*mx1;
  m(x,y,z,11) = m(x,y,z,2) - 3.*my1;
  m(x,y,z,12) = m(x,y,z,3) - 3.*mz1;
  m(x,y,z,13) = mx2 - mx3;
  m(x,y,z,14) = my3 - my2;
  m(x,y,z,15) = mz2 - mz3;
  m(x,y,z,16) = m(x,y,z,0) - 3.*mc1;
  m(x,y,z,17) = myy1 - mxx1 + mxx2 - myy2;
  m(x,y,z,18) = 3.*mzz1 - mc1 + 3.*mzz2 - mc2;

  return;
}

inline void populations(int x, int y, int z, const Array4<Real>& f, const Array4<Real>& msite) {
  Array<Real,ncomp> m;

  Real mc0, mc1, mc2;
  Real mx1, my1, mz1, mx2, my2, mz2, mx3, my3, mz3;
  Real mxx1, myy1, mzz1, mxy, mxz, myz, mxy2, mxz2, myz2;

  m[ 0] = msite(x,y,z, 0) / 36.;
  m[ 1] = msite(x,y,z, 1) / 12.;
  m[ 2] = msite(x,y,z, 2) / 12.;
  m[ 3] = msite(x,y,z, 3) / 12.;
  m[ 4] = msite(x,y,z, 4) / 24.;
  m[ 5] = msite(x,y,z, 5) / 16.;
  m[ 6] = msite(x,y,z, 6) / 48.;
  m[ 7] = msite(x,y,z, 7) / 4.;
  m[ 8] = msite(x,y,z, 8) / 4.;
  m[ 9] = msite(x,y,z, 9) / 4.;
  m[10] = msite(x,y,z,10) / 24.;
  m[11] = msite(x,y,z,11) / 24.;
  m[12] = msite(x,y,z,12) / 24.;
  m[13] = msite(x,y,z,13) / 8.;
  m[14] = msite(x,y,z,14) / 8.;
  m[15] = msite(x,y,z,15) / 8.;
  m[16] = msite(x,y,z,16) / 72.;
  m[17] = msite(x,y,z,17) / 16.;
  m[18] = msite(x,y,z,18) / 48.;

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

  f(x,y,z, 0) = mc0;
  f(x,y,z, 1) = mxx1 + mx1;
  f(x,y,z, 2) = mxx1 - mx1;
  f(x,y,z, 3) = myy1 + my1;
  f(x,y,z, 4) = myy1 - my1;
  f(x,y,z, 5) = mzz1 + mz1;
  f(x,y,z, 6) = mzz1 - mz1;
  f(x,y,z, 7) = mxy2 + mx2 + my3 + mxy;
  f(x,y,z, 8) = mxy2 - mx2 - my3 + mxy;
  f(x,y,z, 9) = mxy2 + mx2 - my3 - mxy;
  f(x,y,z,10) = mxy2 - mx2 + my3 - mxy;
  f(x,y,z,11) = mxz2 + mz2 + mx3 + mxz;
  f(x,y,z,12) = mxz2 - mz2 - mx3 + mxz;
  f(x,y,z,13) = mxz2 - mz2 + mx3 - mxz;
  f(x,y,z,14) = mxz2 + mz2 - mx3 - mxz;
  f(x,y,z,15) = myz2 + my2 + mz3 + myz;
  f(x,y,z,16) = myz2 - my2 - mz3 + myz;
  f(x,y,z,17) = myz2 + my2 - mz3 - myz;
  f(x,y,z,18) = myz2 - my2 + mz3 - myz;

  return;
}

inline Array<Real,ncomp> mequilibrium(const Real& rho, const RealVect& u) {
  Array<Real,ncomp> mEq;

  /* mass density */
  mEq[0] = rho;

  /* momentum density */
  mEq[1] = rho*u[0];
  mEq[2] = rho*u[1];
  mEq[3] = rho*u[2];

  /* equilibrium part of the stress modes */
  mEq[4] = (u[0]*u[0] + u[1]*u[1] + u[2]*u[2])*rho;
  mEq[5] = (u[0]*u[0] - u[1]*u[1])*rho;
  mEq[6] = (u[0]*u[0] + u[1]*u[1] - 2.*u[2]*u[2])*rho;
  mEq[7] = u[0]*u[1]*rho;
  mEq[8] = u[0]*u[2]*rho;
  mEq[9] = u[1]*u[2]*rho;

  /* ghost modes have no equilibrium part due to orthogonality */
  for (int i=10; i<ncomp; ++i) mEq[i] = 0.0;

  return mEq;
}

inline Array<Real,ncomp> mequilibrium(int x, int y, int z, const Array4<Real>& m) {
  const Real rho = m(x,y,z,0);
  const RealVect j = { m(x,y,z,1), m(x,y,z,2), m(x,y,z,3) };
  return mequilibrium(rho, j/rho);
}

#endif
