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
  {  0,  1,  1 },
  {  0, -1, -1 },
  {  0,  1, -1 },
  {  0, -1,  1 },
  {  1,  0,  1 },
  { -1,  0, -1 },
  {  1,  0, -1 },
  { -1,  0,  1 }
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
  4./3.,
  4./9.,
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
  4./3.,
  4./9.
};

inline Array<Real,ncomp> moments(int x, int y, int z, const Array4<Real>& fsite, const Array4<Real>& m) {
  Real f;
  Real mc0, mc1, mc2;
  Real mx1, my1, mz1, mx2, my2, mz2, mx3, my3, mz3;
  Real mxy, mxz, myz, mxx1, myy1, mzz1, mxx2, myy2, mzz2;
  Array<Real,ncomp> mom;

  f = fsite(x,y,z, 0); mc0  = f;
  f = fsite(x,y,z, 1); mx1  = f; mxx1  = f;
  f = fsite(x,y,z, 2); mx1 -= f; mxx1 += f;
  f = fsite(x,y,z, 3); my1  = f; myy1  = f;
  f = fsite(x,y,z, 4); my1 -= f; myy1 += f;
  f = fsite(x,y,z, 5); mz1  = f; mzz1  = f;
  f = fsite(x,y,z, 6); mz1 -= f; mzz1 += f;
  f = fsite(x,y,z, 7); mx2  = f; my3  = f; mxy  = f; mxx2  = f;
  f = fsite(x,y,z, 8); mx2 -= f; my3 -= f; mxy += f; mxx2 += f;
  f = fsite(x,y,z, 9); mx2 += f; my3 -= f; mxy -= f; mxx2 += f;
  f = fsite(x,y,z,10); mx2 -= f; my3 += f; mxy -= f; mxx2 += f;
  f = fsite(x,y,z,11); my2  = f; mz3  = f; myz  = f; myy2  = f;
  f = fsite(x,y,z,12); my2 -= f; mz3 -= f; myz += f; myy2 += f;
  f = fsite(x,y,z,13); my2 += f; mz3 -= f; myz -= f; myy2 += f;
  f = fsite(x,y,z,14); my2 -= f; mz3 += f; myz -= f; myy2 += f;
  f = fsite(x,y,z,15); mz2  = f; mx3  = f; mxz  = f; mzz2  = f;
  f = fsite(x,y,z,16); mz2 -= f; mx3 -= f; mxz += f; mzz2 += f;
  f = fsite(x,y,z,17); mz2 -= f; mx3 += f; mxz -= f; mzz2 += f;
  f = fsite(x,y,z,18); mz2 += f; mx3 -= f; mxz -= f; mzz2 += f;

  mc1 = mxx1 + myy1 + mzz1;
  mc2 = mxx2 + myy2 + mzz2;

  mom[ 0] = m(x,y,z, 0) = mc0 + mc1 + mc2;
  mom[ 1] = m(x,y,z, 1) = mx1 + mx2 + mx3;
  mom[ 2] = m(x,y,z, 2) = my1 + my2 + my3;
  mom[ 3] = m(x,y,z, 3) = mz1 + mz2 + mz3;
  mom[ 4] = m(x,y,z, 4) = mc2 - mc0;
  mom[ 5] = m(x,y,z, 5) = 3.*mxx1 - mc1 + mc2 - 3.*myy2;
  mom[ 6] = m(x,y,z, 6) = myy1 - mzz1 + mxx2 - mzz2;
  mom[ 7] = m(x,y,z, 7) = mxy;
  mom[ 8] = m(x,y,z, 8) = myz;
  mom[ 9] = m(x,y,z, 9) = mxz;
  mom[10] = m(x,y,z,10) = m(x,y,z,1) - 3.*mx1;
  mom[11] = m(x,y,z,11) = m(x,y,z,2) - 3.*my1;
  mom[12] = m(x,y,z,12) = m(x,y,z,3) - 3.*mz1;
  mom[13] = m(x,y,z,13) = mx2 - mx3;
  mom[14] = m(x,y,z,14) = my2 - my3;
  mom[15] = m(x,y,z,15) = mz2 - mz3;
  mom[16] = m(x,y,z,16) = m(x,y,z,0) - 3.*mc1;
  mom[17] = m(x,y,z,17) = mc1 - 3.*mxx1 + mc2 - 3.*myy2;
  mom[18] = m(x,y,z,18) = mzz1 - myy1 + mxx2 - mzz2;

  return mom;
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
  m[ 5] = msite(x,y,z, 5) / 48.;
  m[ 6] = msite(x,y,z, 6) / 16.;
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
  m[17] = msite(x,y,z,17) / 48.;
  m[18] = msite(x,y,z,18) / 16.;

  mc0 = 12.*(m[0] - m[4] + m[16]);
  mc1 =  2.*(m[0] - 2.*m[16]);
  mc2 = m[0] + m[4] + m[16];

  mx1 = 2.*(m[1] - 2.*m[10]);
  my1 = 2.*(m[2] - 2.*m[11]);
  mz1 = 2.*(m[3] - 2.*m[12]);

  mx2 = m[1] + m[10] + m[13];
  my2 = m[2] + m[11] + m[14];
  mz2 = m[3] + m[12] + m[15];

  mx3 = m[1] + m[10] - m[13];
  my3 = m[2] + m[11] - m[14];
  mz3 = m[3] + m[12] - m[15];

  mxx1 = mc1 + 4.*(m[5] - m[17]);
  myy1 = mc1 - 2.*(m[5] - m[6]) + 2.*(m[17] - m[18]);
  mzz1 = mc1 - 2.*(m[5] + m[6]) + 2.*(m[17] + m[18]);

  mxy2 = mc2 + (m[5] + m[6]) + (m[17] + m[18]);
  mxz2 = mc2 + (m[5] - m[6]) + (m[17] - m[18]);
  myz2 = mc2 - 2.*(m[5] + m[17]);

  mxy = m[7];
  myz = m[8];
  mxz = m[9];

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
  f(x,y,z,11) = myz2 + my2 + mz3 + myz;
  f(x,y,z,12) = myz2 - my2 - mz3 + myz;
  f(x,y,z,13) = myz2 + my2 - mz3 - myz;
  f(x,y,z,14) = myz2 - my2 + mz3 - myz;
  f(x,y,z,15) = mxz2 + mz2 + mx3 + mxz;
  f(x,y,z,16) = mxz2 - mz2 - mx3 + mxz;
  f(x,y,z,17) = mxz2 - mz2 + mx3 - mxz;
  f(x,y,z,18) = mxz2 + mz2 - mx3 - mxz;

  return;
}

inline Array<Real,ncomp> hydrovars(const Array<Real,ncomp>& m) {
  Array<Real,ncomp> h;

  h[0] = m[0]; // rho
  h[1] = m[1]; // jx
  h[2] = m[2]; // jy
  h[3] = m[3]; // jz
  h[4] = (m[0]+m[4]+m[5])/3.; // pxx
  h[5] = m[7]; // pxy
  h[6] = m[9]; // pxz
  h[7] = (m[0]+m[4]-0.5*m[5]+1.5*m[6])/3.; // pyy
  h[8] = m[8]; // pyz
  h[9] = (m[0]+m[4]-0.5*m[5]-1.5*m[6])/3.; // pzz

  // deviatoric stress
  if (m[0] > std::numeric_limits<Real>::epsilon()) {
    const Real rho = m[0];
    const RealVect j = { m[1], m[2], m[3] };
    h[4] -= rho*cs2 + j[0]*j[0]/rho;
    h[5] -= j[0]*j[1]/rho;
    h[6] -= j[0]*j[2]/rho;
    h[7] -= rho*cs2 + j[1]*j[1]/rho;
    h[8] -= j[1]*j[2]/rho;
    h[9] -= rho*cs2 + j[2]*j[2]/rho;
  }

  return h;
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
  mEq[5] = (2.*u[0]*u[0] - u[1]*u[1] - u[2]*u[2])*rho;
  mEq[6] = (u[1]*u[1] - u[2]*u[2])*rho;
  mEq[7] = u[0]*u[1]*rho;
  mEq[8] = u[1]*u[2]*rho;
  mEq[9] = u[2]*u[0]*rho;

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
