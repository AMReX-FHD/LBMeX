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

inline Array<Real,ncomp> mequilibrium(const Array<Real,ncomp>& m) {
  Array<Real,ncomp> mEq;

  /* mass density */
  mEq[0] = m[0];

  /* momentum density */
  mEq[1] = m[1];
  mEq[2] = m[2];
  mEq[3] = m[3];

  /* equilibrium part of the stress modes */
  mEq[4] = (m[1]*m[1] + m[2]*m[2] + m[3]*m[3])/m[0];
  mEq[5] = (m[1]*m[1] - m[2]*m[2])/m[0];
  mEq[6] = (m[1]*m[1] + m[2]*m[2] - 2.*m[3]*m[3])/m[0];
  mEq[7] = m[1]*m[2]/m[0];
  mEq[8] = m[1]*m[3]/m[0];
  mEq[9] = m[2]*m[3]/m[0];

  /* ghost modes have no equilibrium part due to orthogonality */
  for (int i=10; i<ncomp; ++i) mEq[i] = 0.0;

  return mEq;
}

#endif
