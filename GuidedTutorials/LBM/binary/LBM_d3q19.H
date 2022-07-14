#ifndef LBM_D3Q19_H_
#define LBM_D3Q19_H_

const int nvel = 19;

const Real cs2 = 1./3.;

AMREX_GPU_MANAGED IntVect c[19] = {
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

AMREX_GPU_MANAGED Real w[19] = {
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

AMREX_GPU_MANAGED Real wg[19][3][3] = {
  { {      0., 0., 0. }, { 0.,     0.,  0. }, { 0., 0.,     0.  } },
  { {  5./36., 0., 0. }, { 0., -1./9.,  0. }, { 0., 0., -1./9.  } },
  { {  5./36., 0., 0. }, { 0., -1./9.,  0. }, { 0., 0., -1./9.  } },
  { { -1./9.,  0., 0. }, { 0.,  5./36., 0. }, { 0., 0., -1./9.  } },
  { { -1./9.,  0., 0. }, { 0.,  5./36., 0. }, { 0., 0., -1./9.  } },
  { { -1./9.,  0., 0. }, { 0., -1./9.,  0. }, { 0., 0.,  5./36. } },
  { { -1./9.,  0., 0. }, { 0., -1./9.,  0. }, { 0., 0.,  5./36. } },
  { { -1./72.,  1./12., 0. }, {  1./12., -1./72., 0. }, { 0., 0.,  1./36. } },
  { { -1./72.,  1./12., 0. }, {  1./12., -1./72., 0. }, { 0., 0.,  1./36. } },
  { { -1./72., -1./12., 0. }, { -1./12., -1./72., 0. }, { 0., 0.,  1./36. } },
  { { -1./72., -1./12., 0. }, { -1./12., -1./72., 0. }, { 0., 0.,  1./36. } },
  { {  1./36., 0., 0. }, { 0., -1./72.,  1./12. }, { 0.,  1./12., -1./72. } },
  { {  1./36., 0., 0. }, { 0., -1./72.,  1./12. }, { 0.,  1./12., -1./72. } },
  { {  1./36., 0., 0. }, { 0., -1./72., -1./12. }, { 0., -1./12., -1./72. } },
  { {  1./36., 0., 0. }, { 0., -1./72., -1./12. }, { 0., -1./12., -1./72. } },
  { { -1./72., 0.,  1./12. }, { 0.,  1./36., 0. }, {  1./12., 0., -1./72. } },
  { { -1./72., 0.,  1./12. }, { 0.,  1./36., 0. }, {  1./12., 0., -1./72. } },
  { { -1./72., 0., -1./12. }, { 0.,  1./36., 0. }, { -1./12., 0., -1./72. } },
  { { -1./72., 0., -1./12. }, { 0.,  1./36., 0. }, { -1./12., 0., -1./72. } },
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Array1D<Real,0,nvel> moments(const Array1D<Real,0,nvel>& fsite) {
  Real f;
  Real mc0, mc1, mc2;
  Real mx1, my1, mz1, mx2, my2, mz2, mx3, my3, mz3;
  Real mxy, mxz, myz, mxx1, myy1, mzz1, mxx2, myy2, mzz2;
  Array1D<Real,0,nvel> m;

  f = fsite( 0); mc0  = f;
  f = fsite( 1); mx1  = f; mxx1  = f;
  f = fsite( 2); mx1 -= f; mxx1 += f;
  f = fsite( 3); my1  = f; myy1  = f;
  f = fsite( 4); my1 -= f; myy1 += f;
  f = fsite( 5); mz1  = f; mzz1  = f;
  f = fsite( 6); mz1 -= f; mzz1 += f;
  f = fsite( 7); mx2  = f; my3  = f; mxy  = f; mxx2  = f;
  f = fsite( 8); mx2 -= f; my3 -= f; mxy += f; mxx2 += f;
  f = fsite( 9); mx2 += f; my3 -= f; mxy -= f; mxx2 += f;
  f = fsite(10); mx2 -= f; my3 += f; mxy -= f; mxx2 += f;
  f = fsite(11); my2  = f; mz3  = f; myz  = f; myy2  = f;
  f = fsite(12); my2 -= f; mz3 -= f; myz += f; myy2 += f;
  f = fsite(13); my2 += f; mz3 -= f; myz -= f; myy2 += f;
  f = fsite(14); my2 -= f; mz3 += f; myz -= f; myy2 += f;
  f = fsite(15); mz2  = f; mx3  = f; mxz  = f; mzz2  = f;
  f = fsite(16); mz2 -= f; mx3 -= f; mxz += f; mzz2 += f;
  f = fsite(17); mz2 -= f; mx3 += f; mxz -= f; mzz2 += f;
  f = fsite(18); mz2 += f; mx3 -= f; mxz -= f; mzz2 += f;

  mc1 = mxx1 + myy1 + mzz1;
  mc2 = mxx2 + myy2 + mzz2;

  m( 0) = mc0 + mc1 + mc2;
  m( 1) = mx1 + mx2 + mx3;
  m( 2) = my1 + my2 + my3;
  m( 3) = mz1 + mz2 + mz3;
  m( 4) = mc2 - mc0;
  m( 5) = 3.*mxx1 - mc1 + mc2 - 3.*myy2;
  m( 6) = myy1 - mzz1 + mxx2 - mzz2;
  m( 7) = mxy;
  m( 8) = myz;
  m( 9) = mxz;
  m(10) = m(1) - 3.*mx1;
  m(11) = m(2) - 3.*my1;
  m(12) = m(3) - 3.*mz1;
  m(13) = mx2 - mx3;
  m(14) = my2 - my3;
  m(15) = mz2 - mz3;
  m(16) = m(0) - 3.*mc1;
  m(17) = mc1 - 3.*mxx1 + mc2 - 3.*myy2;
  m(18) = mzz1 - myy1 + mxx2 - mzz2;

  return m;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Array1D<Real,0,nvel> moments(int x, int y, int z, const Array4<Real>& fsite) {
  Array1D<Real,0,nvel> f;
  for (int i=0; i<nvel; ++i) {
    f(i) = fsite(x,y,z,i);
  }
  return moments(f);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Array1D<Real,0,nvel> populations(const Array1D<Real,0,nvel>& mom) {
  Array1D<Real,0,nvel> f, m;

  Real mc0, mc1, mc2;
  Real mx1, my1, mz1, mx2, my2, mz2, mx3, my3, mz3;
  Real mxx1, myy1, mzz1, mxy, mxz, myz, mxy2, mxz2, myz2;

  m( 0) = mom( 0) / 36.;
  m( 1) = mom( 1) / 12.;
  m( 2) = mom( 2) / 12.;
  m( 3) = mom( 3) / 12.;
  m( 4) = mom( 4) / 24.;
  m( 5) = mom( 5) / 48.;
  m( 6) = mom( 6) / 16.;
  m( 7) = mom( 7) / 4.;
  m( 8) = mom( 8) / 4.;
  m( 9) = mom( 9) / 4.;
  m(10) = mom(10) / 24.;
  m(11) = mom(11) / 24.;
  m(12) = mom(12) / 24.;
  m(13) = mom(13) / 8.;
  m(14) = mom(14) / 8.;
  m(15) = mom(15) / 8.;
  m(16) = mom(16) / 72.;
  m(17) = mom(17) / 48.;
  m(18) = mom(18) / 16.;

  mc0 = 12.*(m(0) - m(4) + m(16));
  mc1 =  2.*(m(0) - 2.*m(16));
  mc2 = m(0) + m(4) + m(16);

  mx1 = 2.*(m(1) - 2.*m(10));
  my1 = 2.*(m(2) - 2.*m(11));
  mz1 = 2.*(m(3) - 2.*m(12));

  mx2 = m(1) + m(10) + m(13);
  my2 = m(2) + m(11) + m(14);
  mz2 = m(3) + m(12) + m(15);

  mx3 = m(1) + m(10) - m(13);
  my3 = m(2) + m(11) - m(14);
  mz3 = m(3) + m(12) - m(15);

  mxx1 = mc1 + 4.*(m(5) - m(17));
  myy1 = mc1 - 2.*(m(5) - m(6)) + 2.*(m(17) - m(18));
  mzz1 = mc1 - 2.*(m(5) + m(6)) + 2.*(m(17) + m(18));

  mxy2 = mc2 + (m(5) + m(6)) + (m(17) + m(18));
  mxz2 = mc2 + (m(5) - m(6)) + (m(17) - m(18));
  myz2 = mc2 - 2.*(m(5) + m(17));

  mxy = m(7);
  myz = m(8);
  mxz = m(9);

  f( 0) = mc0;
  f( 1) = mxx1 + mx1;
  f( 2) = mxx1 - mx1;
  f( 3) = myy1 + my1;
  f( 4) = myy1 - my1;
  f( 5) = mzz1 + mz1;
  f( 6) = mzz1 - mz1;
  f( 7) = mxy2 + mx2 + my3 + mxy;
  f( 8) = mxy2 - mx2 - my3 + mxy;
  f( 9) = mxy2 + mx2 - my3 - mxy;
  f(10) = mxy2 - mx2 + my3 - mxy;
  f(11) = myz2 + my2 + mz3 + myz;
  f(12) = myz2 - my2 - mz3 + myz;
  f(13) = myz2 + my2 - mz3 - myz;
  f(14) = myz2 - my2 + mz3 - myz;
  f(15) = mxz2 + mz2 + mx3 + mxz;
  f(16) = mxz2 - mz2 - mx3 + mxz;
  f(17) = mxz2 - mz2 + mx3 - mxz;
  f(18) = mxz2 + mz2 - mx3 - mxz;

  return f;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Array1D<Real,0,nvel> populations(int x, int y, int z, const Array4<Real>& fsite, const Array1D<Real,0,nvel>& m) {
  Array1D<Real,0,nvel> f = populations(m);
  for (int i=0; i<nvel; ++i) {
    fsite(x,y,z,i) = f(i);
  }
  return f;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Array1D<Real,0,nvel> hydrovars(const Array1D<Real,0,nvel>& m) {
  Array1D<Real,0,nvel> h;

  h(0) = m(0); // rho
  h(1) = m(1); // jx
  h(2) = m(2); // jy
  h(3) = m(3); // jz
  h(4) = (m(0)+m(4)+m(5))/3.; // pxx
  h(5) = m(7); // pxy
  h(6) = m(9); // pxz
  h(7) = (m(0)+m(4)-0.5*m(5)+1.5*m(6))/3.; // pyy
  h(8) = m(8); // pyz
  h(9) = (m(0)+m(4)-0.5*m(5)-1.5*m(6))/3.; // pzz

  // deviatoric stress
  if (m(0) > std::numeric_limits<Real>::epsilon()) {
    const Real rho = m(0);
    const RealVect j = { m(1), m(2), m(3) };
    h(4) -= rho*cs2 + j[0]*j[0]/rho;
    h(5) -= j[0]*j[1]/rho;
    h(6) -= j[0]*j[2]/rho;
    h(7) -= rho*cs2 + j[1]*j[1]/rho;
    h(8) -= j[1]*j[2]/rho;
    h(9) -= rho*cs2 + j[2]*j[2]/rho;
  }

  return h;
}

inline Array<Real,nvel> mequilibrium(const Real& rho, const RealVect& u) {
  Array<Real,nvel> mEq;

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
  for (int i=10; i<nvel; ++i) mEq[i] = 0.0;

  return mEq;
}

inline Array<Real,nvel> mequilibrium(int x, int y, int z, const Array4<Real>& m) {
  const Real rho = m(x,y,z,0);
  const RealVect j = { m(x,y,z,1), m(x,y,z,2), m(x,y,z,3) };
  return mequilibrium(rho, j/rho);
}

#endif
