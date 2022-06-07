#ifndef LBM_H_
#define LBM_H_

#include "d3q19.H"

Real tau;

static void lb_relax_modes(double *m) {

  double rho, u[3], m4eq, m5eq, m6eq, m7eq, m8eq, m9eq;

  /* mass density */
  rho = m[0] + lbpar.rho;

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
  m[4] = m4eq + gamma_even * (m[4] - m4eq);
  m[5] = m5eq + gamma_even * (m[5] - m5eq);
  m[6] = m6eq + gamma_even * (m[6] - m6eq);
  m[7] = m7eq + gamma_even * (m[7] - m7eq);
  m[8] = m8eq + gamma_even * (m[8] - m8eq);
  m[9] = m9eq + gamma_even * (m[9] - m9eq);
  
  /* ghost modes have no equilibrium part due to orthogonality */
  m[10] = gamma_odd  * m[10];
  m[11] = gamma_odd  * m[11];
  m[12] = gamma_odd  * m[12];
  m[13] = gamma_odd  * m[13];
  m[14] = gamma_odd  * m[14];
  m[15] = gamma_odd  * m[15];
  m[16] = gamma_even * m[16];
  m[17] = gamma_even * m[17];
  m[18] = gamma_even * m[18];
  
}

/***********************************************************************/

void lb_thermalize_modes(double *m) {
  
  double rootrho = sqrt(m[0]+lbpar.rho);

  /* stress modes */
  m[4] += rootrho*lb_phi[4]*gaussian_random();
  m[5] += rootrho*lb_phi[5]*gaussian_random();
  m[6] += rootrho*lb_phi[6]*gaussian_random();
  m[7] += rootrho*lb_phi[7]*gaussian_random();
  m[8] += rootrho*lb_phi[8]*gaussian_random();
  m[9] += rootrho*lb_phi[9]*gaussian_random();
    
  /* ghost modes */
  m[10] += rootrho*lb_phi[10]*gaussian_random();
  m[11] += rootrho*lb_phi[11]*gaussian_random();
  m[12] += rootrho*lb_phi[12]*gaussian_random();
  m[13] += rootrho*lb_phi[13]*gaussian_random();
  m[14] += rootrho*lb_phi[14]*gaussian_random();
  m[15] += rootrho*lb_phi[15]*gaussian_random();
  m[16] += rootrho*lb_phi[16]*gaussian_random();
  m[17] += rootrho*lb_phi[17]*gaussian_random();
  m[18] += rootrho*lb_phi[18]*gaussian_random();

}

/***********************************************************************/

static void lb_collisions(double *m) {

  lb_relax_modes(m);
  if (fluct) lb_thermalize_modes(m);

}

/***********************************************************************/

static void lb_calc_modes(double *m, double PFI, int x, int y, int z) {
  int xc, yc;
  double f, mc0, mc1, mc2;
  double mx1, my1, mz1, mx2, my2, mz2, mx3, my3, mz3;
  double mxx1, myy1, mzz1, mxx2, myy2, mzz2;
  double mxy, mxz, myz;

  xc = x%3;
  yc = y+1;

  f = FI( 0, xc, yc)[z]; mc0  = f;
  f = FI( 1, xc, yc)[z]; mx1  = f; mxx1  = f;
  f = FI( 2, xc, yc)[z]; mx1 -= f; mxx1 += f;
  f = FI( 3, xc, yc)[z]; my1  = f; myy1  = f;
  f = FI( 4, xc, yc)[z]; my1 -= f; myy1 += f;
  f = FI( 5, xc, yc)[z]; mz1  = f; mzz1  = f;
  f = FI( 6, xc, yc)[z]; mz1 -= f; mzz1 += f;
  f = FI( 7, xc, yc)[z]; mx2  = f; my3  = f; mxy  = f; mzz2  = f;
  f = FI( 8, xc, yc)[z]; mx2 -= f; my3 -= f; mxy += f; mzz2 += f;
  f = FI( 9, xc, yc)[z]; mx2 += f; my3 -= f; mxy -= f; mzz2 += f;
  f = FI(10, xc, yc)[z]; mx2 -= f; my3 += f; mxy -= f; mzz2 += f;
  f = FI(11, xc, yc)[z]; mz2  = f; mx3  = f; mxz  = f; mxx2  = f;
  f = FI(12, xc, yc)[z]; mz2 -= f; mx3 -= f; mxz += f; mxx2 += f;
  f = FI(13, xc, yc)[z]; mz2 -= f; mx3 += f; mxz -= f; mxx2 += f;
  f = FI(14, xc, yc)[z]; mz2 += f; mx3 -= f; mxz -= f; mxx2 += f;
  f = FI(15, xc, yc)[z]; my2  = f; mz3  = f; myz  = f; myy2  = f;
  f = FI(16, xc, yc)[z]; my2 -= f; mz3 -= f; myz += f; myy2 += f;
  f = FI(17, xc, yc)[z]; my2 += f; mz3 -= f; myz -= f; myy2 += f;
  f = FI(18, xc, yc)[z]; my2 -= f; mz3 += f; myz -= f; myy2 += f;

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

}

/***********************************************************************/

static void lb_calc_fi_stream(double *m, double PFI, int x, int y, int z) {
  int xc, xp, xm, yc, yp, ym, zp, zm;
  double mc0, mc1, mc2;
  double mx1, my1, mz1, mx2, my2, mz2, mx3, my3, mz3;
  double mxx1, myy1, mzz1, mxy, mxz, myz, mxy2, mxz2, myz2;

  xc = x%3; xp = (xc+1)%3; xm = (xc+2)%3;
  yc = y+1; yp = yc+1; ym = yc-1;
  zp = z+1; zm = z-1;

  m[ 0] /= 36.;
  m[ 1] /= 12.;
  m[ 2] /= 12.;
  m[ 3] /= 12.;
  m[ 4] /= 24.;
  m[ 5] /= 16.;
  m[ 6] /= 48.;
  m[ 7] /= 4.;
  m[ 8] /= 4.;
  m[ 9] /= 4.;
  m[10] /= 24.;
  m[11] /= 24.;
  m[12] /= 24.;
  m[13] /= 8.;
  m[14] /= 8.;
  m[15] /= 8.;
  m[16] /= 72.;
  m[17] /= 16.;
  m[18] /= 48.;

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

  FI( 0, xc, yc)[z]  = mc0;
  FI( 1, xp, yc)[z]  = mxx1 + mx1;
  FI( 2, xm, yc)[z]  = mxx1 - mx1;
  FI( 3, xc, yp)[z]  = myy1 + my1;
  FI( 4, xc, ym)[z]  = myy1 - my1;
  FI( 5, xc, yc)[zp] = mzz1 + mz1;
  FI( 6, xc, yc)[zm] = mzz1 - mz1;
  FI( 7, xp, yp)[z]  = mxy2 + mx2 + my3 + mxy;
  FI( 8, xm, ym)[z]  = mxy2 - mx2 - my3 + mxy;
  FI( 9, xp, ym)[z]  = mxy2 + mx2 - my3 - mxy;
  FI(10, xm, yp)[z]  = mxy2 - mx2 + my3 - mxy;
  FI(11, xp, yc)[zp] = mxz2 + mz2 + mx3 + mxz;
  FI(12, xm, yc)[zm] = mxz2 - mz2 - mx3 + mxz;
  FI(13, xp, yc)[zm] = mxz2 - mz2 + mx3 - mxz;
  FI(14, xm, yc)[zp] = mxz2 + mz2 - mx3 - mxz;
  FI(15, xc, yp)[zp] = myz2 + my2 + mz3 + myz;
  FI(16, xc, ym)[zm] = myz2 - my2 - mz3 + myz;
  FI(17, xc, yp)[zm] = myz2 + my2 - mz3 - myz;
  FI(18, xc, ym)[zp] = myz2 - my2 + mz3 - myz;

}

inline Array<Real,ncomp> moments(const Array<Real,ncomp>& f) {
  Real density = 0.0;
  RealVect u, j = { 0., 0., 0. };
  Array<Real,ncomp> m;
  for (int i=0; i<ncomp; ++i) {
    density += f[i];
    j += f[i]*static_cast<RealVect>(c[i]);
  }
  u = j/density;
  m[0] = density;
  m[1] = u[0];
  m[2] = u[1];
  m[3] = u[2];  
  return m;
}

inline Array<Real,ncomp> fequilibrium(const Real& rho, const RealVect& u) {
  Array<Real,ncomp> fEq;
  for (int i=0; i<ncomp; ++i) {
    Real uc = u.dotProduct(c[i]);
    fEq[i] = w[i] * rho * ( 1. + uc/cs2 + (uc*uc - u.dotProduct(u)*cs2)/(2.*cs2*cs2) );
  }
  return fEq;
}

// push scheme
inline void collide_stream(int x, int y, int z, const Array4<Real>& fOld, const Array4<Real>& fNew) {
  Array<Real,ncomp> f, fEq, m;
  Real rho; RealVect u;
  for (int i=0; i<ncomp; ++i) { // read populations from current site
    f[i] = fOld(x,y,z,i);
  }
  m = moments(f);
  rho = m[0];
  u[0] = m[1];
  u[1] = m[2];
  u[2] = m[3];
  fEq = fequilibrium(rho, u);
  for (int i=0; i<ncomp; ++i) { // push populations to neighbors
    int xp = x + c[i][0];
    int yp = y + c[i][1];
    int zp = z + c[i][2];
    fNew(xp,yp,zp,i) = f[i] - 1./tau * ( f[i] - fEq[i] );
  }
}

// pull scheme
inline void stream_collide(int x, int y, int z, const Array4<Real>& fOld, const Array4<Real>& fNew) {
  Array<Real,ncomp> f, fEq, m;
  Real rho; RealVect u;
  for (int i=0; i<ncomp; ++i) { // pull populations from neighbors
    int xm = x - c[i][0];
    int ym = y - c[i][1];
    int zm = z - c[i][2];
    f[i] = fOld(xm,ym,zm,i);
  }
  m = moments(f);
  rho = m[0];
  u[0] = m[1];
  u[1] = m[2];
  u[2] = m[3];
  fEq = fequilibrium(rho, u);
  for (int i=0; i<ncomp; ++i) { // write populations to current site
    fNew(x,y,z,i) = f[i] - 1./tau * ( f[i] - fEq[i] );
  }
}

#endif
