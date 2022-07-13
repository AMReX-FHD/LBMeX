#ifndef LBM_H_
#define LBM_H_

using namespace amrex;

const int ncomp = 19;
const Real cs2 = 1./3.;

AMREX_GPU_MANAGED Real tau = 0.9;

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

void lbm_main();

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real hydrovars(const Array1D<Real,0,ncomp>& f, RealVect& u) {
  Real density = 0.0;
  RealVect j = { 0., 0., 0. };
  for (int i=0; i<ncomp; ++i) {
    density += f(i);
    j += f(i)*static_cast<RealVect>(c[i]);
  }
  u = j/density;
  return density;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Array1D<Real,0,ncomp> fequilibrium(const Real& rho, const RealVect& u) {
  Array1D<Real,0,ncomp> fEq;
  for (int i=0; i<ncomp; ++i) {
    Real uc = u.dotProduct(c[i]);
    fEq(i) = w[i] * rho * ( 1. + uc/cs2 + (uc*uc - u.dotProduct(u)*cs2)/(2.*cs2*cs2) );
  }
  return fEq;
}

// push scheme
inline void collide_stream(int x, int y, int z, const Array4<Real>& fOld, const Array4<Real>& fNew) {
  Array1D<Real,0,ncomp> f, fEq;
  Real rho; RealVect u;
  for (int i=0; i<ncomp; ++i) { // read populations from current site
    f(i) = fOld(x,y,z,i);
  }
  rho = hydrovars(f, u);
  fEq = fequilibrium(rho, u);
  for (int i=0; i<ncomp; ++i) { // push populations to neighbors
    int xp = x + c[i][0];
    int yp = y + c[i][1];
    int zp = z + c[i][2];
    fNew(xp,yp,zp,i) = f(i) - 1./tau * ( f(i) - fEq(i) );
  }
}

// pull scheme
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void stream_collide(int x, int y, int z, const Array4<Real>& fOld, const Array4<Real>& fNew) {
  Array1D<Real,0,ncomp> f, fEq;
  Real rho; RealVect u;
  for (int i=0; i<ncomp; ++i) { // pull populations from neighbors
    int xm = x - c[i][0];
    int ym = y - c[i][1];
    int zm = z - c[i][2];
    f(i) = fOld(xm,ym,zm,i);
  }
  rho = hydrovars(f, u);
  fEq = fequilibrium(rho, u);
  for (int i=0; i<ncomp; ++i) { // write populations to current site
    fNew(x,y,z,i) = f(i) - 1./tau * ( f(i) - fEq(i) );
  }
}

#endif
