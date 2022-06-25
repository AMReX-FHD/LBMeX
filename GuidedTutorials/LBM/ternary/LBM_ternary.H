#ifndef LBM_H_
#define LBM_H_

#include "LBM_d3q19.H"

Real tau_r = 1.0,
     tau_p = 1.0,
     tau_s = 0.666667;

Real kappa1 = 0.01,
     kappa2 = 0.01,
     kappa3 = 0.01;

Real kappa_rr = (kappa1+kappa2)/4.,
     kappa_pp = kappa_rr,
     kappa_ss = (kappa1+kappa2+4.*kappa3)/4.,
     kappa_rp = (kappa1-kappa2)/4.,
     kappa_ps = -kappa_rp,
     kappa_rs = -kappa_rr;

Real gamma_p = 1.0,
     gamma_s = 1.0;

Real alpha = 1.0;

inline RealVect gradient(int x, int y, int z, const Array4<Real>& field, int icomp) {
  RealVect gradient = { 0.0, 0.0, 0.0 };
  for (int i=0; i<nvel; i++) {
    int xp = x + c[i][0];
    int yp = y + c[i][1];
    int zp = z + c[i][2];
    for (int dir=0; dir<3; dir++) {
      gradient[dir] += w[i]/cs2*field(xp,yp,zp,icomp)*c[i][dir];
    }
  }
  return gradient;
}

inline Real laplacian(int x, int y, int z, const Array4<Real>& field, int icomp) {
  Real laplacian = 0.0;
  for (int i=0; i<nvel; i++) {
    int xp = x + c[i][0];
    int yp = y + c[i][1];
    int zp = z + c[i][2];
    laplacian += 2.*w[i]/cs2*(field(xp,yp,zp,icomp)-field(x,y,z,icomp));
  }
  return laplacian;
}

inline Array<Real,3> chemical_potentials(const Array<Real,3>& fields,
					 const Array<RealVect,3>& gradients,
					 const Array<Real,3>& laplacians) {
  const Real alpha2 = alpha*alpha;
  const Real rho = fields[0];
  const Real phi = fields[1];
  const Real psi = fields[2];
  const RealVect Drho = gradients[0];
  const RealVect Dphi = gradients[1];
  const RealVect Dpsi = gradients[2];
  const Real D2rho = laplacians[0];
  const Real D2phi = laplacians[1];
  const Real D2psi = laplacians[2];

  Array<Real,3> mu = { 0.0, 0.0, 0.0 } ;
#if 0
  /* mu_rho is not needed for the calculations.
     The following expression has not been tested.
     (Eq. (38) in Semprebon et al. may contain typos) */
  mu[0] =
      kappa1/8.*(rho+psi-phi)*(rho+phi-psi-2.)*(rho+phi-psi-1.)
    + kappa2/8.*(rho-phi-psi)*(rho-phi-psi-2.)*(rho-phi-psi-1.)
    - alpha2/4.*((kappa1+kappa2)*(D2rho-D2psi)-(kappa1-kappa2)*D2phi);
#endif

  mu[1] = // Eq. (39) in Semprebon et al.
      kappa1/8.*(rho+phi-psi)*(rho+phi-psi-2.)*(rho+phi-psi-1.)
    - kappa2/8.*(rho-phi-psi)*(rho-phi-psi-2.)*(rho-phi-psi-1.)
    - alpha2/4.*((kappa1-kappa2)*(D2rho-D2psi)+(kappa1+kappa2)*D2phi);

  mu[2] = // Eq. (40) in Semprebon et al.
    - kappa1/8.*(rho+phi-psi)*(rho+phi-psi-2.)*(rho+phi-psi-1.)
    - kappa2/8.*(rho-phi-psi)*(rho-phi-psi-2.)*(rho-phi-psi-1.)
    + kappa3*psi*(psi-1.)*(2.*psi-1.)
    + alpha2/4.*((kappa1+kappa2)*D2rho+(kappa1-kappa2)*D2phi
		 -(kappa1+kappa2+4.*kappa3)*D2psi);

  return mu;
}

inline Real pressure(Real rho, Real phi, Real psi) {
  const Real rho2 = rho*rho;
  const Real rho3 = rho2*rho;
  const Real rho4 = rho3*rho;
  const Real phi2 = phi*phi;
  const Real phi3 = phi2*phi;
  const Real phi4 = phi3*phi;
  const Real psi2 = psi*psi;
  const Real psi3 = psi2*psi;
  const Real psi4 = psi3*psi;

  Real p0 = rho*cs2 // Eq. (43) Semprebon et al.
    + (kappa1+kappa2)*(3./32.*(rho4+phi4+psi4)
		       - 1./4.*(rho3+rho*psi-psi3)
		       + 1./8.*(rho2+phi2+psi2)
		       - 3./8.*(rho3*psi+psi3*rho)
		       + 9./16.*(rho2*phi2+rho2*psi2+phi2*psi2)
		       + 3./4.*(rho2*psi-rho*phi2-rho*psi2+phi2*psi)
		       - 9./8.*phi2*psi*rho)
    + (kappa1-kappa2)*(3./8.*(rho3*phi+rho*phi3-phi3*psi-phi*psi3)
		       + 1./4.*(rho*phi-phi*psi-phi3)
		       + 9./8.*(phi*psi2*rho-phi*psi*rho2)
		       - 3./4.*(rho2*phi+phi*psi2)
		       + 3./2.*phi*psi*rho)
    + kappa3*(3./2.*psi4 - 2.*psi3 + 1./2.*psi2);

  return p0;
}

inline void hydrovars(int x, int y, int z,
		      const Array4<Real>& f,
		      const Array4<Real>& g,
		      const Array4<Real>& k,
		      const Array4<Real>& h) {
  Real rho, phi, psi, j[3];
  rho = phi = psi = j[0] = j[1] = j[2] = 0.0;
  for (int i=0; i<nvel; ++i) {
    Real fi, gi, ki;
    fi = f(x,y,z,i);
    gi = g(x,y,z,i);
    ki = k(x,y,z,i);
    rho  += fi;
    phi  += gi;
    psi  += ki;
    j[0] += fi*c[i][0];
    j[1] += fi*c[i][1];
    j[2] += fi*c[i][2];
  }
  h(x,y,z,0) = rho;
  h(x,y,z,1) = phi;
  h(x,y,z,2) = psi;
  h(x,y,z,3) = j[0]/rho;
  h(x,y,z,4) = j[1]/rho;
  h(x,y,z,5) = j[2]/rho;
  h(x,y,z,6) = pressure(rho,phi,psi);
}

inline Array<Real,nvel> fequilibrium(const Array<Real,3>& fields,
				     const Array<RealVect,3>& gradients,
				     const Array<Real,3>& laplacians,
				     const RealVect& u) {
  Array<Real,nvel> fEq;

  const Real rho = fields[0];
  const Real phi = fields[1];
  const Real psi = fields[2];
  const RealVect Drho = gradients[0];
  const RealVect Dphi = gradients[1];
  const RealVect Dpsi = gradients[2];
  const Real D2rho = laplacians[0];
  const Real D2phi = laplacians[1];
  const Real D2psi = laplacians[2];
  const Real pbulk = pressure(rho, phi, psi);

  Real ruu[3][3];
  ruu[0][0] = rho*u[0]*u[0];
  ruu[1][1] = rho*u[1]*u[1];
  ruu[2][2] = rho*u[2]*u[2];
  ruu[0][1] = rho*u[0]*u[1];
  ruu[1][2] = rho*u[1]*u[2];
  ruu[2][0] = rho*u[2]*u[0];

  Real G[3][3];
  G[0][0] = kappa_rr*Drho[0]*Drho[0]+kappa_pp*Dphi[0]*Dphi[0]+kappa_ss*Dpsi[0]*Dpsi[0];
  G[1][1] = kappa_rr*Drho[1]*Drho[1]+kappa_pp*Dphi[1]*Dphi[1]+kappa_ss*Dpsi[1]*Dpsi[1];
  G[2][2] = kappa_rr*Drho[2]*Drho[2]+kappa_pp*Dphi[2]*Dphi[2]+kappa_ss*Dpsi[2]*Dpsi[2];
  G[0][1] = kappa_rr*Drho[0]*Drho[1]+kappa_pp*Dphi[0]*Dphi[1]+kappa_ss*Dpsi[0]*Dpsi[1];
  G[1][2] = kappa_rr*Drho[1]*Drho[2]+kappa_pp*Dphi[1]*Dphi[2]+kappa_ss*Dpsi[1]*Dpsi[2];
  G[2][0] = kappa_rr*Drho[2]*Drho[0]+kappa_pp*Dphi[2]*Dphi[0]+kappa_ss*Dpsi[2]*Dpsi[0];

  Real fi, sumf = 0.0;
  for (int i=1; i<nvel; ++i) { // Eq. (52) Semprebon et al.
    fi  = 3.*w[i]*pbulk;
    fi += 3.*w[i]*rho*(u[0]*c[i][0]+u[1]*c[i][1]+u[2]*c[i][2]);
    fi += 9./2.*w[i]*((ruu[0][0]*c[i][0]+2.*ruu[0][1]*c[i][1])*c[i][0]
			 +(ruu[1][1]*c[i][1]+2.*ruu[1][2]*c[i][2])*c[i][1]
			 +(ruu[2][2]*c[i][2]+2.*ruu[2][0]*c[i][0])*c[i][2]);
    fi -= 3./2.*w[i]*(ruu[0][0]+ruu[1][1]+ruu[2][2]);
    fi -= 3.*w[i]*(kappa_rr*rho*D2rho+kappa_pp*phi*D2phi+kappa_ss*psi*D2psi);
    fi -= 3.*w[i]*(kappa_rp*(rho*D2phi+phi*D2rho)
		      +kappa_rs*(rho*D2psi+psi*D2rho)
		      +kappa_ps*(phi*D2psi+psi*D2phi));
    fi += 3.*(wg[i][0][0]*G[0][0]+wg[i][1][1]*G[1][1]+wg[i][2][2]*G[2][2]
	      +wg[i][0][1]*G[0][1]+wg[i][1][2]*G[1][2]+wg[i][2][0]*G[2][0]);
    fi += 6.*kappa_rp*(wg[i][0][0]*Drho[0]*Dphi[0]
		       +wg[i][1][1]*Drho[1]*Dphi[1]
		       +wg[i][2][2]*Drho[2]*Dphi[2]);
    fi += 6.*kappa_rs*(wg[i][0][0]*Drho[0]*Dpsi[0]
		       +wg[i][1][1]*Drho[1]*Dpsi[1]
		       +wg[i][2][2]*Drho[2]*Dpsi[2]);
    fi += 6.*kappa_ps*(wg[i][0][0]*Dphi[0]*Dpsi[0]
		       +wg[i][1][1]*Dphi[1]*Dpsi[1]
		       +wg[i][2][2]*Dphi[2]*Dpsi[2]);
    fi += 3.*kappa_rp*(wg[i][0][1]*(Drho[0]*Dphi[1]+Drho[1]*Dphi[0])
		       +wg[i][1][2]*(Drho[1]*Dphi[2]+Drho[2]*Dphi[1])
		       +wg[i][2][0]*(Drho[2]*Dphi[0]+Drho[0]*Dphi[2]));
    fi += 3.*kappa_rs*(wg[i][0][1]*(Drho[0]*Dpsi[1]+Drho[1]*Dpsi[0])
		       +wg[i][1][2]*(Drho[1]*Dpsi[2]+Drho[2]*Dpsi[1])
		       +wg[i][2][0]*(Drho[2]*Dpsi[0]+Drho[0]*Dpsi[2]));
    fi += 3.*kappa_ps*(wg[i][0][1]*(Dphi[0]*Dpsi[1]+Dphi[1]*Dpsi[0])
		       +wg[i][1][2]*(Dphi[1]*Dpsi[2]+Dphi[2]*Dpsi[1])
		       +wg[i][2][0]*(Dphi[2]*Dpsi[0]+Dphi[0]*Dpsi[2]));
    fEq[i] = fi;
    sumf += fi;
  }
  fEq[0] = rho - sumf;

  return fEq;
}

inline Array<Real,nvel> gequilibrium(const Real& phi,
				     const Real& mu_phi,
				     const RealVect& u) {
  Array<Real,nvel> gEq;

  Real puu[3][3];
  puu[0][0] = phi*u[0]*u[0];
  puu[1][1] = phi*u[1]*u[1];
  puu[2][2] = phi*u[2]*u[2];
  puu[0][1] = phi*u[0]*u[1];
  puu[1][2] = phi*u[1]*u[2];
  puu[2][0] = phi*u[2]*u[0];

  Real gi, sumg = 0.0;
  for (int i=1; i<nvel; ++i) { // Eq. (53) Semprebon et al.
    gi  = 3.*w[i]*gamma_p*mu_phi;
    gi += 3.*w[i]*phi*(u[0]*c[i][0]+u[1]*c[i][1]+u[2]*c[i][2]);
    gi += 9./2.*w[i]*((puu[0][0]*c[i][0]+2.*puu[0][1]*c[i][1])*c[i][0]
			 +(puu[1][1]*c[i][1]+2.*puu[1][2]*c[i][2])*c[i][1]
			 +(puu[2][2]*c[i][2]+2.*puu[2][0]*c[i][0])*c[i][2]);
    gi -= 3./2.*w[i]*(puu[0][0]+puu[1][1]+puu[2][2]);
    gEq[i] = gi;
    sumg += gi;
  }
  gEq[0] = phi - sumg;

  return gEq;
}

inline Array<Real,nvel> kequilibrium(const Real& psi,
				     const Real& mu_psi,
				     const RealVect& u) {
  Array<Real,nvel> kEq;

  double puu[3][3];
  puu[0][0] = psi*u[0]*u[0];
  puu[1][1] = psi*u[1]*u[1];
  puu[2][2] = psi*u[2]*u[2];
  puu[0][1] = psi*u[0]*u[1];
  puu[1][2] = psi*u[1]*u[2];
  puu[2][0] = psi*u[2]*u[0];

  double ki, sumk = 0.0;
  for (int i=1; i<nvel; ++i) { // Eq. (54) Semprebon et al.
    ki  = 3.*w[i]*gamma_s*mu_psi;
    ki += 3.*w[i]*psi*(u[0]*c[i][0]+u[1]*c[i][1]+u[2]*c[i][2]);
    ki += 9./2.*w[i]*((puu[0][0]*c[i][0]+2.*puu[0][1]*c[i][1])*c[i][0]
			 +(puu[1][1]*c[i][1]+2.*puu[1][2]*c[i][2])*c[i][1]
			 +(puu[2][2]*c[i][2]+2.*puu[2][0]*c[i][0])*c[i][2]);
    ki -= 3./2.*w[i]*(puu[0][0]+puu[1][1]+puu[2][2]);
    kEq[i] = ki;
    sumk += ki;
  }
  kEq[0] = psi - sumk;

  return kEq;
}

inline void collide(int x, int y, int z,
		    Array4<Real> f,
		    Array4<Real> g,
		    Array4<Real> k,
		    Array4<Real> h) {

  const Array<Real,3> fields = { h(x,y,z,0), h(x,y,z,1), h(x,y,z,2) };
  const RealVect u = { h(x,y,z,3), h(x,y,z,4), h(x,y,z,5) };

  const Array<RealVect,3> gradients = {
    gradient(x,y,z,h,0),
    gradient(x,y,z,h,1),
    gradient(x,y,z,h,2)
  };

  const Array<Real,3> laplacians = {
    laplacian(x,y,z,h,0),
    laplacian(x,y,z,h,1),
    laplacian(x,y,z,h,2)
  };

  const Array<Real,3> mu = chemical_potentials(fields, gradients, laplacians);

  Array<Real,nvel> fEq, gEq, kEq;
  fEq = fequilibrium(fields, gradients, laplacians, u);
  gEq = gequilibrium(fields[1], mu[1], u);
  kEq = kequilibrium(fields[2], mu[2], u);

  for (int i=0; i<nvel; ++i) {
    f(x,y,z,i) = f(x,y,z,i) - 1./tau_r * (f(x,y,z,i) - fEq[i]);
    g(x,y,z,i) = g(x,y,z,i) - 1./tau_p * (g(x,y,z,i) - gEq[i]);
    k(x,y,z,i) = k(x,y,z,i) - 1./tau_s * (k(x,y,z,i) - kEq[i]);
  }

}

// stream from \vec{x} to \vec{x}+\vec{c}_i (push scheme)
inline void stream_push(int x, int y, int z,
			const Array4<Real>& fOld,
			const Array4<Real>& gOld,
			const Array4<Real>& kOld,
			const Array4<Real>& fNew,
			const Array4<Real>& gNew,
			const Array4<Real>& kNew) {
  for (int i=0; i<nvel; ++i) { // push populations to neighbors
    int xp = x + c[i][0];
    int yp = y + c[i][1];
    int zp = z + c[i][2];
    fNew(xp,yp,zp,i) = fOld(x,y,z,i);
    gNew(xp,yp,zp,i) = gOld(x,y,z,i);
    kNew(xp,yp,zp,i) = kOld(x,y,z,i);
  }
}

// push scheme
inline void collide_stream(int x, int y, int z,
			   const Array4<Real>& fOld,
			   const Array4<Real>& gOld,
			   const Array4<Real>& kOld,
			   const Array4<Real>& fNew,
			   const Array4<Real>& gNew,
			   const Array4<Real>& kNew,
			   const Array4<Real>& h
			   ) {
  collide(x, y, z, fOld, gOld, kOld, h); // collide populations on current site
  stream_push(x, y, z, fOld, gOld, kOld, fNew, gNew, kNew); // push populations to neighbors
}

inline void LBM_hydrovars(MultiFab& mf, MultiFab& mg, MultiFab& mk,
			  MultiFab& hydrovs) {
  auto const & f = mf.arrays();
  auto const & g = mg.arrays();
  auto const & k = mk.arrays();
  auto const & h = hydrovs.arrays();
  ParallelFor(hydrovs, IntVect(0), [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) {
    hydrovars(x, y, z, f[nbx], g[nbx], k[nbx], h[nbx]);
  });
}

inline void LBM_timestep(const Geometry& geom,
			 MultiFab& fold, MultiFab& gold, MultiFab& kold,
			 MultiFab& fnew, MultiFab& gnew, MultiFab& knew,
			 MultiFab& hydrovs) {
  fold.FillBoundary(geom.periodicity());
  gold.FillBoundary(geom.periodicity());
  kold.FillBoundary(geom.periodicity());
  hydrovs.FillBoundary(geom.periodicity());
  auto const & fOld = fold.arrays();
  auto const & gOld = gold.arrays();
  auto const & kOld = kold.arrays();
  auto const & fNew = fnew.arrays();
  auto const & gNew = gnew.arrays();
  auto const & kNew = knew.arrays();
  auto const & h = hydrovs.arrays();
  IntVect halo = IntVect(1); // need to push from first halo layer into domain
  ParallelFor(fold, halo, [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) {
    collide_stream(x, y, z,
		   fOld[nbx], gOld[nbx], kOld[nbx],
		   fNew[nbx], gNew[nbx], kNew[nbx], h[nbx]);
  });
  MultiFab::Swap(fold, fnew, 0, 0, nvel, 0);
  MultiFab::Swap(gold, gnew, 0, 0, nvel, 0);
  MultiFab::Swap(kold, knew, 0, 0, nvel, 0);
  LBM_hydrovars(fold, gold, kold, hydrovs);
}

// homogeneous mixture of C1, C2, and C3 with random concentration fluctuations
inline void LBM_init_mixture(MultiFab& mf,
			     MultiFab& mg,
			     MultiFab& mk,
			     MultiFab& hydrovs) {
  const Real C1 = 0.33;
  const Real C2 = 0.33;
  const Real C3 = 0.34;
  for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
    const Box& valid_box = mfi.validbox();
    const Array4<Real>& f = mf.array(mfi);
    const Array4<Real>& g = mg.array(mfi);
    const Array4<Real>& k = mk.array(mfi);
    ParallelForRNG(valid_box, [=] AMREX_GPU_DEVICE(int x, int y, int z, RandomEngine const& engine) {
      const Real rho = C1 + C2 + C3;
      const Real phi = (C1-C2)+0.01*RandomNormal(0., 1., engine);
      const Real psi = C3;
      for (int i=0; i<nvel; i++) {
	f(x,y,z,i) = w[i]*rho;
	g(x,y,z,i) = w[i]*phi;
	k(x,y,z,i) = w[i]*psi;
      }
    });
  }
  LBM_hydrovars(mf, mg, mk, hydrovs);
}

// double emulsion droplet of C1 and C2 surrounded by C3
inline void LBM_init_double_emulsion(const Real r,
				     const Geometry geom,
				     MultiFab& mf,
				     MultiFab& mg,
				     MultiFab& mk,
				     MultiFab& hydrovs) {
  const IntVect box = geom.Domain().length();
  const Real R = r*box[0];

  auto const & f = mf.arrays();
  auto const & g = mg.arrays();
  auto const & k = mk.arrays();
  GpuArray<Real,3> dx = geom.CellSizeArray();
  ParallelFor(mf, IntVect(0), [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) {
    Real rx = x-box[0]/2;
    Real ry = y-box[1]/2;
#if (AMREX_SPACEDIM == 2)
    Real r2 = rx*rx+ry*ry;
#elif (AMREX_SPACEDIM == 3)
    Real rz = z-box[0]/2;
    Real r2 = rx*rx+ry*ry+rz*rz;
#endif

    Real const rho=1.0;
    Real phi, psi;

    if (r2 > R*R) {
      phi = 0.0;
      psi = 1.0;
    }  else if (rx < 0) {
      phi = 1.0;
      psi = 0.0;
    } else {
      phi = -1.0;
      psi = 0.0;
    }

    for (int i=0; i<nvel; i++) {
      f[nbx](x,y,z,i) = w[i]*rho;
      g[nbx](x,y,z,i) = w[i]*phi;
      k[nbx](x,y,z,i) = w[i]*psi;
    }

  });
  LBM_hydrovars(mf, mg, mk, hydrovs);
}

// liquid lens of component C3 between layers of C1 and C2
inline void LBM_init_liquid_lens(const Real r,
				 const Geometry& geom,
				 MultiFab& mf,
				 MultiFab& mg,
				 MultiFab& mk,
				 MultiFab& hydrovs) {
  const IntVect box = geom.Domain().length();
  const Real R = r*box[0];

  auto const & f = mf.arrays();
  auto const & g = mg.arrays();
  auto const & k = mk.arrays();
  GpuArray<Real,3> dx = geom.CellSizeArray();
  ParallelFor(mf, IntVect(0), [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) {
    Real rx = x-box[0]/2;
    Real ry = y-box[1]/2;
#if (AMREX_SPACEDIM == 2)
    Real r2 = rx*rx+ry*ry;
#elif (AMREX_SPACEDIM == 3)
    Real rz = z-box[0]/2;
    Real r2 = rx*rx+ry*ry+rz*rz;
#endif

    const Real rho = 1.0;
    Real phi, psi;

    if (r2 < R*R) {
      phi = 0.0;
      psi = 1.0;
    } else if (rz > 0) {
      phi = 1.0;
      psi = 0.0;
    } else {
      phi = -1.0;
      psi = 0.0;
    }

    for (int i=0; i<nvel; i++) {
      f[nbx](x,y,z,i) = w[i]*rho;
      g[nbx](x,y,z,i) = w[i]*phi;
      k[nbx](x,y,z,i) = w[i]*psi;
    }

  });
  LBM_hydrovars(mf, mg, mk, hydrovs);
}

#endif
