#ifndef LBM_H_
#define LBM_H_

#include "LBM_d3q19.H"

// 0.5*(1.+1./sqrt(3.))
AMREX_GPU_MANAGED Real tau_r = 0.7886751345948129; // minimizes spurious currents

AMREX_GPU_MANAGED Real tau_p = 1.0;

AMREX_GPU_MANAGED Real kappa = 0.01;

AMREX_GPU_MANAGED Real lambda = 1.1;
AMREX_GPU_MANAGED Real T = 0.5;

AMREX_GPU_MANAGED Real Gamma = 1.0;

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RealVect gradient(int x, int y, int z, const Array4<Real>& field, int icomp) {
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

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real laplacian(int x, int y, int z, const Array4<Real>& field, int icomp) {
  Real laplacian = 0.0;
  for (int i=0; i<nvel; i++) {
    int xp = x + c[i][0];
    int yp = y + c[i][1];
    int zp = z + c[i][2];
    laplacian += 2.*w[i]/cs2*(field(xp,yp,zp,icomp)-field(x,y,z,icomp));
  }
  return laplacian;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Array<Real,2> chemical_potentials(const Array<Real,2>& fields,
				  const Array<RealVect,2>& gradients,
				  const Array<Real,2>& laplacians) {
  const Real rho = fields[0];
  const Real phi = fields[1];
  const Real D2rho = laplacians[0];
  const Real D2phi = laplacians[1];

  Array<Real,2> mu = { 0.0, 0.0 } ;
#if 0
  /* mu_rho is not needed for the calculations.
     The following expression has not been tested. */
  mu[0] = lambda/4.*(1.+phi2/rho2) + T/2*log((rho2-phi2)/4) - kappa*D2rho;
#endif

  mu[1] = // Eq. (38) in Swift et al.
    - lambda/2.*phi/rho + T/2*log((rho+phi)/(rho-phi)) - kappa*D2phi;

  return mu;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real pressure(Real rho, Real phi) {

  Real p0 = rho*T; // Eq. (40) in Swift et al.
  // note that gradient terms are included in fequilibrium

  return p0;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void hydrovars(int x, int y, int z,
	       const Array4<Real>& f,
	       const Array4<Real>& g,
	       const Array4<Real>& h) {
  Real rho, phi, j[3];
  rho = phi = j[0] = j[1] = j[2] = 0.0;
  for (int i=0; i<nvel; ++i) {
    Real fi, gi;
    fi = f(x,y,z,i);
    gi = g(x,y,z,i);
    rho  += fi;
    phi  += gi;
    j[0] += fi*c[i][0];
    j[1] += fi*c[i][1];
    j[2] += fi*c[i][2];
  }
  h(x,y,z,0) = rho;
  h(x,y,z,1) = phi;
  h(x,y,z,2) = j[0]/rho;
  h(x,y,z,3) = j[1]/rho;
  h(x,y,z,4) = j[2]/rho;
  h(x,y,z,5) = pressure(rho,phi);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Array<Real,nvel> fequilibrium(const Array<Real,2>& fields,
			      const Array<RealVect,2>& gradients,
			      const Array<Real,2>& laplacians,
			      const RealVect& u) {
  Array<Real,nvel> fEq;

  const Real rho = fields[0];
  const Real phi = fields[1];
  const RealVect Drho = gradients[0];
  const RealVect Dphi = gradients[1];
  const Real D2rho = laplacians[0];
  const Real D2phi = laplacians[1];
  const Real pbulk = pressure(rho, phi);

  Real ruu[3][3];
  ruu[0][0] = rho*u[0]*u[0];
  ruu[1][1] = rho*u[1]*u[1];
  ruu[2][2] = rho*u[2]*u[2];
  ruu[0][1] = rho*u[0]*u[1];
  ruu[1][2] = rho*u[1]*u[2];
  ruu[2][0] = rho*u[2]*u[0];

  Real G[3][3]; // Eq. (39) in Swift et al.
  G[0][0] = kappa*Drho[0]*Drho[0] + kappa*Dphi[0]*Dphi[0];
  G[1][1] = kappa*Drho[1]*Drho[1] + kappa*Dphi[1]*Dphi[1];
  G[2][2] = kappa*Drho[2]*Drho[2] + kappa*Dphi[2]*Dphi[2];
  G[0][1] = kappa*Drho[0]*Drho[1] + kappa*Dphi[0]*Dphi[1];
  G[1][2] = kappa*Drho[1]*Drho[2] + kappa*Dphi[1]*Dphi[2];
  G[2][0] = kappa*Drho[2]*Drho[0] + kappa*Dphi[2]*Dphi[0];

  Real fi, sumf = 0.0;
  for (int i=1; i<nvel; ++i) {
    fi  = 3.*w[i]*pbulk;
    fi += 3.*w[i]*rho*(u[0]*c[i][0]+u[1]*c[i][1]+u[2]*c[i][2]);
    fi += 9./2.*w[i]*((ruu[0][0]*c[i][0]+2.*ruu[0][1]*c[i][1])*c[i][0]
		     +(ruu[1][1]*c[i][1]+2.*ruu[1][2]*c[i][2])*c[i][1]
		     +(ruu[2][2]*c[i][2]+2.*ruu[2][0]*c[i][0])*c[i][2]);
    fi -= 3./2.*w[i]*(ruu[0][0]+ruu[1][1]+ruu[2][2]);
    // gradient terms
    fi -= 3.*w[i]*(kappa*rho*D2rho+kappa*phi*D2phi);
    fi += 3.*(wg[i][0][0]*G[0][0]+wg[i][1][1]*G[1][1]+wg[i][2][2]*G[2][2]
	     +wg[i][0][1]*G[0][1]+wg[i][1][2]*G[1][2]+wg[i][2][0]*G[2][0]);
    fEq[i] = fi;
    sumf += fi;
  }
  fEq[0] = rho - sumf;

  return fEq;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Array<Real,nvel> gequilibrium(const Real& phi,
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
  for (int i=1; i<nvel; ++i) {
    gi  = 3.*w[i]*Gamma*mu_phi;
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

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void collide(int x, int y, int z,
	     Array4<Real> f,
	     Array4<Real> g,
	     Array4<Real> h) {

  const Array<Real,2> fields = { h(x,y,z,0), h(x,y,z,1) };
  const RealVect u = { h(x,y,z,2), h(x,y,z,3), h(x,y,z,4) };

  const Array<RealVect,2> gradients = {
    gradient(x,y,z,h,0),
    gradient(x,y,z,h,1),
  };

  const Array<Real,2> laplacians = {
    laplacian(x,y,z,h,0),
    laplacian(x,y,z,h,1),
  };

  const Array<Real,2> mu = chemical_potentials(fields, gradients, laplacians);

  Array<Real,nvel> fEq, gEq;
  fEq = fequilibrium(fields, gradients, laplacians, u);
  gEq = gequilibrium(fields[1], mu[1], u);

  for (int i=0; i<nvel; ++i) {
    f(x,y,z,i) = f(x,y,z,i) - 1./tau_r * (f(x,y,z,i) - fEq[i]);
    g(x,y,z,i) = g(x,y,z,i) - 1./tau_p * (g(x,y,z,i) - gEq[i]);
  }

}

// stream from \vec{x} to \vec{x}+\vec{c}_i (push scheme)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void stream_push(int x, int y, int z,
		 const Array4<Real>& fOld, const Array4<Real>& gOld,
		 const Array4<Real>& fNew, const Array4<Real>& gNew) {
  for (int i=0; i<nvel; ++i) { // push populations to neighbors
    int xp = x + c[i][0];
    int yp = y + c[i][1];
    int zp = z + c[i][2];
    fNew(xp,yp,zp,i) = fOld(x,y,z,i);
    gNew(xp,yp,zp,i) = gOld(x,y,z,i);
  }
}

// push scheme
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void collide_stream(int x, int y, int z,
		    const Array4<Real>& fOld,  const Array4<Real>& gOld,
		    const Array4<Real>& fNew,  const Array4<Real>& gNew,
		    const Array4<Real>& h
			   ) {
  collide(x, y, z, fOld, gOld, h); // collide populations on current site
  stream_push(x, y, z, fOld, gOld, fNew, gNew); // push populations to neighbors
}

inline void LBM_hydrovars(MultiFab& mf, MultiFab& mg, MultiFab& hydrovs) {
  auto const & f = mf.arrays();
  auto const & g = mg.arrays();
  auto const & h = hydrovs.arrays();
  ParallelFor(hydrovs, IntVect(0), [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) {
    hydrovars(x, y, z, f[nbx], g[nbx], h[nbx]);
  });
}

inline void LBM_timestep(const Geometry& geom,
			 MultiFab& fold, MultiFab& gold,
			 MultiFab& fnew, MultiFab& gnew,
			 MultiFab& hydrovs) {
  fold.FillBoundary(geom.periodicity());
  gold.FillBoundary(geom.periodicity());
  hydrovs.FillBoundary(geom.periodicity());
  auto const & fOld = fold.arrays();
  auto const & gOld = gold.arrays();
  auto const & fNew = fnew.arrays();
  auto const & gNew = gnew.arrays();
  auto const & h = hydrovs.arrays();
  IntVect halo = IntVect(1); // need to push from first halo layer into domain
  ParallelFor(fold, halo, [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) {
    collide_stream(x, y, z,
		   fOld[nbx], gOld[nbx],
		   fNew[nbx], gNew[nbx], h[nbx]);
  });
  MultiFab::Swap(fold, fnew, 0, 0, nvel, 0);
  MultiFab::Swap(gold, gnew, 0, 0, nvel, 0);
  LBM_hydrovars(fold, gold, hydrovs);
}

// homogeneous mixture of C1 and C2 with random concentration fluctuations
inline void LBM_init_mixture(MultiFab& mf,
			     MultiFab& mg,
			     MultiFab& hydrovs) {
  const Real C1 = 0.5;
  const Real C2 = 0.5;
  for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
    const Box& valid_box = mfi.validbox();
    const Array4<Real>& f = mf.array(mfi);
    const Array4<Real>& g = mg.array(mfi);
    ParallelForRNG(valid_box, [=] AMREX_GPU_DEVICE(int x, int y, int z, RandomEngine const& engine) {
      const Real rho = C1 + C2;
      const Real phi = (C1-C2)+0.01*RandomNormal(0., 1., engine);
      for (int i=0; i<nvel; i++) {
	f(x,y,z,i) = w[i]*rho;
	g(x,y,z,i) = w[i]*phi;
      }
    });
  }
  LBM_hydrovars(mf, mg, hydrovs);
}

// flat interface between C1 and C2
inline void LBM_init_flat_interface(const Geometry& geom,
				    MultiFab& mf,
				    MultiFab& mg,
				    MultiFab& hydrovs) {
  const IntVect box = geom.Domain().length();

  auto const & f = mf.arrays();
  auto const & g = mg.arrays();
  GpuArray<Real,3> dx = geom.CellSizeArray();
  ParallelFor(mf, IntVect(0), [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) {
    const Real rx = x-box[0]/2;
    const Real ry = y-box[1]/2;
#if (AMREX_SPACEDIM == 2)
    const Real r2 = rx*rx+ry*ry;
#elif (AMREX_SPACEDIM == 3)
    const Real rz = z-box[0]/2;
    const Real r2 = rx*rx+ry*ry+rz*rz;
#endif

    const Real rho=1.0;
    Real phi;

    if (rx < 0.) {
      phi = 0.5;
    } else {
      phi = -0.5;
    }

    for (int i=0; i<nvel; i++) {
      f[nbx](x,y,z,i) = w[i]*rho;
      g[nbx](x,y,z,i) = w[i]*phi;
    }

  });
  LBM_hydrovars(mf, mg, hydrovs);
}

// binary droplet of C1 and C2
inline void LBM_init_droplet(const Real r,
			     const Geometry& geom,
			     MultiFab& mf,
			     MultiFab& mg,
			     MultiFab& hydrovs) {
  const IntVect box = geom.Domain().length();
  const Real R = r*box[0];

  auto const & f = mf.arrays();
  auto const & g = mg.arrays();
  GpuArray<Real,3> dx = geom.CellSizeArray();
  ParallelFor(mf, IntVect(0), [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) {
    const Real rx = x-box[0]/2;
    const Real ry = y-box[1]/2;
#if (AMREX_SPACEDIM == 2)
    const Real r2 = rx*rx+ry*ry;
#elif (AMREX_SPACEDIM == 3)
    const Real rz = z-box[0]/2;
    const Real r2 = rx*rx+ry*ry+rz*rz;
#endif

    const Real rho=1.0;
    Real phi;

    if (r2 < R*R) {
      phi = 0.5;
    } else {
      phi = -0.5;
    }

    for (int i=0; i<nvel; i++) {
      f[nbx](x,y,z,i) = w[i]*rho;
      g[nbx](x,y,z,i) = w[i]*phi;
    }

  });
  LBM_hydrovars(mf, mg, hydrovs);
}

#endif
