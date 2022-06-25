#ifndef LBM_H_
#define LBM_H_

#include "LBM_d3q19.H"

Real density = 1.0;
Real temperature = 0.0;
Real tau = 0.9,
  tau_s=1.0,
  tau_p=1.0;
Real alpha = 1.0;

inline void gradient_laplacian(int x, int y, int z, const Array4<Real>& field, const Array4<Real> gradient, const Array4<Real> laplacian) {
  laplacian(x,y,z) = 0.0;
  for (int dir=0; dir<3; dir++) gradient(x,y,z,dir) = 0.0;
  for (int i=0; i<nvel; i++) {
    int xp = x + c[i][0];
    int yp = y + c[i][1];
    int zp = z + c[i][2];
    for (int dir=0; dir<3; dir++) {
      gradient(x,y,z,dir) += w[i]/cs2*field(xp,yp,zp)*c[i][dir];
    }
    laplacian(x,y,z) += 2.*w[i]/cs2*(field(xp,yp,zp)-field(x,y,z));
  }
}

inline void calc_chemical_potentials(int x, int y, int z) {
  const Real alpha2 = alpha*alpha;
  const Real rho = density[x][y][z];
  const Real phi = phi[x][y][z];
  const Real psi = psi[x][y][z];
  const Real D2rho = laplace_rho[x][y][z];
  const Real D2phi = laplace_phi[x][y][z];
  const Real D2psi = laplace_psi[x][y][z];

#if 0
  /* mu_rho is not needed for the calculations.
     The following expression has not been tested.
     (Eq. (38) in Semprebon et al. may contain typos) */
  mu_rho[x][y][z] =
      kappa1/8.*(rho+psi-phi)*(rho+phi-psi-2.)*(rho+phi-psi-1.)
    + kappa2/8.*(rho-phi-psi)*(rho-phi-psi-2.)*(rho-phi-psi-1.)
    - alpha2/4.*((kappa1+kappa2)*(D2rho-D2psi)-(kappa1-kappa2)*D2phi);
#endif

  mu_phi[x][y][z] = // Eq. (39) in Semprebon et al.
      kappa1/8.*(rho+phi-psi)*(rho+phi-psi-2.)*(rho+phi-psi-1.)
    - kappa2/8.*(rho-phi-psi)*(rho-phi-psi-2.)*(rho-phi-psi-1.)
    - alpha2/4.*((kappa1-kappa2)*(D2rho-D2psi)+(kappa1+kappa2)*D2phi);

  mu_psi[x][y][z] = // Eq. (40) in Semprebon et al.
    - kappa1/8.*(rho+phi-psi)*(rho+phi-psi-2.)*(rho+phi-psi-1.)
    - kappa2/8.*(rho-phi-psi)*(rho-phi-psi-2.)*(rho-phi-psi-1.)
    + kappa3*psi*(psi-1.)*(2.*psi-1.)
    + alpha2/4.*((kappa1+kappa2)*D2rho+(kappa1-kappa2)*D2phi
		 -(kappa1+kappa2+4.*kappa3)*D2psi);

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

inline Array<Real,nvel> fequilibrium(const Real& rho, const RealVect& u) {
  Array<Real,nvel> fEq;
  for (int i=0; i<nvel; ++i) {
    Real uc = u.dotProduct(c[i]);
    fEq[i] = w[i] * rho * ( 1. + uc/cs2 + (uc*uc - u.dotProduct(u)*cs2)/(2.*cs2*cs2) );
  }
  return fEq;
}

inline void calc_feq(int x, int y, int z) {
  const Real rho = density[x][y][z];
  const Real phi = phi[x][y][z];
  const Real psi = psi[x][y][z];
  const Real p0 = pressure[x][y][z];
  const Real *u = u[x][y][z];
  const Real *Drho = density_gradient[x][y][z];
  const Real *Dphi = phi_gradient[x][y][z];
  const Real *Dpsi = psi_gradient[x][y][z];
  const Real D2rho = laplace_rho[x][y][z];
  const Real D2phi = laplace_phi[x][y][z];
  const Real D2psi = laplace_psi[x][y][z];
  Real fi, ruu[3][3], G[3][3];
  int i;

  ruu[0][0] = rho*u[0]*u[0];
  ruu[1][1] = rho*u[1]*u[1];
  ruu[2][2] = rho*u[2]*u[2];
  ruu[0][1] = rho*u[0]*u[1];
  ruu[1][2] = rho*u[1]*u[2];
  ruu[2][0] = rho*u[2]*u[0];

  G[0][0] = kappa_rr*Drho[0]*Drho[0]+kappa_pp*Dphi[0]*Dphi[0]+kappa_ss*Dpsi[0]*Dpsi[0];
  G[1][1] = kappa_rr*Drho[1]*Drho[1]+kappa_pp*Dphi[1]*Dphi[1]+kappa_ss*Dpsi[1]*Dpsi[1];
  G[2][2] = kappa_rr*Drho[2]*Drho[2]+kappa_pp*Dphi[2]*Dphi[2]+kappa_ss*Dpsi[2]*Dpsi[2];
  G[0][1] = kappa_rr*Drho[0]*Drho[1]+kappa_pp*Dphi[0]*Dphi[1]+kappa_ss*Dpsi[0]*Dpsi[1];
  G[1][2] = kappa_rr*Drho[1]*Drho[2]+kappa_pp*Dphi[1]*Dphi[2]+kappa_ss*Dpsi[1]*Dpsi[2];
  G[2][0] = kappa_rr*Drho[2]*Drho[0]+kappa_pp*Dphi[2]*Dphi[0]+kappa_ss*Dpsi[2]*Dpsi[0];

  Real sumf = 0.0;
  for (i=1; i<nvel; ++i) { // Eq. (52) Semprebon et al.
    fi  = 3.*w[i]*p0;
    fi += 3.*w[i]*rho*(u[0]*e[i][0]+u[1]*e[i][1]+u[2]*e[i][2]);
    fi += 9./2.*w[i]*((ruu[0][0]*e[i][0]+2.*ruu[0][1]*e[i][1])*e[i][0]
			 +(ruu[1][1]*e[i][1]+2.*ruu[1][2]*e[i][2])*e[i][1]
			 +(ruu[2][2]*e[i][2]+2.*ruu[2][0]*e[i][0])*e[i][2]);
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
    feq[x][y][z][i] = fi;
    sumf += fi;
  }
  feq[x][y][z][0] = rho - sumf;
}

inline void calc_geq(int x, int y, int z) {
  const Real phi = phi[x][y][z];
  const Real mu_p = mu_phi[x][y][z];
  const Real *u = u[x][y][z];
  Real gi, puu[3][3];
  int i;

  puu[0][0] = phi*u[0]*u[0];
  puu[1][1] = phi*u[1]*u[1];
  puu[2][2] = phi*u[2]*u[2];
  puu[0][1] = phi*u[0]*u[1];
  puu[1][2] = phi*u[1]*u[2];
  puu[2][0] = phi*u[2]*u[0];

  Real sumg = 0.0;
  for (i=1; i<nvel; ++i) { // Eq. (53) Semprebon et al.
    gi  = 3.*w[i]*gamma_p*mu_p;
    gi += 3.*w[i]*phi*(u[0]*e[i][0]+u[1]*e[i][1]+u[2]*e[i][2]);
    gi += 9./2.*w[i]*((puu[0][0]*e[i][0]+2.*puu[0][1]*e[i][1])*e[i][0]
			 +(puu[1][1]*e[i][1]+2.*puu[1][2]*e[i][2])*e[i][1]
			 +(puu[2][2]*e[i][2]+2.*puu[2][0]*e[i][0])*e[i][2]);
    gi -= 3./2.*w[i]*(puu[0][0]+puu[1][1]+puu[2][2]);
    geq[x][y][z][i] = gi;
    sumg += gi;
  }
  geq[x][y][z][0] = phi - sumg;
}

inline void calc_equilibrium(int x, int y, int z) {
  calc_gradient_laplacian(x,y,z, density, density_gradient, laplace_rho);
  calc_gradient_laplacian(x,y,z, phi, phi_gradient, laplace_phi);
  calc_gradient_laplacian(x,y,z, psi, psi_gradient, laplace_psi);
  calc_chemical_potentials(x,y,z);
  calc_feq(x,y,z);
  calc_geq(x,y,z);
  calc_keq(x,y,z);
}

inline void calc_moments(int x, int y, int z) {
  Real rho, phi, psi, j[3], fi, gi, ki;
  int i;
  rho = phi = psi = j[0] = j[1] = j[2] = 0.0;
  for (i=0; i<nvel; ++i) {
    fi = f[x][y][z][i];
    gi = g[x][y][z][i];
    ki = k[x][y][z][i];
    rho  += fi;
    phi  += gi;
    psi  += ki;
    j[0] += fi*e[i][0];
    j[1] += fi*e[i][1];
    j[2] += fi*e[i][2];
  }
  density[x][y][z] = rho;
  phi[x][y][z] = phi;
  psi[x][y][z] = psi;
  u[x][y][z][0] = j[0]/rho;
  u[x][y][z][1] = j[1]/rho;
  u[x][y][z][2] = j[2]/rho;
  pressure[x][y][z] = pressure(rho,phi,psi);
}

inline void collide_stream(int x, int y, int z) {
  int i, xnew, ynew, znew;
  calc_equilibrium(x,y,z);
  for (i=0; i<nvel; ++i) {
    xnew = x + e[i][0];
    ynew = y + e[i][1];
    znew = z + e[i][2];
    fnew[xnew][ynew][znew][i] = f[x][y][z][i] - (f[x][y][z][i] - feq[x][y][z][i])/tau_r;
    gnew[xnew][ynew][znew][i] = g[x][y][z][i] - (g[x][y][z][i] - geq[x][y][z][i])/tau_p;
    knew[xnew][ynew][znew][i] = k[x][y][z][i] - (k[x][y][z][i] - keq[x][y][z][i])/tau_s;
  }
}

inline void relax_moments(int x, int y, int z, const Array4<Real>& m) {
  const Real gamma = 1. - 1./tau;
  const Array<Real,nvel> mEq = mequilibrium(x, y, z, m);
  Array<Real,nvel> mom;
  for (int i=0; i<nvel; ++i) {
    mom[i] = m(x,y,z,i) = mEq[i] + gamma * (m(x,y,z,i) - mEq[i]);
  }
}

inline void collide(int x, int y, int z, const Array4<Real>& f, const Array4<Real>& msite, const Array4<Real>& hsite, RandomEngine const& engine) {
  moments(x, y, z, f, msite);
  relax_moments(x, y, z, msite);
  populations(x, y, z, f, msite);
}

// stream from \vec{x}-\vec{c}_i to \vec{x} (pull scheme)
inline void stream_pull(int x, int y, int z, const Array4<Real>& fOld, const Array4<Real>& fNew) {
  for (int i=0; i<nvel; ++i) { // pull populations from neighbors
    int xm = x - c[i][0];
    int ym = y - c[i][1];
    int zm = z - c[i][2];
    fNew(x,y,z,i) = fOld(xm,ym,zm,i);
  }
}

// pull scheme
inline void stream_collide(int x, int y, int z, const Array4<Real>& m, const Array4<Real>& fOld, const Array4<Real>& fNew, const Array4<Real>& h, RandomEngine const& engine) {
  stream_pull(x, y, z, fOld, fNew);     // pull populations from neighbors
  collide(x, y, z, fNew, m, h, engine); // collide populations on current site
}

#endif
