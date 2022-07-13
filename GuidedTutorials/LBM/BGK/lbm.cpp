#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>

#include "lbm.H"

void lbm_main() {

  // default grid parameters
  int nx = 16;
  int nsteps = 100;
  int plot_int = 10;

  // default amplitude of sinusoidal shear wave
  Real A = 0.001;
  
  // input parameters
  ParmParse pp;
  pp.query("nx", nx);
  pp.query("nsteps", nsteps);
  pp.query("plot_int", plot_int);
  pp.query("tau", tau);
  pp.query("A", A);

  // default one ghost/halo layer
  int nghost = 1;
  
  // make Box and Geomtry
  IntVect dom_lo(0, 0, 0);
  IntVect dom_hi(nx-1, nx-1, nx-1);
  IntVect ngs(nghost);
  Array<int,3> periodicity({1,1,1});

  Box domain(dom_lo, dom_hi);

  RealBox real_box({0.,0.,0.},{1.,1.,1.});
  
  Geometry geom(domain, real_box, CoordSys::cartesian, periodicity);

  // make MultiFab
  BoxArray ba(domain);

  DistributionMapping dm(ba);

  MultiFab fold(ba, dm, ncomp, nghost);
  MultiFab fnew(ba, dm, ncomp, nghost);
  MultiFab hydrovars(ba, dm, 1, nghost);

  // set up references to arrays
  auto const & f = fold.arrays(); // LB populations 
  auto const & u = hydrovars.arrays(); // hydrodynamic fields
  
  // INITIALIZE: set up sinusoidal shear wave u_y(x)=A*sin(k*x)
  Real time = 0.0;
  ParallelFor(fold, ngs, [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) {
    Real uy = A*std::sin(2.*M_PI*x/nx);
    for (int i=0; i<ncomp; ++i) {
      f[nbx](x,y,z,i)= fequilibrium(1.0, {0., uy, 0.})(i);
    }
  });
  
  // Write a plotfile of the initial data if plot_int > 0
  if (plot_int > 0) {
    int step = 0;
    ParallelFor(hydrovars, ngs, [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) {
      u[nbx](x,y,z) = 0.;
      for (int i=0; i<ncomp; ++i) {
	u[nbx](x,y,z) += f[nbx](x,y,z,i)*c[i][1];
      }
    });
    const std::string& pltfile = amrex::Concatenate("plt",step,5);
    WriteSingleLevelPlotfile(pltfile, hydrovars, {"u"}, geom, time, step);
  }

  // TIMESTEP
  for (int step=1; step <= nsteps; ++step) {

    fold.FillBoundary(geom.periodicity());

    for (MFIter mfi(fold); mfi.isValid(); ++mfi) {
      const Box& valid_box = mfi.validbox();
      const Array4<Real>& fOld = fold.array(mfi);
      const Array4<Real>& fNew = fnew.array(mfi);
      ParallelFor(valid_box, [=] AMREX_GPU_DEVICE(int x, int y, int z) {
	stream_collide(x, y, z, fOld, fNew);
      });
    }

    MultiFab::Copy(fold, fnew, 0, 0, ncomp, 0);
    
    Print() << "LB step " << step << "\n";
   
    // OUTPUT
    time = static_cast<Real>(step);
    if (plot_int > 0 && step%plot_int ==0) {
      ParallelFor(hydrovars, ngs, [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) {
	u[nbx](x,y,z) = 0.0;
	for (int i=0; i<ncomp; ++i) {
	  u[nbx](x,y,z) += f[nbx](x,y,z,i)*c[i][1];
	}
      });
      const std::string& pltfile = Concatenate("plt",step,5);
      WriteSingleLevelPlotfile(pltfile, hydrovars, {"u"}, geom, time, step);
    }

  }

}

int main(int argc, char* argv[])
{
  amrex::Initialize(argc,argv);

  lbm_main();
    
  amrex::Finalize();
}
