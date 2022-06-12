#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include "StructFact.H"

using namespace amrex;

#include "lbm.H"

void main_driver(const char* argv) {

  // default grid parameters
  int nx = 16;
  int max_grid_size = 8;
  int nsteps = 100;
  int plot_int = 10;

  // default amplitude of sinusoidal shear wave
  Real A = 0.001;
  
  // input parameters
  ParmParse pp;
  pp.query("nx", nx);
  pp.query("max_grid_size", max_grid_size);
  pp.query("nsteps", nsteps);
  pp.query("plot_int", plot_int);
  pp.query("density", density);
  pp.query("temperature", temperature);
  pp.query("tau", tau);
  pp.query("A", A);

  // default one ghost/halo layer
  int nghost = 1;
  
  // make Box and Geomtry
  IntVect dom_lo(0, 0, 0);
  IntVect dom_hi(nx-1, nx-1, nx-1);
  Array<int,3> periodicity({1,1,1});

  Box domain(dom_lo, dom_hi);

  RealBox real_box({0.,0.,0.},{1.,1.,1.});
  
  Geometry geom(domain, real_box, CoordSys::cartesian, periodicity);

  BoxArray ba(domain);

  // split BoxArray "ba" into chunks no larger than "max_grid_size" along a direction
  ba.maxSize(max_grid_size);

  DistributionMapping dm(ba);

  // make MultiFabs
  MultiFab fold(ba, dm, ncomp, nghost);
  MultiFab fnew(ba, dm, ncomp, nghost);
  MultiFab moments(ba, dm, ncomp, 0);
  MultiFab sf(ba, dm, 1+AMREX_SPACEDIM+AMREX_SPACEDIM*(AMREX_SPACEDIM+1)/2, 0);
  //MultiFab sf(ba, dm, ncomp, 0);

  ///////////////////////////////////////////
  // Initialize structure factor object for analysis
  ///////////////////////////////////////////

  // variables are velocities
  int structVars = sf.nComp();

  Vector< std::string > var_names;
  var_names.resize(structVars);

  int cnt = 0;
  std::string name;

  // velx, vely, velz
  var_names[cnt++] = "rho";
  for (int d=0; d<AMREX_SPACEDIM; d++) {
    name = "u";
    name += (120+d);
    var_names[cnt++] = name;
  }

  for (int i=0; i<AMREX_SPACEDIM; ++i) {
    for (int j=i; j<AMREX_SPACEDIM; ++j) {
      name = "p";
      name += (120+i);
      name += (120+j);
      var_names[cnt++] = name;
    }
  }

  for (; cnt<structVars;) {
    name = "m";
    name += std::to_string(cnt);
    var_names[cnt++] = name;
  }

  Vector<Real> var_scaling(structVars*(structVars+1)/2);
  for (int d=0; d<var_scaling.size(); ++d) {
    var_scaling[d] = temperature;
  }

  StructFact structFact(ba, dm, var_names, var_scaling);

  MultiFab *pfold = &fold;
  MultiFab *pfnew = &fnew;

  // set up references to arrays
  auto const & f = pfold->arrays();  // LB populations
  auto const & m = moments.arrays(); // LB moments
  auto const & h = sf.arrays();      // hydrodynamic fields

  // INITIALIZE: set up sinusoidal shear wave u_y(x)=A*sin(k*x)
  Real time = 0.0;
  ParallelFor(*pfold, IntVect(0), [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) {
    const Real uy = A*std::sin(2.*M_PI*x/nx);
    const RealVect u = {0., uy, 0. };
    for (int i=0; i<ncomp; ++i) {
      m[nbx](x,y,z,i) = mequilibrium(density, u)[i];
      f[nbx](x,y,z,i) = fequilibrium(density, u)[i];
    }
    for (int i=0; i<10; ++i) {
      h[nbx](x,y,z,i) = hydrovars(mequilibrium(density, u))[i];
    }
  });
  //MultiFab::Copy(sf, moments, 0, 0, structVars, 0);
  sf.plus(-density, 0, 1);
  for (int i=0; i<AMREX_SPACEDIM; ++i) {
    MultiFab::Divide(sf, moments, 0, 1+i, 1, 0);
  }
  structFact.FortStructure(sf, geom);

  // Write a plotfile of the initial data if plot_int > 0
  if (plot_int > 0) {
    int step = 0;
    const std::string& pltfile = amrex::Concatenate("plt",step,5);
    WriteSingleLevelPlotfile(pltfile, sf, var_names, geom, time, step);
    structFact.WritePlotFile(0, 0., geom, "plt_SF");
  }

  Print() << "LB initialized\n";

  // TIMESTEP
  for (int step=1; step <= nsteps; ++step) {

    pfold->FillBoundary(geom.periodicity());

    for (MFIter mfi(*pfold); mfi.isValid(); ++mfi) {
      const Box& valid_box = mfi.validbox();
      const Array4<Real>& fOld = pfold->array(mfi);
      const Array4<Real>& fNew = pfnew->array(mfi);
      const Array4<Real>& mom = moments.array(mfi);
      const Array4<Real>& hydrovars = sf.array(mfi);
      ParallelForRNG(valid_box, [=] AMREX_GPU_DEVICE(int x, int y, int z, RandomEngine const& engine) {
        stream_collide(x, y, z, mom, fOld, fNew, hydrovars, engine);
      });
    }
    //MultiFab::Copy(sf, moments, 0, 0, structVars, 0);
    sf.plus(-density, 0, 1);
    for (int i=0; i<AMREX_SPACEDIM; ++i) {
      MultiFab::Divide(sf, moments, 0, 1+i, 1, 0);
    }
    structFact.FortStructure(sf, geom);

    std::swap(pfold,pfnew);
    
    Print() << "LB step " << step << "\n";

    // OUTPUT
    time = static_cast<Real>(step);
    if (plot_int > 0 && step%plot_int ==0) {
      const std::string& pltfile = Concatenate("plt",step,5);
      WriteSingleLevelPlotfile(pltfile, sf, var_names, geom, time, step);
      structFact.WritePlotFile(step, time, geom, "plt_SF");
    }

  }

}
