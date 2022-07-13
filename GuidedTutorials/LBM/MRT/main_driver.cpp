#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include "StructFact.H"

using namespace amrex;

#include "lbm.H"

void TimeCorrelation(const MultiFab& mfData, MultiFab& mfCorr, const int ncorr, const int steps) {
  auto const & data = mfData.arrays();
  auto const & corr = mfCorr.arrays();

  ParallelFor(mfData, IntVect(0), [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) {
    for (int t=0; t<ncorr; ++t) {
      corr[nbx](x,y,z,t) = (data[nbx](x,y,z,ncorr-1)*data[nbx](x,y,z,ncorr-1-t)
			    + corr[nbx](x,y,z,t)*(steps-1))/steps;
    }
  });

}

// Update data points for time correlation (assumes equally spaced)
void UpdateTimeData(MultiFab& mfData, const MultiFab& mfVal, const int comp, const int ncorr) {
  auto const & data = mfData.arrays();
  auto const & val = mfVal.arrays();

  ParallelFor(mfData, IntVect(0), [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) {
    for (int t=0; t<ncorr-1; ++t) {
      data[nbx](x,y,z,t) = data[nbx](x,y,z,t+1);
    }
    data[nbx](x,y,z,ncorr-1) = val[nbx](x,y,z,comp);
  });

}

void main_driver(const char* argv) {

  // store the current time so we can later compute total run time.
  Real strt_time = ParallelDescriptor::second();
  
  const int nHydroVars = 1 + AMREX_SPACEDIM + AMREX_SPACEDIM*(AMREX_SPACEDIM+1)/2;

  // default grid parameters
  int nx = 16;
  int max_grid_size = 8;
  int nsteps = 100;
  int plot_int = 10;
  int ncorr = 100;
  int comp = 5;

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
  pp.query("ncorr", ncorr);
  pp.query("comp", comp);

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
  MultiFab hydro(ba, dm, nHydroVars, 0);
  MultiFab hydroEq(ba, dm, nHydroVars, 0);

  MultiFab *pfold = &fold;
  MultiFab *pfnew = &fnew;

  // set up references to arrays
  auto const & f = pfold->arrays();    // LB populations
  auto const & m = moments.arrays();   // LB moments
  auto const & h = hydro.arrays();     // hydrodynamic fields
  auto const & hEq = hydroEq.arrays(); // equilibrium fields

  MultiFab mfData(ba, dm, ncorr, 0);
  MultiFab mfCorr(ba, dm, ncorr, 0);

  mfData.setVal(0.);
  mfCorr.setVal(0.);

  Vector<std::string> tnames(ncorr);
  for (int t=0; t<ncorr; ++t) {
    tnames[t] = Concatenate("plt_TC",t,4);
  }

  ///////////////////////////////////////////
  // Initialize structure factor object for analysis
  ///////////////////////////////////////////

  // variables are hydrodynamic moments
  int nStructVars = hydro.nComp();

  Vector< std::string > var_names;
  var_names.resize(nStructVars);

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

  for (; cnt<nStructVars;) {
    name = "m";
    name += std::to_string(cnt);
    var_names[cnt++] = name;
  }

  Vector<Real> var_scaling(nStructVars*(nStructVars+1)/2);
  for (int d=0; d<var_scaling.size(); ++d) {
    if (temperature>0) var_scaling[d] = temperature; else var_scaling[d] = 1.;
  }

  StructFact structFact(ba, dm, var_names, var_scaling);

  // INITIALIZE: set up sinusoidal shear wave u_y(x)=A*sin(k*x)
  Real time = 0.0;
  ParallelFor(*pfold, IntVect(0), [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) {
    const Real uy = A*std::sin(2.*M_PI*x/nx);
    const RealVect u = {0., uy, 0. };
    for (int i=0; i<ncomp; ++i) {
      m[nbx](x,y,z,i) = mequilibrium(density, u)(i);
      f[nbx](x,y,z,i) = fequilibrium(density, u)(i);
    }
    for (int i=0; i<10; ++i) {
      h[nbx](x,y,z,i) = hydrovars(mequilibrium(density, u))(i);
      hEq[nbx](x,y,z,i) = h[nbx](x,y,z,i);
    }
  });

  UpdateTimeData(mfData, hydro, comp, ncorr);

  MultiFab::Subtract(hydro, hydroEq, 0, 0, 1, 0);
  MultiFab::Subtract(hydro, hydroEq, 4, 4, 6, 0);
  structFact.FortStructure(hydro, geom);

  // Write a plotfile of the initial data if plot_int > 0
  if (plot_int > 0) {
    int step = 0;
    const std::string& pltfile = amrex::Concatenate("plt",step,5);
    WriteSingleLevelPlotfile(pltfile, hydro, var_names, geom, time, step);
    const std::string& tcfile = Concatenate("plt_TC",step,5);
    WriteSingleLevelPlotfile(tcfile, mfCorr, tnames, geom, time, step);
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
      const Array4<Real>& hyd = hydro.array(mfi);
      const Array4<Real>& hydEq = hydroEq.array(mfi);
      ParallelForRNG(valid_box, [=] AMREX_GPU_DEVICE(int x, int y, int z, RandomEngine const& engine) {
        stream_collide(x, y, z, fOld, fNew, mom, hyd, hydEq, engine);
      });
    }
    std::swap(pfold,pfnew);

    UpdateTimeData(mfData, hydro, comp, ncorr);
    if (step >= ncorr) TimeCorrelation(mfData, mfCorr, ncorr, step+1);

    MultiFab::Subtract(hydro, hydroEq, 0, 0, 1, 0);
    MultiFab::Subtract(hydro, hydroEq, 4, 4, 6, 0);
    structFact.FortStructure(hydro, geom);

    Print() << "LB step " << step << "\n";

    // OUTPUT
    time = static_cast<Real>(step);
    if (plot_int > 0 && step%plot_int ==0) {
      const std::string& pltfile = Concatenate("plt",step,5);
      WriteSingleLevelPlotfile(pltfile, hydro, var_names, geom, time, step);
      const std::string& tcfile = Concatenate("plt_TC",step,5);
      WriteSingleLevelPlotfile(tcfile, mfCorr, tnames, geom, time, step);
      structFact.WritePlotFile(step, time, geom, "plt_SF");
    }

  }

  // FINAL OUTPUT
  Real eta = 0.0;
  for (int i=0; i<ncorr; ++i) {
    eta += mfCorr.sum(i)/(nx*nx*nx);
  }
  eta /= temperature;

  Print() << "Green-Kubo viscosity for tau = " << tau << " eta = " << eta << std::endl;

  // Call the timer again and compute the maximum difference between the start time
  // and stop time over all processors
  Real stop_time = ParallelDescriptor::second() - strt_time;
  ParallelDescriptor::ReduceRealMax(stop_time);
  amrex::Print() << "Run time = " << stop_time << std::endl;
  
}
