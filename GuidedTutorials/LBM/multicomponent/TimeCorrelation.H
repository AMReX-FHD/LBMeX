#include <AMReX_MultiFab.H>

// Update time correlation function
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
