AMREX_HOME ?= ../../../../amrex

DEBUG	= FALSE

DIM	= 3

COMP    = gnu

USE_MPI   = TRUE
USE_OMP   = FALSE
USE_CUDA  = FALSE
USE_HIP   = FALSE

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ./Make.package

include ../../../../../AMReX-FHD/FHDeX/src_common/Make.package
VPATH_LOCATIONS   += ../../../../../AMReX-FHD/FHDeX/src_common/
INCLUDE_LOCATIONS += ../../../../../AMReX-FHD/FHDeX/src_common/

include ../../../../../AMReX-FHD/FHDeX/src_analysis/Make.package
VPATH_LOCATIONS   += ../../../../../AMReX-FHD/FHDeX/src_analysis/
INCLUDE_LOCATIONS += ../../../../../AMReX-FHD/FHDeX/src_analysis/

include $(AMREX_HOME)/Src/Base/Make.package

include $(AMREX_HOME)/Tools/GNUMake/Make.rules

ifeq ($(findstring cgpu, $(HOST)), cgpu)
  CXXFLAGS += $(FFTW)
endif

ifeq ($(USE_CUDA),TRUE)
  LIBRARIES += -lcufft
else
  LIBRARIES += -L$(FFTW_DIR) -lfftw3_mpi -lfftw3
endif
