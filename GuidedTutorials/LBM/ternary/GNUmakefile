AMREX_HOME ?= ../../../../amrex
FHDEX_HOME ?= ../../../../../AMReX-FHD/FHDeX

DEBUG	= FALSE

DIM	= 3

COMP    = gnu

USE_MPI   = TRUE
USE_OMP   = FALSE
USE_CUDA  = FALSE
USE_HIP   = FALSE

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ./Make.package

include $(FHDEX_HOME)/src_common/Make.package
VPATH_LOCATIONS   += $(FHDEX_HOME)/src_common/
INCLUDE_LOCATIONS += $(FHDEX_HOME)/src_common/

include $(FHDEX_HOME)/src_analysis/Make.package
VPATH_LOCATIONS   += $(FHDEX_HOME)/src_analysis/
INCLUDE_LOCATIONS += $(FHDEX_HOME)/src_analysis/

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
