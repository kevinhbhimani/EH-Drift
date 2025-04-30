# Makefile for signal generation from PPC detectors
#   - uses .c library codes by Karin Lagergren, heavily modified by David Radford
#   - uses nvcc cuda library to compile and link GPU code, modified by Kevin Bhimani
#   Nov 2021

CC        := gcc
CPP       := g++
NVCC      := nvcc

CFLAGS    := -O3 -Wall

# Detect your CUDA install based on nvcc's location:
CUDA_DIR  := $(shell dirname $(shell dirname $(shell which nvcc)))
CUDA_INC  := $(CUDA_DIR)/include

# Point at the Cray-HDF5 and MPI installs provided by the modules
HDF5_DIR  ?= $(CRAY_HDF5_DIR)
HDF5_INC   = $(HDF5_DIR)/include
HDF5_LIB   = $(HDF5_DIR)/lib

MPI_DIR   ?= $(CRAY_MPICH_DIR)
MPI_INC    = $(MPI_DIR)/include
MPI_LIB    = $(MPI_DIR)/lib

# GPU architecture
NVCCFLAGS := -std=c++14 -rdc=true \
             -gencode=arch=compute_80,code=compute_80 \
             -I$(CUDA_INC) \
             -I$(HDF5_INC) \
             -I$(MPI_INC)

LDFLAGS   := -L$(HDF5_LIB) -lhdf5 \
             -L$(MPI_LIB)  -lmpi

RM        := rm -f

SOURCES   := ehdrift.c \
             ehd_subs.c \
             ev_gpu.cu \
             gpu_subs.cu \
             charge_drift.cu \
             field_calc.cu \
             rho_sum_calc.cu \
             read_config.c \
             detector_geometry.c \
             cyl_point.c

HEADERS   := mjd_siggen.h \
             detector_geometry.h \
             cyl_point.c \
             gpu_vars.h

.PHONY: All clean

All: ehdrift

ehdrift: $(SOURCES) $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -o $@ $(SOURCES) $(LDFLAGS) -lm

clean:
	$(RM) *.o core* *[~%] *.trace ehdrift
