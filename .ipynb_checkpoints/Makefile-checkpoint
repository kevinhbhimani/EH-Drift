# Makefile for signal generation from PPC detectors
#   - uses .c library codes by Karin Lagergren, heavily modified by David Radford
#	- uses nvcc cuda library to compile and link GPU code, modified by Kevin Bhimani
#   Nov 2021
#
# [-lreadline option required for readline, addhistory...]

CC = gcc 
CPP = g++

NVCC := nvcc


CFLAGS = -O3 -Wall 

# The gencode flag depends on the GPU used and need to be modified if the GPU is changed
# gencode and code flags are for following GPUs:
#-gencode=arch=compute_61,code=sm_61 for GTX 1080, GTX 1070, GTX 1060, GTX 1050, GTX 1030, Titan Xp, Tesla P40, Tesla P4
#-gencode=arch=compute_70,code=sm_70 for DGX-1 with Volta, Tesla V100, GTX 1180 (GV104), Titan V, Quadro GV100
#-gencode=arch=compute_80,code=compute_80 for A100 GPUs

NVCCFLAGS = -std=c++14 -rdc=true -lhdf5 -gencode=arch=compute_70,code=compute_70

# NVCCFLAGS = -std=c++14 -rdc=true -I/global/homes/k/kbhimani/.conda/envs/myenv/include -L/global/homes/k/kbhimani/.conda/envs/myenv/lib -lhdf5 -gencode=arch=compute_80,code=sm_80

RM = rm -f

All: ehdrift 

# interactive interface for signal calculation code

ehdrift: ehdrift.c ehd_subs.c ev_gpu.cu gpu_subs.cu charge_drift.cu field_calc.cu read_config.c detector_geometry.c rho_sum_calc.cu mjd_siggen.h detector_geometry.h cyl_point.c gpu_vars.h
	$(NVCC) $(NVCCFLAGS) -o $@ ehdrift.c ehd_subs.c ev_gpu.cu gpu_subs.cu charge_drift.cu field_calc.cu rho_sum_calc.cu read_config.c detector_geometry.c cyl_point.c -lm

FORCE:

clean: 
	$(RM) *.o core* *[~%] *.trace
	$(RM) ehdrift
