# Makefile for signal generation from PPC detectors
#   - uses .c library codes by Karin Lagergren, heavily modified by David Radford
#
# [-lreadline option required for readline, addhistory...]

CC = gcc
CPP = g++
NVCC := nvcc


CFLAGS = -O3 -Wall
NVCCFLAGS = -rdc=true

RM = rm -f

# common files and headers
mk_signal_files = calc_signal.c cyl_point.c detector_geometry.c fields.c point.c read_config.c
mk_signal_headers = calc_signal.h cyl_point.h detector_geometry.h fields.h mjd_siggen.h point.h

All: stester mjd_fieldgen ehdrift ehd_siggen

# interactive interface for signal calculation code
stester: $(mk_signal_files) $(mk_signal_headers) signal_tester.c
	$(CC) $(CFLAGS) -o $@ $(mk_signal_files) signal_tester.c -lm -lreadline

mjd_fieldgen: mjd_fieldgen.c read_config.c detector_geometry.c mjd_siggen.h detector_geometry.h
	$(CC) $(CFLAGS) -o $@ mjd_fieldgen.c read_config.c detector_geometry.c -lm

ehd_siggen: ehd_siggen.c ehd_subs.c $(mk_signal_files) $(mk_signal_headers) 
	$(CC) $(CFLAGS) -o $@ ehd_siggen.c ehd_subs.c read_config.c detector_geometry.c fields.c cyl_point.c -lm

ehdrift: ehdrift.c cuda_hello_world.cu ehd_subs.c read_config.c detector_geometry.c mjd_siggen.h detector_geometry.h
	$(NVCC) $(NVCCFLAGS) -o $@ ehdrift.c cuda_hello_world.cu ehd_subs.c read_config.c detector_geometry.c -lm

FORCE:

clean: 
	$(RM) *.o core* *[~%] *.trace
	$(RM) stester mjd_fieldgen ehdrift ehd_siggen
