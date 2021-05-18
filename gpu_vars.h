#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>



   #if __cplusplus
  extern "C" {
  #endif

extern __managed__ float fq;
extern __managed__ int testVar;
extern __managed__ float drift_E[20];
extern __managed__ int idid,idod,idd;
extern __managed__ double f_drift;

extern __managed__ float tstep;
extern __managed__ float delta;
extern __managed__ float delta_r;
extern __managed__ float wrap_around_radius;
extern __managed__ float ditch_thickness;
extern __managed__ float ditch_depth;
extern __managed__ float surface_drift_vel_factor;

 extern int gpu_drift(MJD_Siggen_Setup *setup, int L, int R, float grid, float ***rho, int q, double *gone);
 extern int ev_calc_gpu(MJD_Siggen_Setup *setup);


  #if __cplusplus
  }
  #endif


