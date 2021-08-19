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
    extern __managed__ float vacuum_gap;
    //extern __managed__ int old_gpu_relax;
    //extern __managed__ int new_gpu_relax;
    extern __managed__ double e_over_E_gpu;
    extern __managed__ double max_dif_gpu;
    extern __managed__ double sum_dif_gpu;

  #if __cplusplus
  }
  #endif


