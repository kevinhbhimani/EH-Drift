/*
  gpu drift perform diffusion and drift in electric field of electron and hole signals.
  author:           Kevin H Bhimani
  first written:    Nov 2021
*/
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "mjd_siggen.h"
#include "detector_geometry.h"
#include "gpu_vars.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


extern "C" int gpu_drift(MJD_Siggen_Setup *setup, int L, int R, float grid, double ***rho, int q, GPU_data *gpu_setup);
extern "C" void set_rho_zero_gpu(GPU_data *gpu_setup, int L, int R, int num_blocks, int num_threads);
extern "C" void update_impurities_gpu(GPU_data *gpu_setup, int L, int R, int num_blocks, int num_threads, double e_over_E, float grid);

__managed__ double drift_E[20] = {0.000,  100.,  160.,  240.,  300.,  500.,  600., 750.0, 1000., 1250., 1500., 1750., 2000., 2500., 3000., 3500., 4000., 4500., 5000., 1e10};
__managed__ float fq;
__managed__ int idid,idod,idd;
__managed__ double f_drift;
__managed__ float tstep;
__managed__ float delta;
__managed__ float delta_r;
__managed__ float wrap_around_radius;
__managed__ float ditch_thickness;
__managed__ float ditch_depth;
__managed__ float surface_drift_vel_factor;

__global__ void reset_rho(int L, int R, double *rho, int max_threads){

    int r = blockIdx.x%R;
    int z = (floorf(blockIdx.x/R) * max_threads) + threadIdx.x;
    
    if(r>=R || z>=L){
      return;
    }
    rho[(1*(L+1)*(R+1))+((R+1)*z)+r] = rho[(2*(L+1)*(R+1))+((R+1)*z)+r] = 0;
}
/* 
  Uses diffusion eqaution to calculate the diffusion coefficients and stores them in appropriate arrays
 */
__global__ void gpu_diffusion(int L, int R, float grid, double *rho,int q, double *v, char *point_type,  
  double *dr, double *dz, double *s1, double *s2, double *drift_offset, double *drift_slope, int max_threads, double *deltaez_array, double *deltaer_array){

  int r = blockIdx.x%R;
  int z = (floorf(blockIdx.x/R) * max_threads) + threadIdx.x;
  
  if(r==0 || z==0 || r>=R || z>=(L-2)){
    return;
  }
  int i, new_gpu_relax = 0;
  float E, E_r, E_z;
  double ve_z, ve_r, deltaez, deltaer;

  if (rho[(0*(L+1)*(R+1))+((R+1)*z)+r] < 1.0e-14) {
    atomicAdd(&rho[(1*(L+1)*(R+1))+((R+1)*z)+r], rho[(0*(L+1)*(R+1))+((R+1)*z)+r]);
    return;
  }

  // calc E in r-direction
  if (r == 1) {  // r = 0; symmetry implies E_r = 0
    E_r = 0;
  } else if (point_type[((R+1)*z)+r] == CONTACT_EDGE) {
    E_r = ((v[(new_gpu_relax*(L+1)*(R+1))+((R+1)*z)+r] - v[(new_gpu_relax*(L+1)*(R+1))+((R+1)*z)+r+1])*dr[(1*(L+1)*(R+1))+((R+1)*z)+r] +
          (v[(new_gpu_relax*(L+1)*(R+1))+((R+1)*z)+r-1] - v[(new_gpu_relax*(L+1)*(R+1))+((R+1)*z)+r])*dr[(0*(L+1)*(R+1))+((R+1)*z)+r]) / (0.2*grid);
  } else if (point_type[((R+1)*z)+r] < INSIDE &&
            point_type[((R+1)*z)+r-1] == CONTACT_EDGE) {
    E_r =  (v[(new_gpu_relax*(L+1)*(R+1))+((R+1)*z)+r-1] - v[(new_gpu_relax*(L+1)*(R+1))+((R+1)*z)+r]) * dr[(1*(L+1)*(R+1))+((R+1)*z)+r-1] / ( 0.1*grid) ;
  } else if (point_type[((R+1)*z)+r] < INSIDE &&
            point_type[((R+1)*z)+r+1] == CONTACT_EDGE) {
    E_r =  (v[(new_gpu_relax*(L+1)*(R+1))+((R+1)*z)+r] - v[(new_gpu_relax*(L+1)*(R+1))+((R+1)*z)+r+1]) * dr[(0*(L+1)*(R+1))+((R+1)*z)+r+1] / ( 0.1*grid) ;
  } else if (r == R-1) {
    E_r = (v[(new_gpu_relax*(L+1)*(R+1))+((R+1)*z)+r-1] - v[(new_gpu_relax*(L+1)*(R+1))+((R+1)*z)+r])/(0.1*grid);
  } else {
    E_r = (v[(new_gpu_relax*(L+1)*(R+1))+((R+1)*z)+r-1] - v[(new_gpu_relax*(L+1)*(R+1))+((R+1)*z)+r+1])/(0.2*grid);
  }
  // calc E in z-direction
  // enum point_types{PC, HVC, INSIDE, PASSIVE, PINCHOFF, DITCH, DITCH_EDGE, CONTACT_EDGE};
  if (point_type[((R+1)*z)+r] == CONTACT_EDGE) {
    E_z = ((v[(new_gpu_relax*(L+1)*(R+1))+((R+1)*z)+r] - v[(new_gpu_relax*(L+1)*(R+1))+((R+1)*(z+1))+r])*dz[(1*(L+1)*(R+1))+((R+1)*z)+r] +
          (v[(new_gpu_relax*(L+1)*(R+1))+((R+1)*(z-1))+r] - v[(new_gpu_relax*(L+1)*(R+1))+((R+1)*z)+r])*dz[(0*(L+1)*(R+1))+((R+1)*z)+r]) / (0.2*grid);
  } else if (point_type[((R+1)*z)+r] < INSIDE &&
            point_type[((R+1)*(z-1))+r] == CONTACT_EDGE) {
    E_z =  (v[(new_gpu_relax*(L+1)*(R+1))+((R+1)*(z-1))+r] - v[(new_gpu_relax*(L+1)*(R+1))+((R+1)*z)+r]) * dz[(1*(L+1)*(R+1))+((R+1)*(z-1))+r] / ( 0.1*grid) ;
  } else if (point_type[((R+1)*z)+r] < INSIDE &&
            point_type[((R+1)*(z+1))+r] == CONTACT_EDGE) {
    E_z =  (v[(new_gpu_relax*(L+1)*(R+1))+((R+1)*z)+r] - v[(new_gpu_relax*(L+1)*(R+1))+((R+1)*(z+1))+r]) * dz[(0*(L+1)*(R+1))+((R+1)*(z+1))+r] / ( 0.1*grid) ;
  } else if (z == 1) {
    E_z = (v[(new_gpu_relax*(L+1)*(R+1))+((R+1)*z)+r] - v[(new_gpu_relax*(L+1)*(R+1))+((R+1)*(z+1))+r])/(0.1*grid);
  } else if (z == L-1) {
    E_z = (v[(new_gpu_relax*(L+1)*(R+1))+((R+1)*(z-1))+r] - v[(new_gpu_relax*(L+1)*(R+1))+((R+1)*z)+r])/(0.1*grid);
  } else {
    E_z = (v[(new_gpu_relax*(L+1)*(R+1))+((R+1)*(z-1))+r] - v[(new_gpu_relax*(L+1)*(R+1))+((R+1)*(z+1))+r])/(0.2*grid);
  }

  /* do diffusion to neighboring pixels */
  deltaez = deltaer = ve_z = ve_r = 0;

  E = fabs(E_z);
  if (E > 1.0) {
    for (i=0; E > drift_E[i+1]; i++);
    ve_z = (drift_offset[i] + drift_slope[i]*(E - drift_E[i]));
    deltaez = grid * ve_z * f_drift / E;
    }

  E = fabs(E_r);
  if (E > 1.0) {
    for (i=0; E > drift_E[i+1]; i++);
    ve_r = (drift_offset[i] + drift_slope[i]*(E - drift_E[i]));
    deltaer = grid * ve_r * f_drift / E;
    }

  if (0 && r == 100 && z == 10)
    printf("r z: %d %d; E_r deltaer: %f %f; E_z deltaez: %f %f; rho[0] = %f\n",
          r, z, E_r, deltaer, E_z, deltaez, rho[(0*(L+1)*(R+1))+((R+1)*z)+r]);

  /* reduce diffusion at passivated surfaces by a factor of surface_drift_vel_factor */
  if (1 &&
      ((r == idid && z < idd) ||
      (r < idid  && z == 1 ) ||
      (r >= idid && r <= idod && z == idd))) {
    // assume 2-micron-thick roughness/passivation in z
    deltaer *= surface_drift_vel_factor;
    deltaez *= surface_drift_vel_factor * grid/0.002; // * grid/0.002;
    }


  if (0 && z == 1) 
  printf("r,z = %d, %d E_r,z = %f, %f  deltaer,z = %f, %f  s1,s2 = %f, %f\n",
                    r, z, E_r, E_z, deltaer, deltaez, s1[r], s2[r]);

  deltaez_array[((R+1)*z)+r] = deltaez;
  deltaer_array[((R+1)*z)+r] = deltaer;

}

/*
Performs diffusion to values in rho[0] and stores them to rho[1].
Atomic operations are need to perform the update without interference from any other threads.
*/
__global__ void diff_update(int L, int R, double *rho, char *point_type, double *s1, double *s2, double *deltaez_array, double *deltaer_array, int max_threads){

  int r = blockIdx.x%R;
  int z = (floorf(blockIdx.x/R) * max_threads) + threadIdx.x;
  
  if(r==0 || z==0 || r>=R || z>=(L-2)){
    return;
  }
  if (rho[(0*(L+1)*(R+1))+((R+1)*z)+r] < 1.0e-14) {
    return;
  }
  double deltaez = deltaez_array[((R+1)*z)+r];
  double deltaer = deltaer_array[((R+1)*z)+r];
    atomicAdd(&rho[(1*(L+1)*(R+1))+((R+1)*z)+r], rho[(0*(L+1)*(R+1))+((R+1)*z)+r]);

  if (r < R-1 && point_type[((R+1)*z)+r+1] != DITCH) {
    atomicAdd(&rho[(1*(L+1)*(R+1))+((R+1)*z)+r+1], rho[(0*(L+1)*(R+1))+((R+1)*z)+r]*deltaer * s1[r] * (double) (r-1) / (double) (r));
    atomicAdd(&rho[(1*(L+1)*(R+1))+((R+1)*z)+r],-rho[(0*(L+1)*(R+1))+((R+1)*z)+r]*deltaer * s1[r]);
    }

  if (z > 1 && point_type[((R+1)*(z-1))+r] != DITCH) {
    atomicAdd(&rho[(1*(L+1)*(R+1))+((R+1)*(z-1))+r], rho[(0*(L+1)*(R+1))+((R+1)*z)+r]*deltaez);
    atomicAdd(&rho[(1*(L+1)*(R+1))+((R+1)*z)+r], -rho[(0*(L+1)*(R+1))+((R+1)*z)+r]*deltaez);
    }

  if (z < L-1 && point_type[((R+1)*(z+1))+r] != DITCH) {
    atomicAdd(&rho[(1*(L+1)*(R+1))+((R+1)*(z+1))+r], rho[(0*(L+1)*(R+1))+((R+1)*z)+r]*deltaez);
    atomicAdd(&rho[(1*(L+1)*(R+1))+((R+1)*z)+r], -rho[(0*(L+1)*(R+1))+((R+1)*z)+r]*deltaez);
    }


  if (r > 2 && point_type[((R+1)*z)+r-1] != DITCH) {

    atomicAdd(&rho[(1*(L+1)*(R+1))+((R+1)*z)+(r-1)], rho[(0*(L+1)*(R+1))+((R+1)*z)+r]*deltaer * s2[r] * (double) (r-1) / (double) (r-2));
    atomicAdd(&rho[(1*(L+1)*(R+1))+((R+1)*z)+r], -rho[(0*(L+1)*(R+1))+((R+1)*z)+r]*deltaer * s2[r]);
  }
}

/* 
  Sees where the charge would drift in the field and stores the new location
 */
__global__ void gpu_self_repulsion(int L, int R, float grid, double *rho,int q, double *v, char *point_type,  double *dr, double *dz, 
  double *s1, double *s2, double *drift_offset, double *drift_slope, int max_threads, double *fr_array, double *fz_array, int *i_array, int *k_array){

    int new_gpu_relax = 0;
    float E, E_r, E_z;
    double dre, dze;
    double ve_z, ve_r;

    int r = blockIdx.x%R;
    int z = (floorf(blockIdx.x/R) * max_threads) + threadIdx.x;
    
    if(r==0 || z==0 || r>=R || z>=(L-2)){
      return;
    }
  
  if (rho[(1*(L+1)*(R+1))+((R+1)*z)+r] < 1.0e-14) {
    atomicAdd(&rho[(2*(L+1)*(R+1))+((R+1)*z)+r], rho[(1*(L+1)*(R+1))+((R+1)*z)+r]);
    return;
  }
  // need to r-calculate all the fields
  // calc E in r-direction
  if (r == 1) {  // r = 0; symmetry implies E_r = 0
    E_r = 0;
  } else if (point_type[((R+1)*z)+r] == CONTACT_EDGE) {
    E_r = ((v[(new_gpu_relax*(L+1)*(R+1))+((R+1)*z)+r] - v[(new_gpu_relax*(L+1)*(R+1))+((R+1)*z)+r+1])*dr[(1*(L+1)*(R+1))+((R+1)*z)+r] +
          (v[(new_gpu_relax*(L+1)*(R+1))+((R+1)*z)+r-1] - v[(new_gpu_relax*(L+1)*(R+1))+((R+1)*z)+r])*dr[(0*(L+1)*(R+1))+((R+1)*z)+r]) / (0.2*grid);
  } else if (point_type[((R+1)*z)+r] < INSIDE &&
            point_type[((R+1)*z)+r-1] == CONTACT_EDGE) {
    E_r =  (v[(new_gpu_relax*(L+1)*(R+1))+((R+1)*z)+r-1] - v[(new_gpu_relax*(L+1)*(R+1))+((R+1)*z)+r]) * dr[(1*(L+1)*(R+1))+((R+1)*z)+r-1] / ( 0.1*grid) ;
  } else if (point_type[((R+1)*z)+r] < INSIDE &&
            point_type[((R+1)*z)+r+1] == CONTACT_EDGE) {
    E_r =  (v[(new_gpu_relax*(L+1)*(R+1))+((R+1)*z)+r] - v[(new_gpu_relax*(L+1)*(R+1))+((R+1)*z)+r+1]) * dr[(0*(L+1)*(R+1))+((R+1)*z)+r+1] / ( 0.1*grid) ;
  } else if (r == R-1) {
    E_r = (v[(new_gpu_relax*(L+1)*(R+1))+((R+1)*z)+r-1] - v[(new_gpu_relax*(L+1)*(R+1))+((R+1)*z)+r])/(0.1*grid);
  } else {
    E_r = (v[(new_gpu_relax*(L+1)*(R+1))+((R+1)*z)+r-1] - v[(new_gpu_relax*(L+1)*(R+1))+((R+1)*z)+r+1])/(0.2*grid);
  }
  // calc E in z-direction
  // enum point_types{PC, HVC, INSIDE, PASSIVE, PINCHOFF, DITCH, DITCH_EDGE, CONTACT_EDGE};
  if (point_type[((R+1)*z)+r] == CONTACT_EDGE) {
    E_z = ((v[(new_gpu_relax*(L+1)*(R+1))+((R+1)*z)+r] - v[(new_gpu_relax*(L+1)*(R+1))+((R+1)*(z+1))+r])*dz[(1*(L+1)*(R+1))+((R+1)*z)+r] +
          (v[(new_gpu_relax*(L+1)*(R+1))+((R+1)*(z-1))+r] - v[(new_gpu_relax*(L+1)*(R+1))+((R+1)*z)+r])*dz[(0*(L+1)*(R+1))+((R+1)*z)+r]) / (0.2*grid);
  } else if (point_type[((R+1)*z)+r] < INSIDE &&
            point_type[((R+1)*(z-1))+r] == CONTACT_EDGE) {
    E_z =  (v[(new_gpu_relax*(L+1)*(R+1))+((R+1)*(z-1))+r] - v[(new_gpu_relax*(L+1)*(R+1))+((R+1)*z)+r]) * dz[(1*(L+1)*(R+1))+((R+1)*(z-1))+r] / ( 0.1*grid) ;
  } else if (point_type[((R+1)*z)+r] < INSIDE &&
            point_type[((R+1)*(z+1))+r] == CONTACT_EDGE) {
    E_z =  (v[(new_gpu_relax*(L+1)*(R+1))+((R+1)*z)+r] - v[(new_gpu_relax*(L+1)*(R+1))+((R+1)*(z+1))+r]) * dz[(0*(L+1)*(R+1))+((R+1)*(z+1))+r] / ( 0.1*grid) ;
  } else if (z == 1) {
    E_z = (v[(new_gpu_relax*(L+1)*(R+1))+((R+1)*z)+r] - v[(new_gpu_relax*(L+1)*(R+1))+((R+1)*(z+1))+r])/(0.1*grid);
  } else if (z == L-1) {
    E_z = (v[(new_gpu_relax*(L+1)*(R+1))+((R+1)*(z-1))+r] - v[(new_gpu_relax*(L+1)*(R+1))+((R+1)*z)+r])/(0.1*grid);
  } else {
    E_z = (v[(new_gpu_relax*(L+1)*(R+1))+((R+1)*(z-1))+r] - v[(new_gpu_relax*(L+1)*(R+1))+((R+1)*(z+1))+r])/(0.2*grid);
  }
  ve_z = ve_r = 0;
  E = fabs(E_z);
  int b;
  if (E > 1.0) {
    for (b=0; E > drift_E[b+1]; b++);
    ve_z = fq * (drift_offset[b] + drift_slope[b]*(E - drift_E[b]));
  }
  E = fabs(E_r);
  if (E > 1.0) {
    for (b=0; E > drift_E[b+1]; b++);
    ve_r = fq * (drift_offset[b] + drift_slope[b]*(E - drift_E[b]));
  }
  /* reduce drift speed at passivated surfaces by a factor of surface_drift_vel_factor */
  if (1 &&
      ((r == idid && z < idd) ||
      (r < idid  && z == 1 ) ||
      (r >= idid && r <= idod && z == idd))) {
    ve_r *= surface_drift_vel_factor;
    ve_z *= surface_drift_vel_factor * grid/0.002;  // assume 2-micron-thick roughness/passivation in z
  }


  //-----------------------------------------------------------

  /* do drift to neighboring pixels */
  // enum point_types{PC, HVC, INSIDE, PASSIVE, PINCHOFF, DITCH, DITCH_EDGE, CONTACT_EDGE};
  if (E_r > 0) {
    dre = -tstep*ve_r;
  } else {
    dre =  tstep*ve_r;
  }
  if (E_z > 0) {
    dze = -tstep*ve_z;
  } else {
    dze =  tstep*ve_z;
  }

  if (dre == 0.0) {
    i_array[((R+1)*z)+r]= r;
    fr_array[((R+1)*z)+r] = 1.0;
  } else {
    i_array[((R+1)*z)+r]= (double) r + dre;
    fr_array[((R+1)*z)+r] = ceil(dre) - dre;
  }
  if (i_array[((R+1)*z)+r]<1) {
    i_array[((R+1)*z)+r]= 1;
    fr_array[((R+1)*z)+r] = 1.0;
  }
  if (i_array[((R+1)*z)+r]>R-1) {
    i_array[((R+1)*z)+r]= R-1;
    fr_array[((R+1)*z)+r] = 0.0;
  }
  if (dre > 0 && z < idd && r <= idid && i_array[((R+1)*z)+r]>= idid) { // ditch ID
    i_array[((R+1)*z)+r]= idid;
    fr_array[((R+1)*z)+r] = 1.0;
  }
  if (dre < 0 && z < idd && r >= idod && i_array[((R+1)*z)+r]<= idod) { // ditch OD
    i_array[((R+1)*z)+r]= idod;
    fr_array[((R+1)*z)+r] = 0.0;
  }

  if (dze == 0.0) {
    k_array[((R+1)*z)+r]= z;
    fz_array[((R+1)*z)+r] = 1.0;
  } else {
    k_array[((R+1)*z)+r]= (double) z + dze;
    fz_array[((R+1)*z)+r] = ceil(dze) - dze;
  }
  if (k_array[((R+1)*z)+r]<1) {
    k_array[((R+1)*z)+r]= 1;
    fz_array[((R+1)*z)+r] = 1.0;
  }
  if (k_array[((R+1)*z)+r]>L-1) {
    k_array[((R+1)*z)+r]= L-1;
    fz_array[((R+1)*z)+r] = 0.0;
  }
  if (dze < 0 && r > idid && r < idod && k_array[((R+1)*z)+r]< idd) { // ditch depth
    k_array[((R+1)*z)+r]  = idd;
    fr_array[((R+1)*z)+r]  = 1.0;
  }

  

  if (0 && r == 100 && z == 10)
    printf("r z: %d %d; E_r i_array[((R+1)*z)+r]dre: %f %d %f; fr_array[((R+1)*z)+r] = %f\n"
          "r z: %d %d; E_z k_array[((R+1)*z)+r]dze: %f %d %f; fz_array[((R+1)*z)+r] = %f\n",
          r, z, E_r, i_array[((R+1)*z)+r], dre, fr_array[((R+1)*z)+r], r, z, E_z, k_array[((R+1)*z)+r], dze, fz_array[((R+1)*z)+r]);

}

/* 
  Drift the charges stores the new locations to rho[2].
  Atomic operations are need to perform the update without interference from any other threads.
 */

__global__ void gpu_sr_update(int L, int R, double *rho, double *fr_array, double *fz_array, int *i_array, int *k_array, int max_threads){

  int r = blockIdx.x%R;
  int z = (floorf(blockIdx.x/R) * max_threads) + threadIdx.x;
  
  if(r==0 || z==0 || r>=R || z>=(L-2)){
    return;
  }
    
if (rho[(1*(L+1)*(R+1))+((R+1)*z)+r] < 1.0e-14) {
  return;
}

  if (i_array[((R+1)*z)+r]>=1 && i_array[((R+1)*z)+r]<R && k_array[((R+1)*z)+r]>=1 && k_array[((R+1)*z)+r]<L) {
    if (i_array[((R+1)*z)+r] > 1 && r > 1) {
      atomicAdd(&rho[(2*(L+1)*(R+1))+((R+1)*k_array[((R+1)*z)+r])+i_array[((R+1)*z)+r]], rho[(1*(L+1)*(R+1))+((R+1)*z)+r] * fr_array[((R+1)*z)+r]      *fz_array[((R+1)*z)+r]       * (double) (r-1) / (double) (i_array[((R+1)*z)+r]-1));
      atomicAdd(&rho[(2*(L+1)*(R+1))+((R+1)*k_array[((R+1)*z)+r])+i_array[((R+1)*z)+r]+1], rho[(1*(L+1)*(R+1))+((R+1)*z)+r] * (1.0-fr_array[((R+1)*z)+r])*fz_array[((R+1)*z)+r]       * (double) (r-1) / (double) (i_array[((R+1)*z)+r]));
      atomicAdd(&rho[(2*(L+1)*(R+1))+((R+1)*(k_array[((R+1)*z)+r]+1))+i_array[((R+1)*z)+r]], rho[(1*(L+1)*(R+1))+((R+1)*z)+r] * fr_array[((R+1)*z)+r]      *(1.0-fz_array[((R+1)*z)+r]) * (double) (r-1) / (double) (i_array[((R+1)*z)+r]-1));
      atomicAdd(&rho[(2*(L+1)*(R+1))+((R+1)*(k_array[((R+1)*z)+r]+1))+i_array[((R+1)*z)+r]+1], rho[(1*(L+1)*(R+1))+((R+1)*z)+r] * (1.0-fr_array[((R+1)*z)+r])*(1.0-fz_array[((R+1)*z)+r]) * (double) (r-1) / (double) (i_array[((R+1)*z)+r]));
    } 
    else if (i_array[((R+1)*z)+r] > 1) {
      atomicAdd(&rho[(2*(L+1)*(R+1))+((R+1)*k_array[((R+1)*z)+r])+i_array[((R+1)*z)+r]], rho[(1*(L+1)*(R+1))+((R+1)*z)+r] * fr_array[((R+1)*z)+r]      *fz_array[((R+1)*z)+r]       / (double) (8*i_array[((R+1)*z)+r]-8));
      atomicAdd(&rho[(2*(L+1)*(R+1))+((R+1)*k_array[((R+1)*z)+r])+i_array[((R+1)*z)+r]+1], rho[(1*(L+1)*(R+1))+((R+1)*z)+r] * (1.0-fr_array[((R+1)*z)+r])*fz_array[((R+1)*z)+r]       / (double) (8*i_array[((R+1)*z)+r]));
      atomicAdd(&rho[(2*(L+1)*(R+1))+((R+1)*(k_array[((R+1)*z)+r]+1))+i_array[((R+1)*z)+r]], rho[(1*(L+1)*(R+1))+((R+1)*z)+r] * fr_array[((R+1)*z)+r]      *(1.0-fz_array[((R+1)*z)+r]) / (double) (8*i_array[((R+1)*z)+r]-8));
      atomicAdd(&rho[(2*(L+1)*(R+1))+((R+1)*(k_array[((R+1)*z)+r]+1))+i_array[((R+1)*z)+r]+1], rho[(1*(L+1)*(R+1))+((R+1)*z)+r] * (1.0-fr_array[((R+1)*z)+r])*(1.0-fz_array[((R+1)*z)+r]) / (double) (8*i_array[((R+1)*z)+r]));
    } 
    else if (r > 1) {
      atomicAdd(&rho[(2*(L+1)*(R+1))+((R+1)*k_array[((R+1)*z)+r])+i_array[((R+1)*z)+r]], rho[(1*(L+1)*(R+1))+((R+1)*z)+r] * fr_array[((R+1)*z)+r]      *fz_array[((R+1)*z)+r]       * (double) (8*(R+1)-8));
      atomicAdd(&rho[(2*(L+1)*(R+1))+((R+1)*k_array[((R+1)*z)+r])+i_array[((R+1)*z)+r]+1], rho[(1*(L+1)*(R+1))+((R+1)*z)+r] * (1.0-fr_array[((R+1)*z)+r])*fz_array[((R+1)*z)+r]       * (double) (r-1));
      atomicAdd(&rho[(2*(L+1)*(R+1))+((R+1)*(k_array[((R+1)*z)+r]+1))+i_array[((R+1)*z)+r]], rho[(1*(L+1)*(R+1))+((R+1)*z)+r] * fr_array[((R+1)*z)+r]      *(1.0-fz_array[((R+1)*z)+r]) * (double) (8*(R+1)-8));
      atomicAdd(&rho[(2*(L+1)*(R+1))+((R+1)*(k_array[((R+1)*z)+r]+1))+i_array[((R+1)*z)+r]+1], rho[(1*(L+1)*(R+1))+((R+1)*z)+r] * (1.0-fr_array[((R+1)*z)+r])*(1.0-fz_array[((R+1)*z)+r]) * (double) (r-1));
    } 
    else {   
      atomicAdd(&rho[(2*(L+1)*(R+1))+((R+1)*k_array[((R+1)*z)+r])+i_array[((R+1)*z)+r]], rho[(1*(L+1)*(R+1))+((R+1)*z)+r] * fr_array[((R+1)*z)+r]      *fz_array[((R+1)*z)+r]);
      atomicAdd(&rho[(2*(L+1)*(R+1))+((R+1)*k_array[((R+1)*z)+r])+i_array[((R+1)*z)+r]+1], rho[(1*(L+1)*(R+1))+((R+1)*z)+r] * (1.0-fr_array[((R+1)*z)+r])*fz_array[((R+1)*z)+r]       / 8.0); // vol_0 / vol_1 = 1/8
      atomicAdd(&rho[(2*(L+1)*(R+1))+((R+1)*(k_array[((R+1)*z)+r]+1))+i_array[((R+1)*z)+r]], rho[(1*(L+1)*(R+1))+((R+1)*z)+r] * fr_array[((R+1)*z)+r]      *(1.0-fz_array[((R+1)*z)+r]));
      atomicAdd(&rho[(2*(L+1)*(R+1))+((R+1)*(k_array[((R+1)*z)+r]+1))+i_array[((R+1)*z)+r]+1], rho[(1*(L+1)*(R+1))+((R+1)*z)+r] * (1.0-fr_array[((R+1)*z)+r])*(1.0-fz_array[((R+1)*z)+r]) / 8.0);
    }
  }
}

__global__ void hvc_modicication(int L, int R, double *rho, int max_threads, char *point_type){

    int r = blockIdx.x%R;
    int z = (floorf(blockIdx.x/R) * max_threads) + threadIdx.x;
    
    if( r>=R || z>=L ){
      return;
    }
    if(point_type[((R+1)*z)+r]<=HVC){
        rho[(2*(L+1)*(R+1))+((R+1)*z)+r] = 0;
    }
}

__global__ void set_rho_zero(int L, int R, double *rho_e, double *rho_h, int max_threads){

  int r = blockIdx.x%R;
  int z = (floorf(blockIdx.x/R) * max_threads) + threadIdx.x;
  
if(r==0 || z==0 || r>=R || z>=L){
    return;
  }
  // rho_e[0][z][r] = rho_e[2][z][r];
  // rho_h[0][z][r] = rho_h[2][z][r];
  rho_e[(0*(L+1)*(R+1))+((R+1)*z)+r] = rho_e[(2*(L+1)*(R+1))+((R+1)*z)+r];
  rho_h[(0*(L+1)*(R+1))+((R+1)*z)+r] = rho_h[(2*(L+1)*(R+1))+((R+1)*z)+r];
}

__global__ void update_impurities(int L, int R, double *impurity_gpu, double *rho_e, double *rho_h, int max_threads, double e_over_E, float grid){

  int r = blockIdx.x%R;
  int z = (floorf(blockIdx.x/R) * max_threads) + threadIdx.x;
  
if(r==0 || z==0 || r>=R || z>=L){
    return;
  }
  //  setup.impurity[z][r] = rho_e[3][z][r] +
  // (rho_h[0][z][r] - rho_e[0][z][r]) * e_over_E * grid*grid/2.0;
  impurity_gpu[((R+1)*z)+r] = rho_e[(3*(L+1)*(R+1))+((R+1)*z)+r] + (rho_h[(0*(L+1)*(R+1))+((R+1)*z)+r] - rho_e[(0*(L+1)*(R+1))+((R+1)*z)+r]) * e_over_E * grid*grid/2.0;
}

extern "C" int gpu_drift(MJD_Siggen_Setup *setup, int L, int R, float grid, double ***rho, int q, GPU_data *gpu_setup){



  tstep = setup-> step_time_calc;
  delta = 0.07*tstep;
  delta_r = 0.07*tstep;
  wrap_around_radius = setup->wrap_around_radius;
  ditch_thickness =  setup-> ditch_thickness;
  ditch_depth = setup-> ditch_depth;
  surface_drift_vel_factor = setup->surface_drift_vel_factor;
  fq = -q;

  grid = setup->xtal_grid;
    /* NOTE that impurity and field arrays in setup start at (i,j)=(1,1) for (r,z)=(0,0) */
    idid = lrint((wrap_around_radius - ditch_thickness)/grid) + 1; // ditch ID
    idod = lrint(wrap_around_radius/grid) + 1; // ditch OD
    idd =  lrint(ditch_depth/grid) + 1;        // ditch depth


  
  f_drift = 1.2e6; // * setup.xtal_temp/REF_TEMP;
  f_drift *= tstep / 4000.0;
  /* above is my own approximate parameterization of measurements of Jacoboni et al.
     1.2e6 * v_over_E   ~   D in cm2/s
     v_over_E = drift velocity / electric field   ~  mu
     note that Einstein's equation is D = mu*kT/e
     kT/e ~ 0.007/V ~ 0.07 mm/Vcm, => close enough to 0.12, okay
     For 20-micron bins and 1ns steps, DELTA = D / 4000
     For fixed D, DELTA goes as time_step_size/bin_size_squared
  */
  f_drift *= 0.02/grid * 0.02/grid; // correct for grid size


  double *drift_offset_gpu, *drift_slope_gpu;
  double *rho_gpu;
  if (q < 0) { // electrons
    drift_offset_gpu = gpu_setup->drift_offset_e_gpu;
    drift_slope_gpu  = gpu_setup->drift_slope_e_gpu;
    rho_gpu = gpu_setup->rho_e_gpu;
  } 
  else {   // holes
    drift_offset_gpu = gpu_setup->drift_offset_h_gpu;
    drift_slope_gpu  = gpu_setup->drift_slope_h_gpu;
    rho_gpu = gpu_setup->rho_h_gpu;
  }

  int num_threads = 1024;
  int num_blocks = R * (ceil(L/num_threads)+1);

  // if(num_blocks<65536 && num_threads<=1024){
    reset_rho<<<num_blocks,num_threads>>>(L, R, rho_gpu, num_threads);
    cudaDeviceSynchronize();
    gpu_diffusion<<<num_blocks,num_threads>>>(L, R, grid, rho_gpu, q, gpu_setup->v_gpu, gpu_setup->point_type_gpu, gpu_setup->dr_gpu, gpu_setup->dz_gpu, gpu_setup->s1_gpu, gpu_setup->s2_gpu, drift_offset_gpu, drift_slope_gpu, num_threads, gpu_setup->deltaez_array, gpu_setup->deltaer_array);
    cudaDeviceSynchronize();
    diff_update<<<num_blocks,num_threads>>>(L, R, rho_gpu, gpu_setup->point_type_gpu, gpu_setup->s1_gpu, gpu_setup->s2_gpu, gpu_setup->deltaez_array, gpu_setup->deltaer_array, num_threads);
    gpu_self_repulsion<<<num_blocks,num_threads>>>(L, R, grid, rho_gpu, q, gpu_setup->v_gpu, gpu_setup->point_type_gpu, gpu_setup->dr_gpu, gpu_setup->dz_gpu, gpu_setup->s1_gpu, gpu_setup->s2_gpu, drift_offset_gpu, drift_slope_gpu, num_threads, gpu_setup->fr_array, gpu_setup->fz_array, gpu_setup->i_array, gpu_setup->k_array);
    cudaDeviceSynchronize();
    gpu_sr_update<<<num_blocks,num_threads>>>(L, R, rho_gpu, gpu_setup->fr_array, gpu_setup->fz_array, gpu_setup->i_array, gpu_setup->k_array, num_threads);
    cudaDeviceSynchronize();
    hvc_modicication<<<num_blocks,num_threads>>>(L, R, rho_gpu, num_threads, gpu_setup->point_type_gpu);
  // }
  // else{
  //   printf("----------------Pick a smaller block please----------------\n");
  //   return 0;
  // }
  return 0;
}

//Cuda kernals can only be called from files compiled with nvcc compile which are normally with .cu extension

extern "C" void set_rho_zero_gpu(GPU_data *gpu_setup, int L, int R, int num_blocks, int num_threads){
  set_rho_zero<<<num_blocks,num_threads>>>(L, R, gpu_setup->rho_e_gpu, gpu_setup->rho_h_gpu, num_threads);
}

extern "C" void update_impurities_gpu(GPU_data *gpu_setup, int L, int R, int num_blocks, int num_threads, double e_over_E, float grid){
  update_impurities<<<num_blocks,num_threads>>>(L, R, gpu_setup->impurity_gpu, gpu_setup->rho_e_gpu, gpu_setup->rho_h_gpu, num_threads, e_over_E, grid);
}