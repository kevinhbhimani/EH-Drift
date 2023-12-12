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


extern "C" int gpu_drift(MJD_Siggen_Setup *setup, int L, int R, float grid, double ***rho, int q, GPU_data *gpu_setup, int n_iter);
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
__managed__ double passivated_surface_thickness_gpu;
__managed__ double grid;
__managed__ int L;
__managed__ int R;
__managed__ int q;
__managed__ int max_threads;

__global__ void reset_rho(double *rho, double *surface_rho, int max_threads){

    int r = blockIdx.x%R;
    int z = (floorf(blockIdx.x/R) * max_threads) + threadIdx.x;
    
    if(r>=R || z>=L){
      return;
    }
    rho[(1*(L+1)*(R+1))+((R+1)*z)+r] = rho[(2*(L+1)*(R+1))+((R+1)*z)+r] = 0.0;
    if(z==1){
      surface_rho[(R+1)+r] = surface_rho[2*(R+1)+r] = 0.0;
    }
}
/* 
  Uses diffusion eqaution to calculate the diffusion coefficients and stores them in appropriate arrays
 */
 __global__ void gpu_diffusion(double *rho, double *v, char *point_type, double *dr, double *dz, double *s1, double *s2, double *drift_offset, double *drift_slope, double *deltaez_array, double *deltaer_array){

    int r = blockIdx.x%R;
    int z = (floorf(blockIdx.x/R) * max_threads) + threadIdx.x;
    
    if(r==0 || z==0 || r>=R || z>=(L-2)){
      return;
    }
    int i, new_gpu_relax = 0;
    float E, E_r, E_z;
    double ve_z, ve_r, deltaez, deltaer;
  

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
  
    if (0 && z == 1) 
    printf("r,z = %d, %d E_r,z = %f, %f  deltaer,z = %f, %f  s1,s2 = %f, %f\n",
                      r, z, E_r, E_z, deltaer, deltaez, s1[r], s2[r]);
  
    deltaez_array[((R+1)*z)+r] = deltaez;
    deltaer_array[((R+1)*z)+r] = deltaer;
  
  }

/*
  Performs diffusion to values in rho[0] and stores them to rho[1].
  Atomic operations are need to perform the update without interference from any other threads.
  When z=1 we also handle from surface (z=0)

      if (r < R-1 && setup->point_type[z][r+1] != DITCH) {
        //if (setup->point_type[z][r+1] > HVC)
        rho[1][z][r+1] += rho[0][z][r]*deltaer * setup->s1[r] * (double) (r-1) / (double) (r);
        rho[1][z][r]   -= rho[0][z][r]*deltaer * setup->s1[r];
      }
      if (z > 1 && setup->point_type[z-1][r] != DITCH) {
        //if (setup->point_type[z-1][r] > HVC)
        rho[1][z-1][r] += rho[0][z][r]*deltaez;
        rho[1][z][r]   -= rho[0][z][r]*deltaez;
      }
      if (z < L-1 && setup->point_type[z+1][r] != DITCH) {
        //if (setup->point_type[z+1][r] > HVC)
        rho[1][z+1][r] += rho[0][z][r]*deltaez;
        rho[1][z][r]   -= rho[0][z][r]*deltaez;
      }
      if (r > 2 && setup->point_type[z][r-1] != DITCH) {
        //if (setup->point_type[z][r-1] > HVC)
        rho[1][z][r-1] += rho[0][z][r]*deltaer * setup->s2[r] * (double) (r-1) / (double) (r-2);
        rho[1][z][r]   -= rho[0][z][r]*deltaer * setup->s2[r];
      }
*/
__global__ void diff_update(double *rho, char *point_type, double *s1, double *s2, double *deltaez_array, double *deltaer_array, double *surface_rho){

  int r = blockIdx.x%R;
  int z = (floorf(blockIdx.x/R) * max_threads) + threadIdx.x;
  
  if(r==0 || z==0 || r>=R || z>=(L-2)){
    return;
  }

  double deltaez = deltaez_array[((R+1)*z)+r];
  double deltaer = deltaer_array[((R+1)*z)+r];

  atomicAdd(&rho[(1*(L+1)*(R+1))+((R+1)*z)+r], rho[(0*(L+1)*(R+1))+((R+1)*z)+r]);

  if (z==1){
    atomicAdd(&surface_rho[(R+1)+r], surface_rho[r]);
  }

  if (r < R-1 && point_type[((R+1)*z)+r+1] != DITCH) {
    atomicAdd(&rho[(1*(L+1)*(R+1))+((R+1)*z)+r+1], rho[(0*(L+1)*(R+1))+((R+1)*z)+r]*deltaer * s1[r] * (double) (r-1) / (double) (r));
    atomicAdd(&rho[(1*(L+1)*(R+1))+((R+1)*z)+r],-rho[(0*(L+1)*(R+1))+((R+1)*z)+r]*deltaer * s1[r]);
    if(z==1){
        atomicAdd(&surface_rho[(R+1)+r+1], surface_rho[r] * deltaer * surface_drift_vel_factor * s1[r] * (double) (r-1) / (double) (r));
        atomicAdd(&surface_rho[(R+1)+r],-surface_rho[r] * deltaer * surface_drift_vel_factor * s1[r]);
    }
  }
  if (z > 1 && point_type[((R+1)*(z-1))+r] != DITCH) {
      atomicAdd(&rho[(1*(L+1)*(R+1))+((R+1)*(z-1))+r], rho[(0*(L+1)*(R+1))+((R+1)*z)+r]*deltaez);
      atomicAdd(&rho[(1*(L+1)*(R+1))+((R+1)*z)+r], -rho[(0*(L+1)*(R+1))+((R+1)*z)+r]*deltaez);

  }
  // if z==1 we want the charges to diffuse to surface
  if(z==1 && point_type[((R+1)*(z-1))+r] != DITCH){
    atomicAdd(&surface_rho[(R+1)+r], rho[(0*(L+1)*(R+1))+((R+1)*z)+r]*deltaez* surface_drift_vel_factor);
    atomicAdd(&rho[(1*(L+1)*(R+1))+((R+1)*z)+r], -rho[(0*(L+1)*(R+1))+((R+1)*z)+r]*deltaez);
  }

  if (z < L-1 && point_type[((R+1)*(z+1))+r] != DITCH) {
    atomicAdd(&rho[(1*(L+1)*(R+1))+((R+1)*(z+1))+r], rho[(0*(L+1)*(R+1))+((R+1)*z)+r]*deltaez);
    atomicAdd(&rho[(1*(L+1)*(R+1))+((R+1)*z)+r], -rho[(0*(L+1)*(R+1))+((R+1)*z)+r]*deltaez);
    if(z==1){
      atomicAdd(&rho[(1*(L+1)*(R+1))+((R+1)*z)+r],surface_rho[r]*deltaez * surface_drift_vel_factor);
      atomicAdd(&surface_rho[(R+1)+r], -surface_rho[r]*deltaez * surface_drift_vel_factor);
    }
  }
  if (r > 2 && point_type[((R+1)*z)+r-1] != DITCH) {
    atomicAdd(&rho[(1*(L+1)*(R+1))+((R+1)*z)+(r-1)], rho[(0*(L+1)*(R+1))+((R+1)*z)+r]*deltaer * s2[r] * (double) (r-1) / (double) (r-2));
    atomicAdd(&rho[(1*(L+1)*(R+1))+((R+1)*z)+r], -rho[(0*(L+1)*(R+1))+((R+1)*z)+r]*deltaer * s2[r]);
    if(z==1){
      atomicAdd(&surface_rho[(R+1)+r-1], surface_rho[r]*deltaer * surface_drift_vel_factor * s2[r] * (double) (r-1) / (double) (r-2));
      atomicAdd(&surface_rho[(R+1)+r], -surface_rho[r]*deltaer * surface_drift_vel_factor * s2[r]);
    }
  }
}

/* 
  Sees where the charge would drift in the field and stores the new location
 */
__global__ void gpu_self_repulsion(double *rho, double *v, char *point_type,  double *dr, double *dz, 
  double *s1, double *s2, double *drift_offset, double *drift_slope, double *fr_array, double *fz_array, 
  int *i_array, int *k_array, double *velocity_drift_r, double *velocity_drift_z, double *field_r, double *field_z){

    int new_gpu_relax = 0;
    float E, E_r, E_z;
    double dre, dze;
    double ve_z, ve_r;

    int r = blockIdx.x%R;
    int z = (floorf(blockIdx.x/R) * max_threads) + threadIdx.x;
    
    if(r==0 || z==0 || r>=R || z>=(L-2)){
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

  if (z==1){ //stores the field and velocities for surface to perfrom drift separately
    velocity_drift_r[r] = ve_r;
    velocity_drift_z[r] = ve_z;
  }
  field_r[((R+1)*z)+r]=E_r;
  field_z[((R+1)*z)+r]=E_z;


  


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


  //handles charges drifting to the surface
  double z_loc = (double) z + dze;
  if (z_loc<=0.0){
    k_array[((R+1)*z)+r]= 0;
    fz_array[((R+1)*z)+r] = 1.0;
  }

  if (z_loc>0.0 && z_loc<=passivated_surface_thickness_gpu && dze!=0.0){ //CHECK
    k_array[((R+1)*z)+r]=0;
    double top_edge = z_loc+ 0.5;
    if(top_edge<=passivated_surface_thickness_gpu){
      fz_array[((R+1)*z)+r] = 1.0;
    }
    else{
      fz_array[((R+1)*z)+r] = (top_edge-passivated_surface_thickness_gpu)/top_edge;
    }
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


  For z==1 first and fourth condition are meet
  for z==2 first and fourth condition are meet

      if (i>=1 && i<R && k>=1 && k<L) {
        if (i > 1 && r > 1) {
          rho[2][k  ][i  ] += rho[1][z][r] * fr      *fz       * (double) (r-1) / (double) (i-1);
          rho[2][k  ][i+1] += rho[1][z][r] * (1.0-fr)*fz       * (double) (r-1) / (double) (i);
          rho[2][k+1][i  ] += rho[1][z][r] * fr      *(1.0-fz) * (double) (r-1) / (double) (i-1);
          rho[2][k+1][i+1] += rho[1][z][r] * (1.0-fr)*(1.0-fz) * (double) (r-1) / (double) (i);
        } else if (i > 1) {  // r == 0
          rho[2][k  ][i  ] += rho[1][z][r] * fr      *fz       / (double) (8*i-8);
          rho[2][k  ][i+1] += rho[1][z][r] * (1.0-fr)*fz       / (double) (8*i);
          rho[2][k+1][i  ] += rho[1][z][r] * fr      *(1.0-fz) / (double) (8*i-8);
          rho[2][k+1][i+1] += rho[1][z][r] * (1.0-fr)*(1.0-fz) / (double) (8*i);
        } else if (r > 1) {  // i == 0
          rho[2][k  ][i  ] += rho[1][z][r] * fr      *fz       * (double) (8*r-8);
          rho[2][k  ][i+1] += rho[1][z][r] * (1.0-fr)*fz       * (double) (r-1);
          rho[2][k+1][i  ] += rho[1][z][r] * fr      *(1.0-fz) * (double) (8*r-8);
          rho[2][k+1][i+1] += rho[1][z][r] * (1.0-fr)*(1.0-fz) * (double) (r-1);
        } else {             // r == i == 0
          rho[2][k  ][i  ] += rho[1][z][r] * fr      *fz;
          rho[2][k  ][i+1] += rho[1][z][r] * (1.0-fr)*fz       / 8.0; // vol_0 / vol_1 = 1/8
          rho[2][k+1][i  ] += rho[1][z][r] * fr      *(1.0-fz);
          rho[2][k+1][i+1] += rho[1][z][r] * (1.0-fr)*(1.0-fz) / 8.0;
        }
 */

__global__ void gpu_sr_update(double *rho, double *fr_array, double *fz_array, int *i_array, int *k_array, double *surface_rho){

  int r = blockIdx.x%R;
  int z = (floorf(blockIdx.x/R) * max_threads) + threadIdx.x;
  
  if(r==0 || z==0 || r>=R || z>=(L-2)){
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

  else if(k_array[((R+1)*z)+r]==0){
    if (i_array[((R+1)*z)+r] > 1 && r > 1) {
      atomicAdd(&surface_rho[2*(R+1)+i_array[((R+1)*z)+r]],  rho[(1*(L+1)*(R+1))+((R+1)*z)+r] * fr_array[((R+1)*z)+r]      *fz_array[((R+1)*z)+r]       * (double) (r-1) / (double) (i_array[((R+1)*z)+r]-1));
      atomicAdd(&surface_rho[2*(R+1)+i_array[((R+1)*z)+r]+1],  rho[(1*(L+1)*(R+1))+((R+1)*z)+r] * (1.0-fr_array[((R+1)*z)+r])*fz_array[((R+1)*z)+r]       * (double) (r-1) / (double) (i_array[((R+1)*z)+r]));
      atomicAdd(&rho[(2*(L+1)*(R+1))+((R+1)*(k_array[((R+1)*z)+r]+1))+i_array[((R+1)*z)+r]], rho[(1*(L+1)*(R+1))+((R+1)*z)+r] * fr_array[((R+1)*z)+r]      *(1.0-fz_array[((R+1)*z)+r]) * (double) (r-1) / (double) (i_array[((R+1)*z)+r]-1));
      atomicAdd(&rho[(2*(L+1)*(R+1))+((R+1)*(k_array[((R+1)*z)+r]+1))+i_array[((R+1)*z)+r]+1], rho[(1*(L+1)*(R+1))+((R+1)*z)+r] * (1.0-fr_array[((R+1)*z)+r])*(1.0-fz_array[((R+1)*z)+r]) * (double) (r-1) / (double) (i_array[((R+1)*z)+r]));
    
    } 
    else if (i_array[((R+1)*z)+r] > 1) {
      atomicAdd(&surface_rho[2*(R+1)+i_array[((R+1)*z)+r]],  rho[(1*(L+1)*(R+1))+((R+1)*z)+r] * fr_array[((R+1)*z)+r]      *fz_array[((R+1)*z)+r]       / (double) (8*i_array[((R+1)*z)+r]-8));
      atomicAdd(&surface_rho[2*(R+1)+i_array[((R+1)*z)+r]+1],  rho[(1*(L+1)*(R+1))+((R+1)*z)+r] * (1.0-fr_array[((R+1)*z)+r])*fz_array[((R+1)*z)+r]       / (double) (8*i_array[((R+1)*z)+r]));
      atomicAdd(&rho[(2*(L+1)*(R+1))+((R+1)*(k_array[((R+1)*z)+r]+1))+i_array[((R+1)*z)+r]], rho[(1*(L+1)*(R+1))+((R+1)*z)+r] * fr_array[((R+1)*z)+r]      *(1.0-fz_array[((R+1)*z)+r]) / (double) (8*i_array[((R+1)*z)+r]-8));
      atomicAdd(&rho[(2*(L+1)*(R+1))+((R+1)*(k_array[((R+1)*z)+r]+1))+i_array[((R+1)*z)+r]+1], rho[(1*(L+1)*(R+1))+((R+1)*z)+r] * (1.0-fr_array[((R+1)*z)+r])*(1.0-fz_array[((R+1)*z)+r]) / (double) (8*i_array[((R+1)*z)+r]));
    } 
    else if (r > 1) {
      atomicAdd(&surface_rho[2*(R+1)+i_array[((R+1)*z)+r]],  rho[(1*(L+1)*(R+1))+((R+1)*z)+r] * fr_array[((R+1)*z)+r]      *fz_array[((R+1)*z)+r]       * (double) (8*(R+1)-8));
      atomicAdd(&surface_rho[2*(R+1)+i_array[((R+1)*z)+r]+1],  rho[(1*(L+1)*(R+1))+((R+1)*z)+r] * (1.0-fr_array[((R+1)*z)+r])*fz_array[((R+1)*z)+r]       * (double) (r-1));
      atomicAdd(&rho[(2*(L+1)*(R+1))+((R+1)*(k_array[((R+1)*z)+r]+1))+i_array[((R+1)*z)+r]], rho[(1*(L+1)*(R+1))+((R+1)*z)+r] * fr_array[((R+1)*z)+r]      *(1.0-fz_array[((R+1)*z)+r]) * (double) (8*(R+1)-8));
      atomicAdd(&rho[(2*(L+1)*(R+1))+((R+1)*(k_array[((R+1)*z)+r]+1))+i_array[((R+1)*z)+r]+1], rho[(1*(L+1)*(R+1))+((R+1)*z)+r] * (1.0-fr_array[((R+1)*z)+r])*(1.0-fz_array[((R+1)*z)+r]) * (double) (r-1));
    
    } 
    else {
      atomicAdd(&surface_rho[2*(R+1)+i_array[((R+1)*z)+r]],  rho[(1*(L+1)*(R+1))+((R+1)*z)+r] * fr_array[((R+1)*z)+r]      *fz_array[((R+1)*z)+r]);
      atomicAdd(&surface_rho[2*(R+1)+i_array[((R+1)*z)+r]+1],  rho[(1*(L+1)*(R+1))+((R+1)*z)+r] * (1.0-fr_array[((R+1)*z)+r])*fz_array[((R+1)*z)+r]       / 8.0); // vol_0 / vol_1 = 1/8
      atomicAdd(&rho[(2*(L+1)*(R+1))+((R+1)*(k_array[((R+1)*z)+r]+1))+i_array[((R+1)*z)+r]], rho[(1*(L+1)*(R+1))+((R+1)*z)+r] * fr_array[((R+1)*z)+r]      *(1.0-fz_array[((R+1)*z)+r]));
      atomicAdd(&rho[(2*(L+1)*(R+1))+((R+1)*(k_array[((R+1)*z)+r]+1))+i_array[((R+1)*z)+r]+1], rho[(1*(L+1)*(R+1))+((R+1)*z)+r] * (1.0-fr_array[((R+1)*z)+r])*(1.0-fz_array[((R+1)*z)+r]) / 8.0);
    }
   }
}

__global__ void surface_drift_calc(double *rho, double *velocity_drift_r, double *velocity_drift_z, double *field_r, double *field_z, double *fr_array, double *fz_array, int *i_array, int *k_array, double *surface_rho){

  int r = blockIdx.x%R;
  int z = (floorf(blockIdx.x/R) * max_threads) + threadIdx.x;
  double fr, fz, dre, dze;
  if(r==0 || z==0 || r>=R || z>1){
    return;
  }

  double E_r = field_r[((R+1)*z)+r];
  double E_z = field_z[((R+1)*z)+r];
  double ve_r = velocity_drift_r[r]*surface_drift_vel_factor;
  double ve_z = velocity_drift_z[r]*surface_drift_vel_factor;
  int i_drift, k_drift;

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
    i_drift = r;
    fr = 1.0;
  } else {
    i_drift = (double) r + dre;
    fr = ceil(dre) - dre;
  }
  if (i_drift<1) {
    i_drift = 1;
    fr = 1.0;
  }
  if (i_drift>R-1) {
    i_drift = R-1;
    fr = 0.0;
  }
  if (dre > 0 && z < idd && r <= idid && i_drift >= idid) { // ditch ID
    i_drift = idid;
    fr = 1.0;
  }
  if (dre < 0 && z < idd && r >= idod && i_drift <= idod) { // ditch OD
    i_drift = idod;
    fr = 0.0;
  }

  if (dze == 0.0) {
    k_drift = 0;
    fz = 1.0;
  } else {
    k_drift = dze;
    fz = ceil(dze) - dze;
  }

    //handles charges drifting to the surface
  double z_loc = dze;
  if (z_loc<=0.0){
    k_drift= 0;
    fz = 1.0;
  }

  if (z_loc>0.0 && z_loc<=passivated_surface_thickness_gpu && dze!=0.0){ //CHECK
    k_drift=0;
    double top_edge = z_loc+ 0.5;
    if(top_edge<=passivated_surface_thickness_gpu){
      fz = 1.0;
    }
    else{
      fz= (top_edge-passivated_surface_thickness_gpu)/top_edge;
    }
  }

  if (k_drift>L-1) {
    k_drift = L-1;
    fz = 0.0;
  }
  if (dze < 0 && r > idid && r < idod && k_drift < idd) { // ditch depth
    k_drift   = idd;
    fr  = 1.0;
  }

  fr_array[((R+1)*z)+r]=fr;
  fz_array[((R+1)*z)+r]=fz;
  i_array[((R+1)*z)+r]=i_drift;
  k_array[((R+1)*z)+r]=k_drift;
}

__global__ void surface_drift(double *rho, double *fr_array, double *fz_array, int *i_array, int *k_array, double *surface_rho){
  
    int r = blockIdx.x%R;
    int z = (floorf(blockIdx.x/R) * max_threads) + threadIdx.x;

    if(r==0 || z==0 || r>=R || z>1){
    return;
    }
    double fr, fz;
    int i_drift, k_drift;
    fr =  fr_array[((R+1)*z)+r];
    fz = fz_array[((R+1)*z)+r];
    i_drift = i_array[((R+1)*z)+r];
    k_drift = k_array[((R+1)*z)+r];
    if (i_drift>=1 && i_drift <R && k_drift >=1 && k_drift<L) {
        if (i_drift > 1 && r > 1) {
            atomicAdd(&rho[(2*(L+1)*(R+1))+((R+1)*k_drift)+i_drift],surface_rho[(R+1)+r] * fr      *fz       * (double) (r-1) / (double) (i_drift-1));
            atomicAdd(&rho[(2*(L+1)*(R+1))+((R+1)*k_drift)+i_drift+1],surface_rho[(R+1)+r] * (1.0-fr)*fz       * (double) (r-1) / (double) (i_drift));
            atomicAdd(&rho[(2*(L+1)*(R+1))+((R+1)*(k_drift+1))+i_drift],surface_rho[(R+1)+r] * fr      *(1.0-fz) * (double) (r-1) / (double) (i_drift-1));
            atomicAdd(&rho[(2*(L+1)*(R+1))+((R+1)*(k_drift+1))+i_drift+1],surface_rho[(R+1)+r] * (1.0-fr)*(1.0-fz) * (double) (r-1) / (double) (i_drift));
        } 
        else if (i_drift > 1) {
            atomicAdd(&rho[(2*(L+1)*(R+1))+((R+1)*k_drift)+i_drift],surface_rho[(R+1)+r] * fr      *fz       / (double) (8*i_drift-8));
            atomicAdd(&rho[(2*(L+1)*(R+1))+((R+1)*k_drift)+i_drift+1],surface_rho[(R+1)+r] * (1.0-fr)*fz       / (double) (8*i_drift));
            atomicAdd(&rho[(2*(L+1)*(R+1))+((R+1)*(k_drift+1))+i_drift],surface_rho[(R+1)+r] * fr      *(1.0-fz) / (double) (8*i_drift-8));
            atomicAdd(&rho[(2*(L+1)*(R+1))+((R+1)*(k_drift+1))+i_drift+1],surface_rho[(R+1)+r] * (1.0-fr)*(1.0-fz) / (double) (8*i_drift));
        
        } 
        else if (r > 1) {
            atomicAdd(&rho[(2*(L+1)*(R+1))+((R+1)*k_drift)+i_drift],surface_rho[(R+1)+r] * fr      *fz       * (double) (8*(R+1)-8));
            atomicAdd(&rho[(2*(L+1)*(R+1))+((R+1)*k_drift)+i_drift+1],surface_rho[(R+1)+r] * (1.0-fr)*fz       * (double) (r-1));
            atomicAdd(&rho[(2*(L+1)*(R+1))+((R+1)*(k_drift+1))+i_drift],surface_rho[(R+1)+r] * fr      *(1.0-fz) * (double) (8*(R+1)-8));
            atomicAdd(&rho[(2*(L+1)*(R+1))+((R+1)*(k_drift+1))+i_drift+1],surface_rho[(R+1)+r] * (1.0-fr)*(1.0-fz) * (double) (r-1));
        }
        
        else {
        
            atomicAdd(&rho[(2*(L+1)*(R+1))+((R+1)*k_drift)+i_drift],surface_rho[(R+1)+r] * fr      *fz);
            atomicAdd(&rho[(2*(L+1)*(R+1))+((R+1)*k_drift)+i_drift+1],surface_rho[(R+1)+r] * (1.0-fr)*fz       / 8.0); // vol_0 / vol_1 = 1/8
            atomicAdd(&rho[(2*(L+1)*(R+1))+((R+1)*(k_drift+1))+i_drift],surface_rho[(R+1)+r] * fr      *(1.0-fz));
            atomicAdd(&rho[(2*(L+1)*(R+1))+((R+1)*(k_drift+1))+i_drift+1],surface_rho[(R+1)+r] * (1.0-fr)*(1.0-fz) / 8.0);
        }
    }
    else if(k_drift==0){
        if (i_drift > 1 && r > 1) {
            atomicAdd(&surface_rho[2*(R+1)+i_drift], surface_rho[(R+1)+r] * fr      * fz       * (double) (r-1) / (double) (i_drift-1));
            atomicAdd(&surface_rho[2*(R+1)+i_drift+1], surface_rho[(R+1)+r] * (1.0-fr)* fz       * (double) (r-1) / (double) (i_drift));
            atomicAdd(&rho[(2*(L+1)*(R+1))+((R+1)*(k_drift+1))+i_drift],surface_rho[(R+1)+r] * fr      *(1.0-fz) * (double) (r-1) / (double) (i_drift-1));
            atomicAdd(&rho[(2*(L+1)*(R+1))+((R+1)*(k_drift+1))+i_drift+1],surface_rho[(R+1)+r] * (1.0-fr)*(1.0-fz) * (double) (r-1) / (double) (i_drift));
        } 
        else if (i_drift > 1) {
            atomicAdd(&surface_rho[2*(R+1)+i_drift], surface_rho[(R+1)+r] * fr      *fz       / (double) (8*i_drift-8));
            atomicAdd(&surface_rho[2*(R+1)+i_drift+1], surface_rho[(R+1)+r] * (1.0-fr)*fz       / (double) (8*i_drift));
            atomicAdd(&rho[(2*(L+1)*(R+1))+((R+1)*(k_drift+1))+i_drift],surface_rho[(R+1)+r] * fr      *(1.0-fz) / (double) (8*i_drift-8));
            atomicAdd(&rho[(2*(L+1)*(R+1))+((R+1)*(k_drift+1))+i_drift+1],surface_rho[(R+1)+r] * (1.0-fr)*(1.0-fz) / (double) (8*i_drift));
        } 
        else if (r > 1) {
            atomicAdd(&surface_rho[2*(R+1)+i_drift], surface_rho[(R+1)+r] * fr      *fz       * (double) (8*(R+1)-8));
            atomicAdd(&surface_rho[2*(R+1)+i_drift+1], surface_rho[(R+1)+r] * (1.0-fr)*fz       * (double) (r-1));
            atomicAdd(&rho[(2*(L+1)*(R+1))+((R+1)*(k_drift+1))+i_drift],surface_rho[(R+1)+r] * fr      *(1.0-fz) * (double) (8*(R+1)-8));
            atomicAdd(&rho[(2*(L+1)*(R+1))+((R+1)*(k_drift+1))+i_drift+1],surface_rho[(R+1)+r] * (1.0-fr)*(1.0-fz) * (double) (r-1));
        } 
        else {   
            atomicAdd(&surface_rho[2*(R+1)+i_drift], surface_rho[(R+1)+r] * fr      *fz );
            atomicAdd(&surface_rho[2*(R+1)+i_drift+1], surface_rho[(R+1)+r] * (1.0-fr)*fz      / 8.0); // vol_0 / vol_1 = 1/8
            atomicAdd(&rho[(2*(L+1)*(R+1))+((R+1)*(k_drift+1))+i_drift],surface_rho[(R+1)+r] * fr      *(1.0-fz));
            atomicAdd(&rho[(2*(L+1)*(R+1))+((R+1)*(k_drift+1))+i_drift+1],surface_rho[(R+1)+r] * (1.0-fr)*(1.0-fz) / 8.0);
        }  
  }
}

__global__ void hvc_modicication(double *rho, double *surface_rho, char *point_type){

    int r = blockIdx.x%R;
    int z = (floorf(blockIdx.x/R) * max_threads) + threadIdx.x;
    
    if( r>=R || z>=L ){
      return;
    }
    if(point_type[((R+1)*z)+r]<=HVC){
        rho[(2*(L+1)*(R+1))+((R+1)*z)+r] = 0.0;
        if(z==1){
          surface_rho[2*(R+1)+r]=0.0;
        }
     }
}

__global__ void set_rho_zero( double *rho_e, double *rho_h, double *surface_rho_e, double *surface_rho_h){

  int r = blockIdx.x%R;
  int z = (floorf(blockIdx.x/R) * max_threads) + threadIdx.x;
  
  if(r==0 || z==0 || r>=R || z>=L){
      return;
    }
  rho_e[(0*(L+1)*(R+1))+((R+1)*z)+r] = rho_e[(2*(L+1)*(R+1))+((R+1)*z)+r];
  rho_h[(0*(L+1)*(R+1))+((R+1)*z)+r] = rho_h[(2*(L+1)*(R+1))+((R+1)*z)+r];
  if(z==1){
    surface_rho_e[r] = surface_rho_e[2*(R+1)+r];
    surface_rho_h[r] = surface_rho_h[2*(R+1)+r];
  }
}

__global__ void update_impurities(double *impurity_gpu, double *rho_e, double *rho_h, double *surface_rho_e, double *surface_rho_h, double e_over_E){

  int r = blockIdx.x%R;
  int z = (floorf(blockIdx.x/R) * max_threads) + threadIdx.x;
  
  if(r==0 || z==0 || r>=R || z>=L){
    return;
  }
  //  setup.impurity[z][r] = rho_e[3][z][r] + (rho_h[0][z][r] - rho_e[0][z][r]) * e_over_E * grid*grid/2.0;
  if(z==1){
    impurity_gpu[((R+1)*z)+r] = rho_e[(3*(L+1)*(R+1))+((R+1)*z)+r] + (rho_h[(0*(L+1)*(R+1))+((R+1)*z)+r]-rho_e[(0*(L+1)*(R+1))+((R+1)*z)+r]) * e_over_E * grid*grid/2.0;
    impurity_gpu[((R+1)*z)+r] += (surface_rho_h[r] - surface_rho_e[r])* e_over_E * grid*grid/2.0;
  }
  else{
    impurity_gpu[((R+1)*z)+r] = rho_e[(3*(L+1)*(R+1))+((R+1)*z)+r] + (rho_h[(0*(L+1)*(R+1))+((R+1)*z)+r] - rho_e[(0*(L+1)*(R+1))+((R+1)*z)+r]) * e_over_E * grid*grid/2.0;

  }
}

extern "C" int gpu_drift(MJD_Siggen_Setup *setup, int L_in, int R_in, float grid_in, double ***rho, int q_in, GPU_data *gpu_setup, int n_iter){


    grid = grid_in;
    L=L_in;
    R=R_in;
    q=q_in;
    max_threads=1024;
    tstep = setup-> step_time_calc;
    delta = 0.07*tstep;
    delta_r = 0.07*tstep;
    wrap_around_radius = setup->wrap_around_radius;
    ditch_thickness =  setup-> ditch_thickness;
    ditch_depth = setup-> ditch_depth;
    surface_drift_vel_factor = setup->surface_drift_vel_factor;
    fq = -q;
    passivated_surface_thickness_gpu = setup->passivated_thickness/grid;
  
  
  
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
    double *rho_gpu, *surface_rho;
    if (q < 0) { // electrons
      drift_offset_gpu = gpu_setup->drift_offset_e_gpu;
      drift_slope_gpu  = gpu_setup->drift_slope_e_gpu;
      rho_gpu = gpu_setup->rho_e_gpu;
      surface_rho=gpu_setup->surface_rho_e;
    } 
    else {   // holes
      drift_offset_gpu = gpu_setup->drift_offset_h_gpu;
      drift_slope_gpu  = gpu_setup->drift_slope_h_gpu;
      rho_gpu = gpu_setup->rho_h_gpu;
      surface_rho=gpu_setup->surface_rho_h;
    }
  
    int num_blocks = R * (ceil(L/max_threads)+1);
    // if(num_blocks<65536 && max_threads<=1024){
  
      reset_rho<<<num_blocks,max_threads>>>(rho_gpu, surface_rho, max_threads);
      cudaDeviceSynchronize();
  
      gpu_diffusion<<<num_blocks,max_threads>>>(rho_gpu, gpu_setup->v_gpu, gpu_setup->point_type_gpu, gpu_setup->dr_gpu, gpu_setup->dz_gpu, gpu_setup->s1_gpu, gpu_setup->s2_gpu, 
      drift_offset_gpu, drift_slope_gpu, gpu_setup->deltaez_array, gpu_setup->deltaer_array);
      cudaDeviceSynchronize();
      diff_update<<<num_blocks,max_threads>>>(rho_gpu, gpu_setup->point_type_gpu, gpu_setup->s1_gpu, gpu_setup->s2_gpu, gpu_setup->deltaez_array, gpu_setup->deltaer_array, surface_rho);
      cudaDeviceSynchronize();
  
      gpu_self_repulsion<<<num_blocks,max_threads>>>(rho_gpu, gpu_setup->v_gpu, gpu_setup->point_type_gpu, gpu_setup->dr_gpu, gpu_setup->dz_gpu, 
        gpu_setup->s1_gpu, gpu_setup->s2_gpu, drift_offset_gpu, drift_slope_gpu, gpu_setup->fr_array, gpu_setup->fz_array, 
        gpu_setup->i_array, gpu_setup->k_array, gpu_setup->velocity_drift_r, gpu_setup->velocity_drift_z, gpu_setup->field_r, gpu_setup->field_z);
      cudaDeviceSynchronize();
      gpu_sr_update<<<num_blocks,max_threads>>>(rho_gpu, gpu_setup->fr_array, gpu_setup->fz_array, gpu_setup->i_array, gpu_setup->k_array, surface_rho);
      cudaDeviceSynchronize();
  
      surface_drift_calc<<<num_blocks,max_threads>>>(rho_gpu, gpu_setup->velocity_drift_r, gpu_setup->velocity_drift_z, gpu_setup->field_r, gpu_setup->field_z, gpu_setup->fr_array, gpu_setup->fz_array, gpu_setup->i_array, gpu_setup->k_array, surface_rho);
      cudaDeviceSynchronize();
      surface_drift<<<num_blocks,max_threads>>>(rho_gpu, gpu_setup->fr_array, gpu_setup->fz_array, gpu_setup->i_array, gpu_setup->k_array, surface_rho);
      cudaDeviceSynchronize();
  
      hvc_modicication<<<num_blocks,max_threads>>>(rho_gpu, surface_rho, gpu_setup->point_type_gpu);
      cudaDeviceSynchronize();

      // //save velocities for holes
      // if(q>0){
      //   printf("CP 0\n");
  

      //   double *surface_rho_flat = (double*)malloc(sizeof(double)*(L+1)*(R+1)*1000000);
      //   double *rho_flat_h_cpu = (double*)malloc(sizeof(double)*(R+1)*1000000);
      //   double *vel_r = (double*)malloc(sizeof(double)*(L+1)*(R+1)*1000000);
      //   double *vel_z = (double*)malloc(sizeof(double)*(L+1)*(R+1)*1000000);

      //   for(int j=0; j<L; j++){
      //     for(int k=0; k<R; k++){
      //       // if(j==327){
      //       // printf("j=%d,k=%d ", j, k);
      //       // }
      //       rho_flat_h_cpu[((R+1)*j)+k] = 0.0;
      //       // vel_r[((R+1)*j)+k] = 0.0;
      //       // vel_z[((R+1)*j)+k] = 0.0;
      //     }
      //   }

      //   // cudaMemcpy(rho_flat_e_cpu, gpu_setup->rho_e_gpu, 4*sizeof(double)*(L+1)*(R+1), cudaMemcpyDeviceToHost);
      //   cudaMemcpy(rho_flat_h_cpu, rho_gpu, sizeof(double)*(L+1)*(R+1)*3, cudaMemcpyDeviceToHost);
      //   cudaMemcpy(surface_rho_flat, surface_rho, sizeof(double)*(R+1), cudaMemcpyDeviceToHost);
      //   cudaMemcpy(vel_r, gpu_setup->velocity_test_r, sizeof(double)*(L+1)*(R+1), cudaMemcpyDeviceToHost);
      //   cudaMemcpy(vel_z, gpu_setup->velocity_test_z, sizeof(double)*(L+1)*(R+1), cudaMemcpyDeviceToHost);


      //   char fname[1000];
      //   sprintf(fname, "/pscratch/sd/k/kbhimani/siggen_ccd_data/velocity_test/data_iter_%d.txt",n_iter);


      //   int written_3 = 0;

      //   FILE *f_3 = fopen(fname,"w");
      //   //written = fwrite(sig, sizeof(float), sizeof(sig), f);
      //   written_3 = fprintf(f_3,"r,z,vel_r,vel_z,rho\n");

      //   for(int x=0; x<R; x++){
      //     int y=1;
      //     if(surface_rho_flat[x]>0)
      //     {
      //       written_3 = fprintf(f_3,"%d,%d,%f,%f,%.10f\n",x,y,vel_r[((R+1)*x)+y]*surface_drift_vel_factor,vel_z[((R+1)*x)+y]*surface_drift_vel_factor, surface_rho_flat[x]);
      //     }
      //   }

      //   for(int j=0; j<L; j++){
      //     for(int k=0; k<R; k++){
      //       // printf("j=%d,k=%d ", j, k);
      //       if(rho_flat_h_cpu[(2*(L+1)*(R+1))+((R+1)*j)+k]>0){
      //       written_3 = fprintf(f_3,"%d,%d,%f,%f,%.10f\n",k,j,vel_r[((R+1)*j)+k],vel_z[((R+1)*j)+k], rho_flat_h_cpu[(2*(L+1)*(R+1))+((R+1)*j)+k]);
      //       }
      //       if (written_3 == 0) {
      //       printf("Error during writing to courant vector file!");
      //       break;
      //       }
      //     }
      //   }
      //   fclose(f_3);
      //   printf("Done writting velocity test\n");
      //   free(surface_rho_flat);
      //   free(rho_flat_h_cpu);
      //   free(vel_r);
      //   free(vel_z);
      // }



    return 0;
  }

extern "C" void set_rho_zero_gpu(GPU_data *gpu_setup, int L_in, int R_in, int num_blocks, int num_threads){
    L=L_in;
    R=R_in;
    max_threads=1024;
    set_rho_zero<<<num_blocks,num_threads>>>(gpu_setup->rho_e_gpu, gpu_setup->rho_h_gpu, gpu_setup->surface_rho_e, gpu_setup->surface_rho_h);
  }
  
extern "C" void update_impurities_gpu(GPU_data *gpu_setup, int L_in, int R_in, int num_blocks, int num_threads, double e_over_E, float grid_in){
    grid = grid_in;
    L=L_in;
    R=R_in;
    max_threads=1024;
  
    update_impurities<<<num_blocks,num_threads>>>(gpu_setup->impurity_gpu, gpu_setup->rho_e_gpu, gpu_setup->rho_h_gpu, gpu_setup->surface_rho_e, gpu_setup->surface_rho_h, e_over_E);
  }