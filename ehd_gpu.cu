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


/*
  Can be run as ./ehdrift config_files/P42575A.config -a 25.00 -z 0.10 -g P42575A -s 0.00
  WP can be calculated as ./ehdrift config_files/P42575A_calc_wp.config -a 15.00 -z 0.10 -g P42575A -s 0.00
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size (x,y,z): (65535, 65535, 65535)
*/

extern "C" int gpu_drift(MJD_Siggen_Setup *setup, int L, int R, float grid, float ***rho, float ***rho_test, int q, double *gone);
int drift_rho(MJD_Siggen_Setup *setup, int L, int R, float grid, float ***rho, int q, double *gone);

__managed__ float fq;
__managed__ int testVar;
__managed__ float drift_E[20]= {0.000,  100.,  160.,  240.,  300.,  500.,  600., 750.0, 1000., 1250., 1500., 1750., 2000., 2500., 3000., 3500., 4000., 4500., 5000., 1e10};
__managed__ int idid,idod,idd;
__managed__ double f_drift;

__managed__ float tstep;
__managed__ float delta;
__managed__ float delta_r;
__managed__ float wrap_around_radius;
__managed__ float ditch_thickness;
__managed__ float ditch_depth;
__managed__ float surface_drift_vel_factor;


__global__ void gpu_diffusion(int L, int R, float grid, float *rho,int q, double *v, char *point_type,  
  double *dr, double *dz, double *s1, double *s2, float *drift_offset, float *drift_slope, int max_threads, double *deltaez_array, double *deltaer_array){


  //printf("Time step is %d \n", setup-> step_time_calc);
  //formlaa for index n,z,r is (n*(L+1)*(R+1))+((R+1)*z)+r

  int r = blockIdx.x%R;
  int z = (floorf(blockIdx.x/R) * max_threads) + threadIdx.x;
  
  if(r==0 || z==0 || r>=R || z>=(L-2)){
    return;
  }
  int i, new_gpu_relax = 0;
  float E, E_r, E_z;
  double ve_z, ve_r, deltaez, deltaer;
  // if((r>=R) || (z>=L-2)){
  // printf("-----------Segfault error at radius=%d and z position=%d-----------\n",r,z);
  // }

  
  if (rho[(0*(L+1)*(R+1))+((R+1)*z)+r] < 1.0e-14) {
    rho[(1*(L+1)*(R+1))+((R+1)*z)+r] += rho[(0*(L+1)*(R+1))+((R+1)*z)+r];
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

  if(z==6 && r==1253){
    printf("In GPU at 1, at z=%d and r=%d, the value of rho is %.7f deltaez is %.7f and deltaer is %.7f\n", z, r, rho[(1*(L+1)*(R+1))+((R+1)*z)+r], deltaez, deltaer);
  }

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

    if(z==6 && r==1253){
      printf("In GPU at 2, at z=%d and r=%d, the value of rho is %.7f deltaez is %.7f and deltaer is %.7f\n", z, r, rho[(1*(L+1)*(R+1))+((R+1)*z)+r], deltaez, deltaer);
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

  if(z==6 && r==1253){
    printf("In GPU at 3, at z=%d and r=%d, the value of rho is %.7f deltaez is %.7f and deltaer is %.7f\n", z, r, rho[(1*(L+1)*(R+1))+((R+1)*z)+r], deltaez, deltaer);
  }

  deltaez_array[((R+1)*z)+r] = deltaez;
  deltaer_array[((R+1)*z)+r] = deltaer;

  // if (r < R-1 && point_type[((R+1)*z)+r+1] != DITCH) {
  //   rho[(1*(L+1)*(R+1))+((R+1)*z)+r+1] += rho[(0*(L+1)*(R+1))+((R+1)*z)+r]*deltaer * s1[r] * (double) (r-1) / (double) (r);
  //   rho[(1*(L+1)*(R+1))+((R+1)*z)+r]   -= rho[(0*(L+1)*(R+1))+((R+1)*z)+r]*deltaer * s1[r];
  //   //printf("value of rho at checkpoint 2 is %f \n",rho[(1*(L+1)*(R+1))+((R+1)*z)+r]);

  //   }
  // if (z > 1 && point_type[((R+1)*(z-1))+r] != DITCH) {
  //   rho[(1*(L+1)*(R+1))+((R+1)*(z-1))+r] += rho[(0*(L+1)*(R+1))+((R+1)*z)+r]*deltaez;
  //   rho[(1*(L+1)*(R+1))+((R+1)*z)+r] -= rho[(0*(L+1)*(R+1))+((R+1)*z)+r]*deltaez;
  //   }
  // if (z < L-1 && point_type[((R+1)*(z+1))+r] != DITCH) {
  //   rho[(1*(L+1)*(R+1))+((R+1)*(z+1))+r] += rho[(0*(L+1)*(R+1))+((R+1)*z)+r]*deltaez;
  //   rho[(1*(L+1)*(R+1))+((R+1)*z)+r]   -= rho[(0*(L+1)*(R+1))+((R+1)*z)+r]*deltaez;
  //   }
  // if (r > 2 && point_type[((R+1)*z)+r-1] != DITCH) {

  //   rho[(1*(L+1)*(R+1))+((R+1)*z)+(r-1)] += rho[(0*(L+1)*(R+1))+((R+1)*z)+r]*deltaer * s2[r] * (double) (r-1) / (double) (r-2);
  //   rho[(1*(L+1)*(R+1))+((R+1)*z)+r]   -= rho[(0*(L+1)*(R+1))+((R+1)*z)+r]*deltaer * s2[r];
  // }

}

  __global__ void diff_update(int L, int R, float *rho, char *point_type, double *s1, double *s2, double *deltaez_array, double *deltaer_array){
    for (int r=1; r<R; r++) {
      for (int z=1; z<L-2; z++) {
        if (rho[(0*(L+1)*(R+1))+((R+1)*z)+r] < 1.0e-14) {
          //printf("-----------EXCITING THE KERNAL-----------\n");
          continue;
        }
        double deltaez = deltaez_array[((R+1)*z)+r];
        double deltaer = deltaer_array[((R+1)*z)+r];

          // enum point_types{PC, HVC, INSIDE, PASSIVE, PINCHOFF, DITCH, DITCH_EDGE, CONTACT_EDGE};
        rho[(1*(L+1)*(R+1))+((R+1)*z)+r] += rho[(0*(L+1)*(R+1))+((R+1)*z)+r];

        if((z==6||z==5||z==4) && (r==1253||r==1252||r==1254)){
          printf("In GPU at 1, at z=%d and r=%d, the value of rho is %.7f \n", z, r, rho[(1*(L+1)*(R+1))+((R+1)*z)+r]);
        }

        if (r < R-1 && point_type[((R+1)*z)+r+1] != DITCH) {
          rho[(1*(L+1)*(R+1))+((R+1)*z)+r+1] += rho[(0*(L+1)*(R+1))+((R+1)*z)+r]*deltaer * s1[r] * (double) (r-1) / (double) (r);
          rho[(1*(L+1)*(R+1))+((R+1)*z)+r]   -= rho[(0*(L+1)*(R+1))+((R+1)*z)+r]*deltaer * s1[r];
          //printf("value of rho at checkpoint 2 is %f \n",rho[(1*(L+1)*(R+1))+((R+1)*z)+r]);
      
          }

          if((z==6||z==5||z==4) && (r==1253||r==1252||r==1254)){
            printf("In GPU at 2, at z=%d and r=%d, the value of rho is %.7f \n", z, r, rho[(1*(L+1)*(R+1))+((R+1)*z)+r]);
          }
        if (z > 1 && point_type[((R+1)*(z-1))+r] != DITCH) {
          rho[(1*(L+1)*(R+1))+((R+1)*(z-1))+r] += rho[(0*(L+1)*(R+1))+((R+1)*z)+r]*deltaez;
          rho[(1*(L+1)*(R+1))+((R+1)*z)+r] -= rho[(0*(L+1)*(R+1))+((R+1)*z)+r]*deltaez;
          }

          if((z==6||z==5||z==4) && (r==1253||r==1252||r==1254)){
            printf("In GPU at 3, at z=%d and r=%d, the value of rho is %.7f \n", z, r, rho[(1*(L+1)*(R+1))+((R+1)*z)+r]);
          }
        if (z < L-1 && point_type[((R+1)*(z+1))+r] != DITCH) {
          rho[(1*(L+1)*(R+1))+((R+1)*(z+1))+r] += rho[(0*(L+1)*(R+1))+((R+1)*z)+r]*deltaez;
          rho[(1*(L+1)*(R+1))+((R+1)*z)+r]   -= rho[(0*(L+1)*(R+1))+((R+1)*z)+r]*deltaez;
          }

          if((z==6||z==5||z==4) && (r==1253||r==1252||r==1254)){
            printf("In GPU at 4, at z=%d and r=%d, the value of rho is %.7f \n", z, r, rho[(1*(L+1)*(R+1))+((R+1)*z)+r]);
          }
        if (r > 2 && point_type[((R+1)*z)+r-1] != DITCH) {
      
          rho[(1*(L+1)*(R+1))+((R+1)*z)+(r-1)] += rho[(0*(L+1)*(R+1))+((R+1)*z)+r]*deltaer * s2[r] * (double) (r-1) / (double) (r-2);
          rho[(1*(L+1)*(R+1))+((R+1)*z)+r]   -= rho[(0*(L+1)*(R+1))+((R+1)*z)+r]*deltaer * s2[r];
        }

        if((z==6||z==5||z==4) && (r==1253||r==1252||r==1254)){
          printf("In GPU at 5, at z=%d and r=%d, the value of rho is %.7f \n", z, r, rho[(1*(L+1)*(R+1))+((R+1)*z)+r]);
        }
      }
    }
    printf("In GPU at 7, at z=%d and r=%d, the value of rho is %.7f \n", 6, 1253, rho[(1*(L+1)*(R+1))+((R+1)*6)+1253]);

  }

  __global__ void gpu_self_repulsion(int L, int R, float grid, float *rho,int q, double *v, char *point_type,  double *dr, double *dz, 
    double *s1, double *s2, float *drift_offset, float *drift_slope, int max_threads, double *fr_array, double *fz_array, int *i_array, int *k_array){
  
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
      rho[(2*(L+1)*(R+1))+((R+1)*z)+r] += rho[(1*(L+1)*(R+1))+((R+1)*z)+r];
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
  
    
  
    if (1 && r == 100 && z == 10)
      printf("r z: %d %d; E_r i_array[((R+1)*z)+r]dre: %f %d %f; fr_array[((R+1)*z)+r] = %f\n"
            "r z: %d %d; E_z k_array[((R+1)*z)+r]dze: %f %d %f; fz_array[((R+1)*z)+r] = %f\n",
            r, z, E_r, i_array[((R+1)*z)+r], dre, fr_array[((R+1)*z)+r], r, z, E_z, k_array[((R+1)*z)+r], dze, fz_array[((R+1)*z)+r]);

  }
  
  __global__ void gpu_sr_update(int L, int R, float *rho, double *fr_array, double *fz_array, int *i_array, int *k_array){

    for (int r=1; r<R; r++) {
      for (int z=1; z<L-2; z++) {
      
      if (rho[(1*(L+1)*(R+1))+((R+1)*z)+r] < 1.0e-14) {
        continue;
      }

        if (i_array[((R+1)*z)+r]>=1 && i_array[((R+1)*z)+r]<R && k_array[((R+1)*z)+r]>=1 && k_array[((R+1)*z)+r]<L) {
          if (i_array[((R+1)*z)+r] > 1 && r > 1) {
            rho[(2*(L+1)*(R+1))+((R+1)*k_array[((R+1)*z)+r])+i_array[((R+1)*z)+r]] += rho[(1*(L+1)*(R+1))+((R+1)*z)+r] * fr_array[((R+1)*z)+r]      *fz_array[((R+1)*z)+r]       * (double) (r-1) / (double) (i_array[((R+1)*z)+r]-1);
            rho[(2*(L+1)*(R+1))+((R+1)*k_array[((R+1)*z)+r])+i_array[((R+1)*z)+r]+1] += rho[(1*(L+1)*(R+1))+((R+1)*z)+r] * (1.0-fr_array[((R+1)*z)+r])*fz_array[((R+1)*z)+r]       * (double) (r-1) / (double) (i_array[((R+1)*z)+r]);
            rho[(2*(L+1)*(R+1))+((R+1)*(k_array[((R+1)*z)+r]+1))+i_array[((R+1)*z)+r]] += rho[(1*(L+1)*(R+1))+((R+1)*z)+r] * fr_array[((R+1)*z)+r]      *(1.0-fz_array[((R+1)*z)+r]) * (double) (r-1) / (double) (i_array[((R+1)*z)+r]-1);
            rho[(2*(L+1)*(R+1))+((R+1)*(k_array[((R+1)*z)+r]+1))+i_array[((R+1)*z)+r]+1] += rho[(1*(L+1)*(R+1))+((R+1)*z)+r] * (1.0-fr_array[((R+1)*z)+r])*(1.0-fz_array[((R+1)*z)+r]) * (double) (r-1) / (double) (i_array[((R+1)*z)+r]);

          } 
          else if (i_array[((R+1)*z)+r] > 1) {  // r == 0
            rho[(2*(L+1)*(R+1))+((R+1)*k_array[((R+1)*z)+r])+i_array[((R+1)*z)+r]] += rho[(1*(L+1)*(R+1))+((R+1)*z)+r] * fr_array[((R+1)*z)+r]      *fz_array[((R+1)*z)+r]       / (double) (8*i_array[((R+1)*z)+r]-8);
            rho[(2*(L+1)*(R+1))+((R+1)*k_array[((R+1)*z)+r])+i_array[((R+1)*z)+r]+1] += rho[(1*(L+1)*(R+1))+((R+1)*z)+r] * (1.0-fr_array[((R+1)*z)+r])*fz_array[((R+1)*z)+r]       / (double) (8*i_array[((R+1)*z)+r]);
            rho[(2*(L+1)*(R+1))+((R+1)*(k_array[((R+1)*z)+r]+1))+i_array[((R+1)*z)+r]] += rho[(1*(L+1)*(R+1))+((R+1)*z)+r] * fr_array[((R+1)*z)+r]      *(1.0-fz_array[((R+1)*z)+r]) / (double) (8*i_array[((R+1)*z)+r]-8);
            rho[(2*(L+1)*(R+1))+((R+1)*(k_array[((R+1)*z)+r]+1))+i_array[((R+1)*z)+r]+1] += rho[(1*(L+1)*(R+1))+((R+1)*z)+r] * (1.0-fr_array[((R+1)*z)+r])*(1.0-fz_array[((R+1)*z)+r]) / (double) (8*i_array[((R+1)*z)+r]);
          } 
          else if (r > 1) {  // i_array[((R+1)*z)+r] == 0
            rho[(2*(L+1)*(R+1))+((R+1)*k_array[((R+1)*z)+r])+i_array[((R+1)*z)+r]] += rho[(1*(L+1)*(R+1))+((R+1)*z)+r] * fr_array[((R+1)*z)+r]      *fz_array[((R+1)*z)+r]       * (double) (8*(R+1)-8);
            rho[(2*(L+1)*(R+1))+((R+1)*k_array[((R+1)*z)+r])+i_array[((R+1)*z)+r]+1] += rho[(1*(L+1)*(R+1))+((R+1)*z)+r] * (1.0-fr_array[((R+1)*z)+r])*fz_array[((R+1)*z)+r]       * (double) (r-1);
            rho[(2*(L+1)*(R+1))+((R+1)*(k_array[((R+1)*z)+r]+1))+i_array[((R+1)*z)+r]] += rho[(1*(L+1)*(R+1))+((R+1)*z)+r] * fr_array[((R+1)*z)+r]      *(1.0-fz_array[((R+1)*z)+r]) * (double) (8*(R+1)-8);
            rho[(2*(L+1)*(R+1))+((R+1)*(k_array[((R+1)*z)+r]+1))+i_array[((R+1)*z)+r]+1] += rho[(1*(L+1)*(R+1))+((R+1)*z)+r] * (1.0-fr_array[((R+1)*z)+r])*(1.0-fz_array[((R+1)*z)+r]) * (double) (r-1);
          } 
          else {             // r == i_array[((R+1)*z)+r] == 0
            rho[(2*(L+1)*(R+1))+((R+1)*k_array[((R+1)*z)+r])+i_array[((R+1)*z)+r]] += rho[(1*(L+1)*(R+1))+((R+1)*z)+r] * fr_array[((R+1)*z)+r]      *fz_array[((R+1)*z)+r];
            rho[(2*(L+1)*(R+1))+((R+1)*k_array[((R+1)*z)+r])+i_array[((R+1)*z)+r]+1] += rho[(1*(L+1)*(R+1))+((R+1)*z)+r] * (1.0-fr_array[((R+1)*z)+r])*fz_array[((R+1)*z)+r]       / 8.0; // vol_0 / vol_1 = 1/8
            rho[(2*(L+1)*(R+1))+((R+1)*(k_array[((R+1)*z)+r]+1))+i_array[((R+1)*z)+r]] += rho[(1*(L+1)*(R+1))+((R+1)*z)+r] * fr_array[((R+1)*z)+r]      *(1.0-fz_array[((R+1)*z)+r]);
            rho[(2*(L+1)*(R+1))+((R+1)*(k_array[((R+1)*z)+r]+1))+i_array[((R+1)*z)+r]+1] += rho[(1*(L+1)*(R+1))+((R+1)*z)+r] * (1.0-fr_array[((R+1)*z)+r])*(1.0-fz_array[((R+1)*z)+r]) / 8.0;
          }
        }
      }
    }
  }

/* -------------------------------------- gpu_drift ------------------- */
// do the diffusion and drifting of the charge cloud densities
extern "C" int gpu_drift(MJD_Siggen_Setup *setup, int L, int R, float grid, float ***rho, float ***rho_test, int q, double *gone){
  
  tstep = setup-> step_time_calc;
  delta = 0.07*tstep;
  delta_r = 0.07*tstep;
  wrap_around_radius = setup->wrap_around_radius;
  ditch_thickness =  setup-> ditch_thickness;
  ditch_depth = setup-> ditch_depth;
  surface_drift_vel_factor = setup->surface_drift_vel_factor;

  grid = setup->xtal_grid;
    /* NOTE that impurity and field arrays in setup start at (i,j)=(1,1) for (r,z)=(0,0) */
    idid = lrint((wrap_around_radius - ditch_thickness)/grid) + 1; // ditch ID
    idod = lrint(wrap_around_radius/grid) + 1; // ditch OD
    idd =  lrint(ditch_depth/grid) + 1;        // ditch depth



   
    double *v_gpu;
    double *v_flat;
    v_flat = (double*)malloc(2*sizeof(double)*(L+1)*(R+1));
    cudaMalloc((void**)&v_gpu, 2*sizeof(double)*(L+1)*(R+1));
    for(int i=0; i<2; i++) {
      for(int j=0; j<=L; j++){
        for(int k=0; k<=R; k++){
          v_flat[(i*(L+1)*(R+1))+((R+1)*j)+k] = setup->v[i][j][k];
        }
      }
    }
    cudaMemcpy(v_gpu, v_flat, 2*sizeof(double)*(L+1)*(R+1), cudaMemcpyHostToDevice);  


    char  *point_type_flat;
    char  *point_type_gpu;
    point_type_flat = (char*)malloc(sizeof(char)*(L+1)*(R+1));
    cudaMalloc((void**)&point_type_gpu, sizeof(char)*(L+1)*(R+1));
    for(int j=0; j<=L; j++){
        for(int k=0; k<=R; k++){
          //printf("The value of point type at r = %d and z = %d is %.4c \n", k, j, setup->point_type[j][k]);
          point_type_flat[((R+1)*j)+k] = setup->point_type[j][k];
        }
      }
    cudaMemcpy(point_type_gpu, point_type_flat, sizeof(char)*(L+1)*(R+1), cudaMemcpyHostToDevice);
  
    double *dr_flat;
    double *dr_gpu;
    dr_flat = (double*)malloc(2*sizeof(double)*(L+1)*(R+1));
    cudaMalloc((void**)&dr_gpu, 2*sizeof(double)*(L+1)*(R+1));
      for(int i=0; i<2; i++) {
        for(int j=1; j<=L; j++){
            for(int k=0; k<=R; k++){
              dr_flat[(i*(L+1)*(R+1))+((R+1)*j)+k] = setup->dr[i][j][k];
            }
          }
        }
    cudaMemcpy(dr_gpu, dr_flat, 2*sizeof(double)*(L+1)*(R+1), cudaMemcpyHostToDevice);
    
    double *dz_flat;
    double *dz_gpu;
    dz_flat = (double*)malloc(2*sizeof(double)*(L+1)*(R+1));
    cudaMalloc((void**)&dz_gpu, 2*sizeof(double)*(L+1)*(R+1));
      for(int i=0; i<2; i++) {
        for(int j=1; j<=L; j++){
            for(int k=0; k<=R; k++){
              dz_flat[(i*(L+1)*(R+1))+((R+1)*j)+k] = setup->dz[i][j][k];
            }
          }
        }
    cudaMemcpy(dz_gpu, dz_flat, 2*sizeof(double)*(L+1)*(R+1), cudaMemcpyHostToDevice);
    
    double *s1_gpu, *s2_gpu;
    
    cudaMalloc((void**)&s1_gpu, sizeof(double)*(R+1));
    cudaMemcpy(s1_gpu, setup->s1, sizeof(double)*(R+1), cudaMemcpyHostToDevice);
    
    
    cudaMalloc((void**)&s2_gpu, sizeof(double)*(R+1));
    cudaMemcpy(s2_gpu, setup->s2, sizeof(double)*(R+1), cudaMemcpyHostToDevice);

    double *deltaez_array, *deltaer_array;
    cudaMalloc((void**)&deltaez_array, sizeof(double)*(L+1)*(R+1));
    cudaMalloc((void**)&deltaer_array, sizeof(double)*(L+1)*(R+1));

    double *fr_array, *fz_array;
    int *i_array, *k_array;
    cudaMalloc((void**)&fr_array, sizeof(double)*(L+1)*(R+1));
    cudaMalloc((void**)&fz_array, sizeof(double)*(L+1)*(R+1));
    cudaMalloc((void**)&i_array, sizeof(int)*(L+1)*(R+1));
    cudaMalloc((void**)&k_array, sizeof(int)*(L+1)*(R+1));


  // int index = 0;
  // for(int i=0; i<3; i++) {
  //   for(int j=0; j<L; j++){
  //     for(int k=0; k<R; k++){
  //       printf("Value at i=%d, j=%d, k=%d is=%f\n", i, j, k,rho[i][j][k]);
  //       printf("index is: %d\n",index);
  //       printf("Simulated index is: %d\n",(i*(L+1)*(R+1))+((R+1)*j)+k);
  //       index++;
  //     }
  //   }
  // }
 
    /* ASSUMPTIONS:
       0.1 mm grid size, 1 ns steps
       detector temperature = REF_TEMP = 77 K
       dealing only with electrons (no holes)
    */
    int i, r, z;

    float E_r, E_z;
    double ve_z, ve_r, deltaez, deltaer;
    
    fq = -q;
    deltaez = delta;
    deltaer = delta_r;


    float drift_offset_e[20]={0.0,   0.027, 0.038, 0.049 ,0.055, 0.074, 0.081,
          0.089, 0.101, 0.109, 0.116, 0.119, 0.122, 0.125,
          0.1275,0.1283,0.1288,0.1291,0.1293,0.1293};
    float drift_slope_e[20];

    float drift_offset_h[20]={0.0,   0.036, 0.047, 0.056, 0.06,  0.072, 0.077,
          0.081, 0.086, 0.089, 0.0925,0.095, 0.097, 0.1,
          0.1025,0.1036,0.1041,0.1045,0.1047,0.1047};
    float drift_slope_h[20];
    float *drift_offset, *drift_slope;
  
    for (i=0; i<20; i++) {
      drift_offset_e[i] /= grid;   // drift velocities in units of grid length
      drift_offset_h[i] /= grid;
    }
    for (i=0; i<19; i++) {
      drift_slope_e[i] = (drift_offset_e[i+1] - drift_offset_e[i]) /
                         (drift_E[i+1] - drift_E[i]);
      drift_slope_h[i] = (drift_offset_h[i+1] - drift_offset_h[i]) /
                         (drift_E[i+1] - drift_E[i]);
    }
    if (q < 0) { // electrons
      drift_offset = drift_offset_e;
      drift_slope  = drift_slope_e;
    } else {   // holes
      drift_offset = drift_offset_h;
      drift_slope  = drift_slope_h;
    }
  
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
    // f *= 0.5;                   // artifically reduce diffusion to 50%
    E_r = E_z = 100; // just to get started; will change later
    for (i=0; E_z > drift_E[i+1]; i++);
    ve_z = fq * (drift_offset[i] + drift_slope[i]*(E_z - drift_E[i]))/E_z;
    deltaez = grid * ve_z * f_drift;
    for (i=0; E_r > drift_E[i+1]; i++);
    ve_r = fq * (drift_offset[i] + drift_slope[i]*(E_r - drift_E[i]))/E_r;
    deltaer = grid * ve_r * f_drift;
    printf ("D_z, D_r values (q=%d) at 100 V/cm: %f %f\n", q, deltaez, deltaer);
  

    for (z=0; z<L; z++) {
      for (r=0; r<R; r++) {
        rho_test[1][z][r] = rho_test[2][z][r] = 0;
      }
    }

  float *drift_offset_gpu, *drift_slope_gpu;
  
  cudaMalloc((void**)&drift_offset_gpu, sizeof(float)*20);
  cudaMemcpy(drift_offset_gpu, drift_offset, sizeof(float)*20, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&drift_slope_gpu, sizeof(float)*20);
  cudaMemcpy(drift_slope_gpu, drift_slope, sizeof(float)*20, cudaMemcpyHostToDevice);

  float *rho_cpu_flat;
  float *rho_gpu;

  rho_cpu_flat = (float*)malloc(3*sizeof(float)*(L+1)*(R+1));
  cudaMalloc((void**)&rho_gpu, 3*sizeof(float)*(L+1)*(R+1));

  for(int i=0; i<3; i++) {
    for(int j=0; j<L; j++){
        for(int k=0; k<R; k++){
          rho_cpu_flat[(i*(L+1)*(R+1))+((R+1)*j)+k]=rho_test[i][j][k];
        }
      }
    }


  // for(int i=0; i<3; i++) {
  //   //cudaMemcpy(&rho_gpu[(i*(L+1)*(R+1))], &rho[i], sizeof(float)*(L+1), cudaMemcpyHostToDevice);
  //   for(int j=0; j<L; j++){
  //     cudaMemcpy(&rho_gpu[(i*(L+1)*(R+1))+((R+1)*j)], &rho[i][j], sizeof(float)*(R+1), cudaMemcpyHostToDevice);
  //   }
  // }

  cudaMemcpy(rho_gpu, rho_cpu_flat, 3*sizeof(float)*(L+1)*(R+1), cudaMemcpyHostToDevice);


  // printf("Allocation and memory copy successfull\n");

  // printf("Executing the kernel\n");
  int num_threads = 300;
  int num_blocks = R * (ceil(L/num_threads)+1); //The +1 is just a precaution to make sure all R and Z values are included

  if(num_blocks<65535){
    gpu_diffusion<<<num_blocks,num_threads>>>(L, R, grid, rho_gpu, q, v_gpu, point_type_gpu, dr_gpu, dz_gpu, s1_gpu, s2_gpu, drift_offset_gpu, drift_slope_gpu, num_threads, deltaez_array, deltaer_array);
    cudaDeviceSynchronize();
    diff_update<<<1,1>>>(L, R, rho_gpu, point_type_gpu, s1_gpu, s2_gpu, deltaez_array, deltaer_array);
    cudaDeviceSynchronize();
    gpu_self_repulsion<<<num_blocks,num_threads>>>(L, R, grid, rho_gpu, q, v_gpu, point_type_gpu, dr_gpu, dz_gpu, s1_gpu, s2_gpu, drift_offset_gpu, drift_slope_gpu, num_threads, fr_array, fz_array, i_array, k_array);
    cudaDeviceSynchronize();
    gpu_sr_update<<<1,1>>>(L, R, rho_gpu, fr_array, fz_array, i_array, k_array);
  }
  else{
    printf("----------------Pick a smaller block please----------------\n");
    return 0;
  }

  // printf("Done executing the kernel \n");

  cudaDeviceSynchronize();


  // for(int o=0; o<3; o++) {
  //   //cudaMemcpy(&rho[i], &rho_gpu[(i*(L+1)*(R+1))], sizeof(float)*(L+1), cudaMemcpyDeviceToHost);
  //   for(int p=0; p<L; p++){
  //     cudaMemcpy(&rho[o][p], &rho_gpu[(o*(L+1)*(R+1))+((R+1)*p)], sizeof(float)*(R+1), cudaMemcpyDeviceToHost);
  //   }
  // }

      
  // printf("copying memory the rho density\n");
  cudaMemcpy(rho_cpu_flat, rho_gpu, 3*sizeof(float)*(L+1)*(R+1), cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  for(int i=0; i<3; i++) {
    for(int j=0; j<L; j++){
        for(int k=0; k<R; k++){
          memcpy(&rho_test[i][j][k], &rho_cpu_flat[(i*(L+1)*(R+1))+((R+1)*j)+k], sizeof(float));
        }
      }
    }


  for (z=0; z<L; z++) {
    for (r=0; r<R; r++) {
      if (setup->point_type[z][r] <= HVC) {
        //*gone += rho_test[2][z][r] * r;
        rho_test[2][z][r] = 0;
      }
    }
  }


  drift_rho(setup, L, R, grid, rho, q, gone);

  // printf("R=%d, Z=%d\n",R,L);
  printf("\n--------Running tests to compare the outcomes of CPU and GPU--------\n");

  #define MAX_ERR 1e-7
  int error_count = 0;
  for(int i=0; i<3; i++) {
    for(int j=0; j<L; j++){
      for(int k=0; k<R; k++){
        if(fabs(rho[i][j][k] - rho_test[i][j][k]) > MAX_ERR){
          error_count++;
          printf("Error at i=%d, z=%d and r=%d, actual answer is %f, calculated answer is %f \n",i,j,k, rho[i][j][k], rho_test[i][j][k]);
        }
      }
    }
  }
  if(error_count>0){
    printf("Total number of errors = %d \n", error_count);
  }
  else{
    printf("No errors found!!!!\n\n");
  }

  // for(int i=0; i<3; i++) {
  //   for(int j=0; j<L; j++){
  //       free(rho_test[i][j]);
  //     }
  //   }


  cudaFree(rho_gpu);
  cudaFree(v_gpu);
  cudaFree(point_type_gpu);
  cudaFree(dr_gpu);
  cudaFree(dz_gpu);
  cudaFree(s1_gpu);
  cudaFree(s2_gpu);
  cudaFree(fr_array);
  cudaFree(fz_array);
  cudaFree(i_array);
  cudaFree(k_array);
  cudaFree(deltaez_array);
  cudaFree(deltaer_array);

  cudaFree(drift_offset_gpu);
  cudaFree(drift_slope_gpu);
  free(rho_cpu_flat);
  free(v_flat);
  free(point_type_flat);
  free(dr_flat);
  free(dz_flat);

  return 0;

}

/* -------------------------------------- drift_rho ------------------- */
// do the diffusion and drifting of the charge cloud densities

int drift_rho(MJD_Siggen_Setup *setup, int L, int R, float grid, float ***rho,
  int q, double *gone) {

  #define TSTEP   setup-> step_time_calc  // time step of calculation, in ns; do not exceed 2.0 !!!
  #define DELTA   (0.07*TSTEP) /* Prob. of a charge moving to next 20-micron bin during a time step */
  #define DELTA_R (0.07*TSTEP) /* ... in r-direction */

  /* ASSUMPTIONS:
  0.1 mm grid size, 1 ns steps
  detector temperature = REF_TEMP = 77 K
  dealing only with electrons (no holes)
  */

  int    i, k, r, z, new_var = 0;
  double dre, dze;
  double ***v = setup->v;
  float  E_r, E_z, E, fq = -q;
  double ve_z, ve_r, f, fr, fz, deltaez = DELTA, deltaer = DELTA_R;
  float drift_E[20]=       {0.000,  100.,  160.,  240.,  300.,  500.,  600.,
  750.0, 1000., 1250., 1500., 1750., 2000., 2500.,
  3000., 3500., 4000., 4500., 5000., 1e10};
  float drift_offset_e[20]={0.0,   0.027, 0.038, 0.049 ,0.055, 0.074, 0.081,
  0.089, 0.101, 0.109, 0.116, 0.119, 0.122, 0.125,
  0.1275,0.1283,0.1288,0.1291,0.1293,0.1293};
  float drift_slope_e[20];

  float drift_offset_h[20]={0.0,   0.036, 0.047, 0.056, 0.06,  0.072, 0.077,
  0.081, 0.086, 0.089, 0.0925,0.095, 0.097, 0.1,
  0.1025,0.1036,0.1041,0.1045,0.1047,0.1047};
  float drift_slope_h[20];
  float *drift_offset, *drift_slope;


  for (i=0; i<20; i++) {
  drift_offset_e[i] /= grid;   // drift velocities in units of grid length
  drift_offset_h[i] /= grid;
  }
  for (i=0; i<19; i++) {
  drift_slope_e[i] = (drift_offset_e[i+1] - drift_offset_e[i]) /
            (drift_E[i+1] - drift_E[i]);
  drift_slope_h[i] = (drift_offset_h[i+1] - drift_offset_h[i]) /
            (drift_E[i+1] - drift_E[i]);
  }
  if (q < 0) { // electrons
  drift_offset = drift_offset_e;
  drift_slope  = drift_slope_e;
  } else {   // holes
  drift_offset = drift_offset_h;
  drift_slope  = drift_slope_h;
  }

  f = 1.2e6; // * setup.xtal_temp/REF_TEMP;
  f *= TSTEP / 4000.0;
  /* above is my own approximate parameterization of measurements of Jacoboni et al.
  1.2e6 * v_over_E   ~   D in cm2/s
  v_over_E = drift velocity / electric field   ~  mu
  note that Einstein's equation is D = mu*kT/e
  kT/e ~ 0.007/V ~ 0.07 mm/Vcm, => close enough to 0.12, okay
  For 20-micron bins and 1ns steps, DELTA = D / 4000
  For fixed D, DELTA goes as time_step_size/bin_size_squared
  */
  f *= 0.02/grid * 0.02/grid; // correct for grid size
  // f *= 0.5;                   // artifically reduce diffusion to 50%

  E_r = E_z = 100; // just to get started; will change later
  for (i=0; E_z > drift_E[i+1]; i++);
  ve_z = fq * (drift_offset[i] + drift_slope[i]*(E_z - drift_E[i]))/E_z;
  deltaez = grid * ve_z * f;
  for (i=0; E_r > drift_E[i+1]; i++);
  ve_r = fq * (drift_offset[i] + drift_slope[i]*(E_r - drift_E[i]))/E_r;
  deltaer = grid * ve_r * f;
  printf ("D_z, D_r values (q=%d) at 100 V/cm: %f %f\n", q, deltaez, deltaer);

  for (z=0; z<L; z++) {
  for (r=0; r<R; r++) {
  rho[1][z][r] = rho[2][z][r] = 0;
  }
  }
  /* NOTE that impurity and field arrays in setup start at (i,j)=(1,1) for (r,z)=(0,0) */
  int idid = lrint((setup->wrap_around_radius - setup->ditch_thickness)/grid) + 1; // ditch ID
  int idod = lrint(setup->wrap_around_radius/grid) + 1; // ditch OD
  int idd =  lrint(setup->ditch_depth/grid) + 1;        // ditch depth
  for (r=1; r<R; r++) {
    for (z=1; z<L-2; z++) {
      if (rho[0][z][r] < 1.0e-14) {
        rho[1][z][r] += rho[0][z][r];
        continue;
      }
      // calc E in r-direction
      if (r == 1) {  // r = 0; symmetry implies E_r = 0
        E_r = 0;
      } 
      else if (setup->point_type[z][r] == CONTACT_EDGE) {
        E_r = ((v[new_var][z][r] - v[new_var][z][r+1])*setup->dr[1][z][r] +
        (v[new_var][z][r-1] - v[new_var][z][r])*setup->dr[0][z][r]) / (0.2*grid);
      } 
      else if (setup->point_type[z][r] < INSIDE &&
          setup->point_type[z][r-1] == CONTACT_EDGE) {
        E_r =  (v[new_var][z][r-1] - v[new_var][z][r]) * setup->dr[1][z][r-1] / ( 0.1*grid) ;
      } 
      else if (setup->point_type[z][r] < INSIDE &&
          setup->point_type[z][r+1] == CONTACT_EDGE) {
        E_r =  (v[new_var][z][r] - v[new_var][z][r+1]) * setup->dr[0][z][r+1] / ( 0.1*grid) ;
      } 
      else if (r == R-1) {
        E_r = (v[new_var][z][r-1] - v[new_var][z][r])/(0.1*grid);
      } 
      else {
        E_r = (v[new_var][z][r-1] - v[new_var][z][r+1])/(0.2*grid);
      }
      // calc E in z-direction
      // enum point_types{PC, HVC, INSIDE, PASSIVE, PINCHOFF, DITCH, DITCH_EDGE, CONTACT_EDGE};
      if (setup->point_type[z][r] == CONTACT_EDGE) {
        E_z = ((v[new_var][z][r] - v[new_var][z+1][r])*setup->dz[1][z][r] +
        (v[new_var][z-1][r] - v[new_var][z][r])*setup->dz[0][z][r]) / (0.2*grid);
      } 
      else if (setup->point_type[z][r] < INSIDE &&
          setup->point_type[z-1][r] == CONTACT_EDGE) {
        E_z =  (v[new_var][z-1][r] - v[new_var][z][r]) * setup->dz[1][z-1][r] / ( 0.1*grid) ;
      } 
      else if (setup->point_type[z][r] < INSIDE &&
          setup->point_type[z+1][r] == CONTACT_EDGE) {
        E_z =  (v[new_var][z][r] - v[new_var][z+1][r]) * setup->dz[0][z+1][r] / ( 0.1*grid) ;
      } 
      else if (z == 1) {
        E_z = (v[new_var][z][r] - v[new_var][z+1][r])/(0.1*grid);
      } 
      else if (z == L-1) {
        E_z = (v[new_var][z-1][r] - v[new_var][z][r])/(0.1*grid);
      } 
      else {
        E_z = (v[new_var][z-1][r] - v[new_var][z+1][r])/(0.2*grid);
      }

      /* do diffusion to neighboring pixels */
      deltaez = deltaer = ve_z = ve_r = 0;

      // if(z==6 && r==1253){
      //   printf("In CPU at 1, at z=%d and r=%d, the value of rho is %.7f deltaez is %.7f and deltaer is %.7f\n", z, r, rho[1][z][r], deltaez, deltaer);
      // }

      E = fabs(E_z);
      if (E > 1.0) {
      for (i=0; E > drift_E[i+1]; i++);
      ve_z = (drift_offset[i] + drift_slope[i]*(E - drift_E[i]));
      deltaez = grid * ve_z * f / E;
      }

      E = fabs(E_r);
      if (E > 1.0) {
      for (i=0; E > drift_E[i+1]; i++);
      ve_r = (drift_offset[i] + drift_slope[i]*(E - drift_E[i]));
      deltaer = grid * ve_r * f / E;
      }
      if (0 && r == 100 && z == 10)
      printf("r z: %d %d; E_r deltaer: %f %f; E_z deltaez: %f %f; rho[0] = %f\n",
        r, z, E_r, deltaer, E_z, deltaez, rho[0][z][r]);

      /* reduce diffusion at passivated surfaces by a factor of surface_drift_vel_factor */
      if (1 &&
      ((r == idid && z < idd) ||
      (r < idid  && z == 1 ) ||
      (r >= idid && r <= idod && z == idd))) {
      // assume 2-micron-thick roughness/passivation in z
      deltaer *= setup->surface_drift_vel_factor;
      deltaez *= setup->surface_drift_vel_factor * grid/0.002; // * grid/0.002;
      }


      if((z==6||z==5||z==4) && (r==1253||r==1252||r==1254)){
        printf("In CPU at 1, at z=%d and r=%d, the value of rho is %.7f \n", z, r, rho[1][z][r]);
      }
      // enum point_types{PC, HVC, INSIDE, PASSIVE, PINCHOFF, DITCH, DITCH_EDGE, CONTACT_EDGE};
      rho[1][z][r]   += rho[0][z][r];
      if (0 && z == 1) printf("r,z = %d, %d E_r,z = %f, %f  deltaer,z = %f, %f  s1,s2 = %f, %f\n",
                  r, z, E_r, E_z, deltaer, deltaez, setup->s1[r], setup->s2[r]);

                  if((z==6||z==5||z==4) && (r==1253||r==1252||r==1254)){
                    printf("In CPU at 2, at z=%d and r=%d, the value of rho is %.7f \n", z, r, rho[1][z][r]);
      }

      if (r < R-1 && setup->point_type[z][r+1] != DITCH) {
      //if (setup->point_type[z][r+1] > HVC)
      rho[1][z][r+1] += rho[0][z][r]*deltaer * setup->s1[r] * (double) (r-1) / (double) (r);
      rho[1][z][r]   -= rho[0][z][r]*deltaer * setup->s1[r];
      }

      if((z==6||z==5||z==4) && (r==1253||r==1252||r==1254)){
        printf("In CPU at 3, at z=%d and r=%d, the value of rho is %.7f \n", z, r, rho[1][z][r]);
      }

      if (z > 1 && setup->point_type[z-1][r] != DITCH) {
      //if (setup->point_type[z-1][r] > HVC)
      rho[1][z-1][r] += rho[0][z][r]*deltaez;
      rho[1][z][r]   -= rho[0][z][r]*deltaez;
      }

      if((z==6||z==5||z==4) && (r==1253||r==1252||r==1254)){
        printf("In CPU at 4, at z=%d and r=%d, the value of rho is %.7f \n", z, r, rho[1][z][r]);
      }
      if (z < L-1 && setup->point_type[z+1][r] != DITCH) {
      //if (setup->point_type[z+1][r] > HVC)
      rho[1][z+1][r] += rho[0][z][r]*deltaez;
      rho[1][z][r]   -= rho[0][z][r]*deltaez;
      }

      if((z==6||z==5||z==4) && (r==1253||r==1252||r==1254)){
        printf("In CPU at 5, at z=%d and r=%d, the value of rho is %.7f \n", z, r, rho[1][z][r]);
      }
      if (r > 2 && setup->point_type[z][r-1] != DITCH) {
      //if (setup->point_type[z][r-1] > HVC)
      rho[1][z][r-1] += rho[0][z][r]*deltaer * setup->s2[r] * (double) (r-1) / (double) (r-2);
      rho[1][z][r]   -= rho[0][z][r]*deltaer * setup->s2[r];
      }

      if((z==6||z==5||z==4) && (r==1253||r==1252||r==1254)){
        printf("In CPU at 6, at z=%d and r=%d, the value of rho is %.7f \n", z, r, rho[1][z][r]);
      }

      if(rho[1][6][1253]-14.731561<0.000001){
        printf("The change happnds at z=%d and r=%d \n", z, r);
      }

      //-----------------------------------------------------------
    }
  }

  printf("In CPU at 7, at z=%d and r=%d, the value of rho is %.7f \n", 6, 1253, rho[1][6][1253]);

    

  for (r=1; r<R; r++) {
    for (z=1; z<L-2; z++) {
      if (rho[1][z][r] < 1.0e-14) {
        rho[2][z][r] += rho[1][z][r];
        continue;
      }
      // need to r-calculate all the fields
      // calc E in r-direction
      if (r == 1) {  // r = 0; symmetry implies E_r = 0
      E_r = 0;
      } 
      else if (setup->point_type[z][r] == CONTACT_EDGE) {
      E_r = ((v[new_var][z][r] - v[new_var][z][r+1])*setup->dr[1][z][r] +
        (v[new_var][z][r-1] - v[new_var][z][r])*setup->dr[0][z][r]) / (0.2*grid);
      } 
      else if (setup->point_type[z][r] < INSIDE &&
          setup->point_type[z][r-1] == CONTACT_EDGE) {
      E_r =  (v[new_var][z][r-1] - v[new_var][z][r]) * setup->dr[1][z][r-1] / ( 0.1*grid) ;
      } 
      else if (setup->point_type[z][r] < INSIDE &&
          setup->point_type[z][r+1] == CONTACT_EDGE) {
      E_r =  (v[new_var][z][r] - v[new_var][z][r+1]) * setup->dr[0][z][r+1] / ( 0.1*grid) ;
      } 
      else if (r == R-1) {
      E_r = (v[new_var][z][r-1] - v[new_var][z][r])/(0.1*grid);
      } 
      else {
      E_r = (v[new_var][z][r-1] - v[new_var][z][r+1])/(0.2*grid);
      }
      // calc E in z-direction
      // enum point_types{PC, HVC, INSIDE, PASSIVE, PINCHOFF, DITCH, DITCH_EDGE, CONTACT_EDGE};
      if (setup->point_type[z][r] == CONTACT_EDGE) {
      E_z = ((v[new_var][z][r] - v[new_var][z+1][r])*setup->dz[1][z][r] +
        (v[new_var][z-1][r] - v[new_var][z][r])*setup->dz[0][z][r]) / (0.2*grid);
      } 
      else if (setup->point_type[z][r] < INSIDE &&
          setup->point_type[z-1][r] == CONTACT_EDGE) {
      E_z =  (v[new_var][z-1][r] - v[new_var][z][r]) * setup->dz[1][z-1][r] / ( 0.1*grid) ;
      } 
      else if (setup->point_type[z][r] < INSIDE &&
          setup->point_type[z+1][r] == CONTACT_EDGE) {
      E_z =  (v[new_var][z][r] - v[new_var][z+1][r]) * setup->dz[0][z+1][r] / ( 0.1*grid) ;
      } 
      else if (z == 1) {
      E_z = (v[new_var][z][r] - v[new_var][z+1][r])/(0.1*grid);
      } 
      else if (z == L-1) {
      E_z = (v[new_var][z-1][r] - v[new_var][z][r])/(0.1*grid);
      } 
      else {
      E_z = (v[new_var][z-1][r] - v[new_var][z+1][r])/(0.2*grid);
      }
      ve_z = ve_r = 0;
      E = fabs(E_z);
      if (E > 1.0) {
      for (i=0; E > drift_E[i+1]; i++);
      ve_z = fq * (drift_offset[i] + drift_slope[i]*(E - drift_E[i]));
      }
      E = fabs(E_r);
      if (E > 1.0) {
      for (i=0; E > drift_E[i+1]; i++);
      ve_r = fq * (drift_offset[i] + drift_slope[i]*(E - drift_E[i]));
      }
      /* reduce drift speed at passivated surfaces by a factor of surface_drift_vel_factor */
      if (1 &&
      ((r == idid && z < idd) ||
      (r < idid  && z == 1 ) ||
      (r >= idid && r <= idod && z == idd))) {
      ve_r *= setup->surface_drift_vel_factor;
      ve_z *= setup->surface_drift_vel_factor * grid/0.002;  // assume 2-micron-thick roughness/passivation in z
      }


      //-----------------------------------------------------------

      /* do drift to neighboring pixels */
      // enum point_types{PC, HVC, INSIDE, PASSIVE, PINCHOFF, DITCH, DITCH_EDGE, CONTACT_EDGE};
      if (E_r > 0) {
      dre = -TSTEP*ve_r;
      } 
      else {
      dre =  TSTEP*ve_r;
      }
      if (E_z > 0) {
      dze = -TSTEP*ve_z;
      } 
      else {
      dze =  TSTEP*ve_z;
      }

      if (dre == 0.0) {
      i = r;
      fr = 1.0;
      } 
      else {
      i = (double) r + dre;
      fr = ceil(dre) - dre;
      }
      if (i<1) {
      i = 1;
      fr = 1.0;
      }
      if (i>R-1) {
      i = R-1;
      fr = 0.0;
      }
      if (dre > 0 && z < idd && r <= idid && i >= idid) { // ditch ID
      i = idid;
      fr = 1.0;
      }
      if (dre < 0 && z < idd && r >= idod && i <= idod) { // ditch OD
      i = idod;
      fr = 0.0;
      }

      if (dze == 0.0) {
      k = z;
      fz = 1.0;
      } 
      else {
      k = (double) z + dze;
      fz = ceil(dze) - dze;
      }
      if (k<1) {
      k = 1;
      fz = 1.0;
      }
      if (k>L-1) {
      k = L-1;
      fz = 0.0;
      }
      if (dze < 0 && r > idid && r < idod && k < idd) { // ditch depth
      k   = idd;
      fr  = 1.0;
      }
      if (0 && r == 100 && z == 10)
      printf("r z: %d %d; E_r i dre: %f %d %f; fr = %f\n"
        "r z: %d %d; E_z k dze: %f %d %f; fz = %f\n",
        r, z, E_r, i, dre, fr, r, z, E_z, k, dze, fz);

      if (i>=1 && i<R && k>=1 && k<L) {
        if (i > 1 && r > 1) {
          rho[2][k  ][i  ] += rho[1][z][r] * fr      *fz       * (double) (r-1) / (double) (i-1);
          rho[2][k  ][i+1] += rho[1][z][r] * (1.0-fr)*fz       * (double) (r-1) / (double) (i);
          rho[2][k+1][i  ] += rho[1][z][r] * fr      *(1.0-fz) * (double) (r-1) / (double) (i-1);
          rho[2][k+1][i+1] += rho[1][z][r] * (1.0-fr)*(1.0-fz) * (double) (r-1) / (double) (i);
        } 
        else if (i > 1) {  // r == 0
          rho[2][k  ][i  ] += rho[1][z][r] * fr      *fz       / (double) (8*i-8);
          rho[2][k  ][i+1] += rho[1][z][r] * (1.0-fr)*fz       / (double) (8*i);
          rho[2][k+1][i  ] += rho[1][z][r] * fr      *(1.0-fz) / (double) (8*i-8);
          rho[2][k+1][i+1] += rho[1][z][r] * (1.0-fr)*(1.0-fz) / (double) (8*i);
        } 
        else if (r > 1) {  // i == 0
          rho[2][k  ][i  ] += rho[1][z][r] * fr      *fz       * (double) (8*(R+1)-8);
          rho[2][k  ][i+1] += rho[1][z][r] * (1.0-fr)*fz       * (double) (r-1);
          rho[2][k+1][i  ] += rho[1][z][r] * fr      *(1.0-fz) * (double) (8*(R+1)-8);
          rho[2][k+1][i+1] += rho[1][z][r] * (1.0-fr)*(1.0-fz) * (double) (r-1);
        } 
        else {             // r == i == 0
          rho[2][k  ][i  ] += rho[1][z][r] * fr      *fz;
          rho[2][k  ][i+1] += rho[1][z][r] * (1.0-fr)*fz       / 8.0; // vol_0 / vol_1 = 1/8
          rho[2][k+1][i  ] += rho[1][z][r] * fr      *(1.0-fz);
          rho[2][k+1][i+1] += rho[1][z][r] * (1.0-fr)*(1.0-fz) / 8.0;
        }
      }
    }
  }

  for (z=0; z<L; z++) {
    for (r=0; r<R; r++) {
      if (setup->point_type[z][r] <= HVC) {
        *gone += rho[2][z][r] * r;
        rho[2][z][r] = 0;
      }
  }
}


return 0;
}