/*
  Allocated memory for pointers on GPU memory that are stored in GPU struct. The pointer are then assigned values from CPU memory.
  author:           Kevin H Bhimani
  first written:    Nov 2021

  For better memory management on GPU, all multidemensional arrays are flattened.
  The conversion for flattening array[i][j][k] = flat_array[(i*(L+1)*(R+1))+((R+1)*j)+k]
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

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

#include <time.h>

#include "gpu_vars.h"

extern "C" void gpu_init(MJD_Siggen_Setup *setup, float ***rho_e, float ***rho_h, GPU_data *gpu_setup);
extern "C" void get_densities(int L, int R, float ***rho_e, float ***rho_h, GPU_data *gpu_setup);
extern "C" void get_potential(int L, int R, MJD_Siggen_Setup *setup, GPU_data *gpu_setup);
extern "C" void free_gpu_mem(GPU_data *gpu_setup);


extern "C" void gpu_init(MJD_Siggen_Setup *setup, float ***rho_e, float ***rho_h, GPU_data *gpu_setup){
/*
Below we allocate and copy values to GPU. For better memory management and indexing, all multidemensional arrays are flattened.
The conversion for flattening array[i][j][k] = flat_array[(i*(L+1)*(R+1))+((R+1)*j)+k]
*/
  float  grid = setup->xtal_grid;
  int    L  = lrint(setup->xtal_length/grid)+2;
  int    R  = lrint(setup->xtal_radius/grid)+2;

  double *v_flat;
  v_flat = (double*)malloc(2*sizeof(double)*(L+1)*(R+1));
  cudaMalloc((void**)&gpu_setup->v_gpu, 2*sizeof(double)*(L+1)*(R+1));
  for(int i=0; i<2; i++) {
    for(int j=0; j<=L; j++){
      for(int k=0; k<=R; k++){
        v_flat[(i*(L+1)*(R+1))+((R+1)*j)+k] = setup->v[i][j][k];
      }
    }
  }
  cudaMemcpy(gpu_setup->v_gpu, v_flat, 2*sizeof(double)*(L+1)*(R+1), cudaMemcpyHostToDevice);
  
  char *point_type_flat;
  double *impurity_flat;

  point_type_flat = (char*)malloc(sizeof(char)*(L+1)*(R+1));
  impurity_flat = (double*)malloc(sizeof(double)*(L+1)*(R+1));

  cudaMalloc((void**)&gpu_setup->point_type_gpu, sizeof(char)*(L+1)*(R+1));
  cudaMalloc((void**)&gpu_setup->impurity_gpu, sizeof(double)*(L+1)*(R+1));

  for(int j=0; j<=L; j++){
      for(int k=0; k<=R; k++){
        point_type_flat[((R+1)*j)+k] = setup->point_type[j][k];
        impurity_flat[((R+1)*j)+k] = setup->impurity[j][k];
      }
    }
    
  cudaMemcpy(gpu_setup->point_type_gpu, point_type_flat, sizeof(char)*(L+1)*(R+1), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_setup->impurity_gpu, impurity_flat, sizeof(double)*(L+1)*(R+1), cudaMemcpyHostToDevice);
  
  double *dr_flat, *dz_flat;
  dr_flat = (double*)malloc(2*sizeof(double)*(L+1)*(R+1));
  dz_flat = (double*)malloc(2*sizeof(double)*(L+1)*(R+1));

  cudaMalloc((void**)&gpu_setup->dr_gpu, 2*sizeof(double)*(L+1)*(R+1));
  cudaMalloc((void**)&gpu_setup->dz_gpu, 2*sizeof(double)*(L+1)*(R+1));

    for(int i=0; i<2; i++) {
      for(int j=1; j<=L; j++){
          for(int k=0; k<=R; k++){
            dr_flat[(i*(L+1)*(R+1))+((R+1)*j)+k] = setup->dr[i][j][k];
            dz_flat[(i*(L+1)*(R+1))+((R+1)*j)+k] = setup->dz[i][j][k];
          }
        }
      }
  cudaMemcpy(gpu_setup->dr_gpu, dr_flat, 2*sizeof(double)*(L+1)*(R+1), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_setup->dz_gpu, dz_flat, 2*sizeof(double)*(L+1)*(R+1), cudaMemcpyHostToDevice);

  
  cudaMalloc((void**)&gpu_setup->s1_gpu, sizeof(double)*(R+1));
  cudaMemcpy(gpu_setup->s1_gpu, setup->s1, sizeof(double)*(R+1), cudaMemcpyHostToDevice);
  
  
  cudaMalloc((void**)&gpu_setup->s2_gpu, sizeof(double)*(R+1));
  cudaMemcpy(gpu_setup->s2_gpu, setup->s2, sizeof(double)*(R+1), cudaMemcpyHostToDevice);
  
  double *eps_dr_flat, *eps_dz_flat;
  eps_dr_flat = (double*)malloc(sizeof(double)*(L+1)*(R+1));
  eps_dz_flat = (double*)malloc(sizeof(double)*(L+1)*(R+1));
  cudaMalloc((void**)&gpu_setup->eps_dr_gpu, sizeof(double)*(L+1)*(R+1));
  cudaMalloc((void**)&gpu_setup->eps_dz_gpu, sizeof(double)*(L+1)*(R+1));
  for(int j=0; j<=L; j++){
    for(int k=0; k<=R; k++){
      eps_dr_flat[((R+1)*j)+k] = setup->eps_dr[j][k];
      eps_dz_flat[((R+1)*j)+k] = setup->eps_dz[j][k];
    }
  }
  cudaMemcpy(gpu_setup->eps_dr_gpu, eps_dr_flat, sizeof(double)*(L+1)*(R+1), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_setup->eps_dz_gpu, eps_dz_flat, sizeof(double)*(L+1)*(R+1), cudaMemcpyHostToDevice);

    
  double *diff_array_cpu;
  diff_array_cpu = (double*)malloc(sizeof(double)*(L+1)*(R+1));
  cudaMalloc((void**)&gpu_setup->diff_array, sizeof(double)*(L+1)*(R+1));

  for (int i = 0; i<(L+1)*(R+1); i++){
    diff_array_cpu[i] =0.00;
  }
  cudaMemcpy(gpu_setup->diff_array, diff_array_cpu, sizeof(double)*(L+1)*(R+1), cudaMemcpyHostToDevice);

  cudaMalloc((void**)&gpu_setup->deltaez_array, sizeof(double)*(L+1)*(R+1));
  cudaMalloc((void**)&gpu_setup->deltaer_array, sizeof(double)*(L+1)*(R+1));

  cudaMalloc((void**)&gpu_setup->fr_array, sizeof(double)*(L+1)*(R+1));
  cudaMalloc((void**)&gpu_setup->fz_array, sizeof(double)*(L+1)*(R+1));
  cudaMalloc((void**)&gpu_setup->i_array, sizeof(int)*(L+1)*(R+1));
  cudaMalloc((void**)&gpu_setup->k_array, sizeof(int)*(L+1)*(R+1));

  float *rho_flat_e_cpu;
  rho_flat_e_cpu = (float*)malloc(4*sizeof(float)*(L+1)*(R+1));
  cudaMalloc((void**)&gpu_setup->rho_e_gpu, 4*sizeof(float)*(L+1)*(R+1));
  for(int i=0; i<4; i++) {
      for(int j=0; j<L; j++){
          for(int k=0; k<R; k++){
              rho_flat_e_cpu[(i*(L+1)*(R+1))+((R+1)*j)+k] = rho_e[i][j][k];
          }
      }
  }
  cudaMemcpy(gpu_setup->rho_e_gpu, rho_flat_e_cpu, 4*sizeof(float)*(L+1)*(R+1), cudaMemcpyHostToDevice);

  float *rho_flat_h_cpu;
  rho_flat_h_cpu = (float*)malloc(3*sizeof(float)*(L+1)*(R+1));
  cudaMalloc((void**)&gpu_setup->rho_h_gpu, 3*sizeof(float)*(L+1)*(R+1));
  for(int i=0; i<3; i++) {
      for(int j=0; j<L; j++){
          for(int k=0; k<R; k++){
              rho_flat_h_cpu[(i*(L+1)*(R+1))+((R+1)*j)+k]=rho_h[i][j][k];
          }
      }
  }
  cudaMemcpy(gpu_setup->rho_h_gpu, rho_flat_h_cpu, 3*sizeof(float)*(L+1)*(R+1), cudaMemcpyHostToDevice);


  float drift_E[20] = {0.000,  100.,  160.,  240.,  300.,  500.,  600., 750.0, 1000., 1250., 1500., 1750., 2000., 2500., 3000., 3500., 4000., 4500., 5000., 1e10};

  float drift_offset_e[20]={0.0,   0.027, 0.038, 0.049 ,0.055, 0.074, 0.081,
      0.089, 0.101, 0.109, 0.116, 0.119, 0.122, 0.125,
      0.1275,0.1283,0.1288,0.1291,0.1293,0.1293};
  float drift_slope_e[20];

  float drift_offset_h[20]={0.0,   0.036, 0.047, 0.056, 0.06,  0.072, 0.077,
      0.081, 0.086, 0.089, 0.0925,0.095, 0.097, 0.1,
      0.1025,0.1036,0.1041,0.1045,0.1047,0.1047};
  float drift_slope_h[20];
  int i;
  for (i=0; i<20; i++) {
      drift_offset_e[i] /= grid;   // drift velocities in units of grid length
      drift_offset_h[i] /= grid;
  }
  for (i=0; i<19; i++) {
      drift_slope_e[i] = (drift_offset_e[i+1] - drift_offset_e[i]) /(drift_E[i+1] - drift_E[i]);
      drift_slope_h[i] = (drift_offset_h[i+1] - drift_offset_h[i]) /(drift_E[i+1] - drift_E[i]);
  }

  cudaMalloc((void**)&gpu_setup->drift_offset_e_gpu, sizeof(float)*20);
  cudaMemcpy(gpu_setup->drift_offset_e_gpu, drift_offset_e, sizeof(float)*20, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&gpu_setup->drift_offset_h_gpu, sizeof(float)*20);
  cudaMemcpy(gpu_setup->drift_offset_h_gpu, drift_offset_h, sizeof(float)*20, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&gpu_setup->drift_slope_e_gpu, sizeof(float)*20);
  cudaMemcpy(gpu_setup->drift_slope_e_gpu, drift_slope_e, sizeof(float)*20, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&gpu_setup->drift_slope_h_gpu, sizeof(float)*20);
  cudaMemcpy(gpu_setup->drift_slope_h_gpu, drift_slope_h, sizeof(float)*20, cudaMemcpyHostToDevice);

  
  cudaMalloc((void**)&gpu_setup->rho_sum, sizeof(double)*(L+1)*(R+1));
  cudaMalloc((void**)&gpu_setup->wpot_gpu, sizeof(float)*(L+1)*(R+1));

  float *rho_sum_flat = (float*)malloc(sizeof(double)*(L+1)*(R+1));
  float *wpot_flat = (float*)malloc(sizeof(float)*(L+1)*(R+1));

  for (int i = 0; i<(L+1)*(R+1); i++){
    rho_sum_flat[i] = 0.00;
  }

  for(int j=0; j<L; j++){
      for(int k=0; k<R; k++){
        wpot_flat[((R+1)*j)+k] = setup->wpot[k][j];
      }
    }
  cudaMemcpy(gpu_setup->rho_sum, rho_sum_flat, sizeof(double)*(L+1)*(R+1), cudaMemcpyHostToDevice); 
  cudaMemcpy(gpu_setup->wpot_gpu, wpot_flat, sizeof(float)*(L+1)*(R+1), cudaMemcpyHostToDevice); 

  free(rho_flat_e_cpu);
  free(rho_flat_h_cpu);
  free(v_flat);
  free(point_type_flat);
  free(dr_flat);
  free(dz_flat);
  free(eps_dr_flat);
  free(eps_dz_flat);
  free(impurity_flat);
  free(diff_array_cpu);
  free(wpot_flat);
  free(rho_sum_flat);
}

// Copies the densities back to CPU from GPU
extern "C" void get_densities(int L, int R, float ***rho_e, float ***rho_h, GPU_data *gpu_setup){

  float *rho_flat_e_cpu;
  rho_flat_e_cpu = (float*)malloc(4*sizeof(float)*(L+1)*(R+1));

  float *rho_flat_h_cpu;
  rho_flat_h_cpu = (float*)malloc(3*sizeof(float)*(L+1)*(R+1));

  cudaMemcpy(rho_flat_e_cpu, gpu_setup->rho_e_gpu, 4*sizeof(float)*(L+1)*(R+1), cudaMemcpyDeviceToHost);
  cudaMemcpy(rho_flat_h_cpu, gpu_setup->rho_h_gpu, 3*sizeof(float)*(L+1)*(R+1), cudaMemcpyDeviceToHost);

  for(int i=0; i<4; i++) {
    for(int j=0; j<L; j++){
        for(int k=0; k<R; k++){
          memcpy(&rho_e[i][j][k], &rho_flat_e_cpu[(i*(L+1)*(R+1))+((R+1)*j)+k], sizeof(float));
        }
    }
  }


  for(int i=0; i<3; i++) {
    for(int j=0; j<L; j++){
        for(int k=0; k<R; k++){
              memcpy(&rho_h[i][j][k], &rho_flat_h_cpu[(i*(L+1)*(R+1))+((R+1)*j)+k], sizeof(float));
          }
      }
  }
  free(rho_flat_e_cpu);
  free(rho_flat_h_cpu);
}

// Copies the potentials back to CPU from GPU
extern "C" void get_potential(int L, int R, MJD_Siggen_Setup *setup, GPU_data *gpu_setup){

  double *v_flat;
  v_flat = (double*)malloc(2*sizeof(double)*(L+1)*(R+1));
  cudaMemcpy(v_flat, gpu_setup->v_gpu, 2*sizeof(double)*(L+1)*(R+1), cudaMemcpyDeviceToHost);

  for(int i=0; i<2; i++) {
    for(int j=0; j<=L; j++){
      for(int k=0; k<=R; k++){
        memcpy(&setup->v[i][j][k], &v_flat[(i*(L+1)*(R+1))+((R+1)*j)+k], sizeof(double));
      }
    }
  }
  free(v_flat);
}

//Frees all pointer on GPU memory
extern "C" void free_gpu_mem(GPU_data *gpu_setup){
  cudaFree(gpu_setup->v_gpu);
  cudaFree(gpu_setup->point_type_gpu);
  cudaFree(gpu_setup->dr_gpu);
  cudaFree(gpu_setup->dz_gpu);
  cudaFree(gpu_setup->eps_dr_gpu);
  cudaFree(gpu_setup->eps_dz_gpu);
  cudaFree(gpu_setup->s1_gpu);
  cudaFree(gpu_setup->s2_gpu);
  cudaFree(gpu_setup->impurity_gpu);
  cudaFree(gpu_setup->diff_array);
  cudaFree(gpu_setup->rho_e_gpu);
  cudaFree(gpu_setup->rho_h_gpu);
  cudaFree(gpu_setup->drift_offset_e_gpu);
  cudaFree(gpu_setup->drift_offset_h_gpu);
  cudaFree(gpu_setup->drift_slope_e_gpu);
  cudaFree(gpu_setup->drift_slope_h_gpu);
  cudaFree(gpu_setup->deltaez_array);
  cudaFree(gpu_setup->deltaer_array);
  cudaFree(gpu_setup->fr_array);
  cudaFree(gpu_setup->fz_array);
  cudaFree(gpu_setup->i_array);
  cudaFree(gpu_setup->k_array);
  cudaFree(gpu_setup->wpot_gpu);
  cudaFree(gpu_setup->rho_sum);
}


