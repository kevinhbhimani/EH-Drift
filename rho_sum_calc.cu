/*
    Performs sum of charge densities which is needed to calculate signal collected
    author:           Kevin H Bhimani
    first written:    Dec 2021
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

__managed__ double grid_sig;
__managed__ double passivated_surface_thickness_sig;
__managed__ double dz_correction;



extern "C" double get_signal_gpu(MJD_Siggen_Setup *setup, GPU_data *gpu_setup, int L, int R, int n_iter, double grid, int save_time, int num_threads);

__global__ void cal_esum1(int L, int R, double *rho_sum, double *rho_e, double *surface_rho_e, double *w_pot, int max_threads){
    // esum1 += rho_e[0][z][r] * (double) (r-1) * setup.wpot[r-1][z-1];
    int r = blockIdx.x%R;
    int z = (floorf(blockIdx.x/R) * max_threads) + threadIdx.x;
    
    if(r==0 || z==0 || r>=R || z>=L){
      return;
    }

    if(z==1){
      rho_sum[((R+1)*z)+r] = rho_e[(0*(L+1)*(R+1))+((R+1)*z)+r] * (double) (r-1) * w_pot[((R+1)*(z-1))+(r-1)] ;
      rho_sum[((R+1)*z)+r] += surface_rho_e[r] * (double) (r-1) * w_pot[((R+1)*(z-1))+(r-1)]*dz_correction;
    }
    else{
      rho_sum[((R+1)*z)+r] = rho_e[(0*(L+1)*(R+1))+((R+1)*z)+r] * (double) (r-1) * w_pot[((R+1)*(z-1))+(r-1)];
    }
  }


__global__ void cal_esum2(int L, int R, double *rho_sum, double *rho_e, double *surface_rho_e, int max_threads){
  int r = blockIdx.x%R;
  int z = (floorf(blockIdx.x/R) * max_threads) + threadIdx.x;
  
  if(r==0 || z==0 || r>=R || z>=L){
    return;
  }
  if(z==1){
    rho_sum[((R+1)*z)+r] = rho_e[(0*(L+1)*(R+1))+((R+1)*z)+r] * (double) (r-1);
    rho_sum[((R+1)*z)+r] += surface_rho_e[r] * (double) (r-1);
  }
  else{
    rho_sum[((R+1)*z)+r] = rho_e[(0*(L+1)*(R+1))+((R+1)*z)+r] * (double) (r-1);
  }
}
__global__ void cal_hsum1(int L, int R, double *rho_sum, double *rho_h, double *surface_rho_h, double *w_pot, int max_threads){
  int r = blockIdx.x%R;
  int z = (floorf(blockIdx.x/R) * max_threads) + threadIdx.x;
  
  if(r==0 || z==0 || r>=R || z>=L){
    return;
  }

  if(z==1){
    rho_sum[((R+1)*z)+r] = rho_h[(0*(L+1)*(R+1))+((R+1)*z)+r] * (double) (r-1) * w_pot[((R+1)*(z-1))+(r-1)];
    rho_sum[((R+1)*z)+r] += surface_rho_h[r] * (double) (r-1) * w_pot[((R+1)*(z-1))+(r-1)]*dz_correction;
  }
  else{
    rho_sum[((R+1)*z)+r] = rho_h[(0*(L+1)*(R+1))+((R+1)*z)+r] * (double) (r-1) * w_pot[((R+1)*(z-1))+(r-1)];
  }
}

__global__ void cal_hsum2(int L, int R, double *rho_sum, double *rho_h, double *surface_rho_h, int max_threads){
  int r = blockIdx.x%R;
  int z = (floorf(blockIdx.x/R) * max_threads) + threadIdx.x;
  
  if(r==0 || z==0 || r>=R || z>=L){
    return;
  }
  if(z==1){
    rho_sum[((R+1)*z)+r] = rho_h[(0*(L+1)*(R+1))+((R+1)*z)+r] * (double) (r-1);
    rho_sum[((R+1)*z)+r] += surface_rho_h[r] * (double) (r-1);
  }
  else{
    rho_sum[((R+1)*z)+r] = rho_h[(0*(L+1)*(R+1))+((R+1)*z)+r] * (double) (r-1);

  }
}

__global__ void clear_courant(double *courant_array, int L, int R, int max_threads){
  int r = blockIdx.x%R;
  int z = (floorf(blockIdx.x/R) * max_threads) + threadIdx.x;
  
  if(r==0 || z==0 || r>=R || z>=L){
    return;
  }
  courant_array[((R+1)*z)+r]=0;
}

__global__ void print_sorted(double *courant_array, int L, int R){
  int count=0;
  for(int j=0; j<(L+1)*(R+1); j++){
    if(courant_array[j]>0){
      printf("%.5f\n",courant_array[j]);
      count++;
    }
  }
  printf("Number of points with non-zero courant number=%d\n", count);
}
extern "C" double get_signal_gpu(MJD_Siggen_Setup *setup, GPU_data *gpu_setup, int L, int R, int n_iter, double grid, int save_time, int num_threads){

  grid_sig=grid;
  passivated_surface_thickness_sig= setup->passivated_thickness;
  dz_correction=passivated_surface_thickness_sig/grid;

  int num_blocks = R * (ceil(L/num_threads)+1);
  double signal=0.f, hsum1=0.f, hsum2=0.f, esum1=0.f, esum2=0.f;


  cal_esum1<<<num_blocks, num_threads>>>(L, R, gpu_setup->rho_sum, gpu_setup->rho_e_gpu, gpu_setup->surface_rho_e, gpu_setup->wpot_gpu, num_threads);

  esum1 = thrust::reduce(thrust::device, gpu_setup->rho_sum, gpu_setup->rho_sum + (L+1)*(R+1));

  cal_esum2<<<num_blocks, num_threads>>>(L, R, gpu_setup->rho_sum, gpu_setup->rho_e_gpu, gpu_setup->surface_rho_e, num_threads);
  esum2 = thrust::reduce(thrust::device, gpu_setup->rho_sum, gpu_setup->rho_sum + (L+1)*(R+1));

  cal_hsum1<<<num_blocks, num_threads>>>(L, R, gpu_setup->rho_sum, gpu_setup->rho_h_gpu, gpu_setup->surface_rho_h, gpu_setup->wpot_gpu, num_threads);
  hsum1 = thrust::reduce(thrust::device, gpu_setup->rho_sum, gpu_setup->rho_sum + (L+1)*(R+1));


  cal_hsum2<<<num_blocks, num_threads>>>(L, R, gpu_setup->rho_sum, gpu_setup->rho_h_gpu, gpu_setup->surface_rho_h, num_threads);
  hsum2 = thrust::reduce(thrust::device, gpu_setup->rho_sum, gpu_setup->rho_sum + (L+1)*(R+1));
  
  if (n_iter > save_time && gpu_setup->hsum02 > hsum2) hsum1 += gpu_setup->hsum02 - hsum2;

  if (n_iter==save_time) {
      gpu_setup->esum01 = esum1; gpu_setup->esum02 = esum2;
      gpu_setup->hsum01 = hsum1; gpu_setup->hsum02 = hsum2;
    }


  //printf("hsum1=%.5f-hsum01=%.5f/hsum2=%.5f - esum1=%.5f-esum01=%.5f/esum2=%.5f,\n", hsum1, gpu_setup->hsum01, hsum2, esum1, gpu_setup->esum01, esum2);
  signal = 1000.0 * ((hsum1 - gpu_setup->hsum01) / gpu_setup->hsum02 - (esum1 - gpu_setup->esum01) / gpu_setup->esum02);
  //printf("Signals collected:%.5f\n", signal/1000);

  return signal;
}
  extern "C" double get_courant_number(GPU_data *gpu_setup, int L, int R, int num_blocks, int num_threads){

    // double courant_number;
    double courant_number = thrust::max_element(thrust::device_pointer_cast(gpu_setup->courant_array), thrust::device_pointer_cast(gpu_setup->courant_array) + (L+1)*(R+1))[0];
    // thrust::sort(thrust::device_pointer_cast(gpu_setup->courant_array), thrust::device_pointer_cast(gpu_setup->courant_array) + (L+1)*(R+1));
    // cudaDeviceSynchronize();
    // print_sorted<<<1, 1>>>(gpu_setup->courant_array, L, R);
    // cudaMemcpy(&courant_number, gpu_setup->courant_array+ (L+1)*(R+1)-1, sizeof(double), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    // printf("\nCourant number is %f\n", courant_number);

    // gpu_setup->courant_number_n=courant_number;
    clear_courant<<<num_blocks, num_threads>>>(gpu_setup->courant_array, L, R, num_threads);
    cudaDeviceSynchronize();
    return courant_number;
  }