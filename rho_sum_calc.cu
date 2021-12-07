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

extern "C" float get_signal_gpu(GPU_data *gpu_setup, int L, int R, int n, int num_blocks, int num_threads);

__global__ void cal_esum1(int L, int R, float *rho_sum, float *rho_e, float *w_pot, int max_threads){
    // esum1 += rho_e[0][z][r] * (double) (r-1) * setup.wpot[r-1][z-1];
    int r = blockIdx.x%R;
    int z = (floorf(blockIdx.x/R) * max_threads) + threadIdx.x;
    
    if(r==0 || z==0 || r>=R || z>=L){
      return;
    }
  rho_sum[((R+1)*z)+r] = rho_e[(0*(L+1)*(R+1))+((R+1)*z)+r] * (float) (r-1) * w_pot[((R+1)*(z-1))+(r-1)];
}

__global__ void cal_esum2(int L, int R, float *rho_sum, float *rho_e, int max_threads){
  // esum2 += rho_e[0][z][r] * (float) (r-1);
  int r = blockIdx.x%R;
  int z = (floorf(blockIdx.x/R) * max_threads) + threadIdx.x;
  
  if(r==0 || z==0 || r>=R || z>=L){
    return;
  }
  rho_sum[((R+1)*z)+r] = rho_e[(0*(L+1)*(R+1))+((R+1)*z)+r] * (float) (r-1);
}

__global__ void cal_hsum1(int L, int R, float *rho_sum, float *rho_h, float *w_pot, int max_threads){
  // hsum1 += rho_h[0][z][r] * (float) (r-1) * setup.wpot[r-1][z-1];
  int r = blockIdx.x%R;
  int z = (floorf(blockIdx.x/R) * max_threads) + threadIdx.x;
  
  if(r==0 || z==0 || r>=R || z>=L){
    return;
  }
  rho_sum[((R+1)*z)+r] = rho_h[(0*(L+1)*(R+1))+((R+1)*z)+r] * (float) (r-1) * w_pot[((R+1)*(z-1))+(r-1)];
}

__global__ void cal_hsum2(int L, int R, float *rho_sum, float *rho_h, int max_threads){
  // hsum2 += rho_h[0][z][r] * (float) (r-1);
  int r = blockIdx.x%R;
  int z = (floorf(blockIdx.x/R) * max_threads) + threadIdx.x;
  
  if(r==0 || z==0 || r>=R || z>=L){
    return;
  }
  rho_sum[((R+1)*z)+r] = rho_h[(0*(L+1)*(R+1))+((R+1)*z)+r] * (float) (r-1);
}

extern "C" float get_signal_gpu(GPU_data *gpu_setup, int L, int R, int n, int num_blocks, int num_threads){

    double signal, hsum1=0.f, hsum2=0.f, esum1=0.f, esum2=0.f;
    // esum1 += rho_e[0][z][r] * (float) (r-1) * setup.wpot[r-1][z-1];
    // esum2 += rho_e[0][z][r] * (float) (r-1);
    // hsum1 += rho_h[0][z][r] * (float) (r-1) * setup.wpot[r-1][z-1];
    // hsum2 += rho_h[0][z][r] * (float) (r-1);

    cal_esum1<<<num_blocks, num_threads>>>(L, R, gpu_setup->rho_sum, gpu_setup->rho_e_gpu, gpu_setup->wpot_gpu, num_threads);
    esum1 = thrust::reduce(thrust::device, gpu_setup->rho_sum, gpu_setup->rho_sum + (L+1)*(R+1));
    cal_esum2<<<num_blocks, num_threads>>>(L, R, gpu_setup->rho_sum, gpu_setup->rho_e_gpu, num_threads);
    esum2 = thrust::reduce(thrust::device, gpu_setup->rho_sum, gpu_setup->rho_sum + (L+1)*(R+1));
    cal_hsum1<<<num_blocks, num_threads>>>(L, R, gpu_setup->rho_sum, gpu_setup->rho_h_gpu, gpu_setup->wpot_gpu, num_threads);
    hsum1 = thrust::reduce(thrust::device, gpu_setup->rho_sum, gpu_setup->rho_sum + (L+1)*(R+1));
    cal_hsum2<<<num_blocks, num_threads>>>(L, R, gpu_setup->rho_sum, gpu_setup->rho_h_gpu, num_threads);
    hsum2 = thrust::reduce(thrust::device, gpu_setup->rho_sum, gpu_setup->rho_sum + (L+1)*(R+1));

    if (n > 10 && gpu_setup->hsum02 > hsum2) hsum1 += gpu_setup->hsum02 - hsum2;

    if (n==10) {
        gpu_setup->esum01 = esum1; gpu_setup->esum02 = esum2;
        gpu_setup->hsum01 = hsum1; gpu_setup->hsum02 = hsum2;
      }
    
    printf("esum1=%.5f, esum2=%.5f, hsum1=%.5f, hsum2 =%.5f\n", esum1, esum2, hsum1, hsum2);
    signal = 1000.0 * ((hsum1 - gpu_setup->hsum01) / gpu_setup->hsum02 - (esum1 - gpu_setup->esum01) / gpu_setup->esum02);
    printf("Signals collected:%.5f\n", signal/1000);
    return signal;
  }