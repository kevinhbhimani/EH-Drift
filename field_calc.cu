/*
    This program is called to calculate electric field in each tme step of the signal.
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

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include "gpu_vars.h"

extern __global__ void vacuum_gap_calc(double *impurity_gpu,double *v_gpu, int L, int R, float grid, int old_gpu_relax, int new_gpu_relax, float vacuum_gap_gpu, double e_over_E_gpu);
extern __global__ void z_relection_set(double* v_gpu, int L, int R, int old_gpu_relax, int new_gpu_relax, double *eps_dr_gpu, double *eps_dz_gpu, char *point_type_gpu);
extern __global__ void reflection_symmetry_set(double *v_gpu, int L, int R, int old_gpu_relax, int new_gpu_relax);
extern __global__ void relax_step(int is_red, int ev_calc, int L, int R, float grid, double OR_fact, double *v_gpu, char *point_type_gpu, double *dr_gpu, double *dz_gpu, 
              double *eps_dr_gpu, double *eps_dz_gpu, double *s1_gpu, double *s2_gpu, double *impurity_gpu, double *diff_array, int old_gpu_relax, int new_gpu_relax, int max_threads, float vacuum_gap_gpu);



extern "C" int ev_calc_gpu(int ev_calc, MJD_Siggen_Setup *setup, GPU_data *gpu_setup);

__global__ void set_bound_v(int L, int R, double *v_gpu, float xtal_HV, char *point_type, int max_threads){

    int r = blockIdx.x%R;
    int z = (floorf(blockIdx.x/R) * max_threads) + threadIdx.x;
    
    if(r==0 || z==0 || r>=R+1 || z>=L+1){
        return;
    }
    if(point_type[((R+1)*z)+r] == HVC){
        v_gpu[(0*(L+1)*(R+1))+((R+1)*z)+r]= v_gpu[((1*(L+1)*(R+1))+((R+1)*z)+r)] = xtal_HV;
    }
    if(point_type[((R+1)*z)+r] == PC){
        v_gpu[((0*(L+1)*(R+1))+((R+1)*z)+r)] = v_gpu[((1*(L+1)*(R+1))+((R+1)*z)+r)] = 0.0;
    }
}

__global__ void set_passivated_imp(int R, double *impurity_gpu){
    int r = blockIdx.x+1;
    if(r>=R){
        return;
    }
    impurity_gpu[((R+1)*0)+r] = impurity_gpu[((R+1)*1)+r];
}

__global__ void set_passivated_wp_calc(int L, int R, double *impurity_gpu, int max_threads){
    int r = blockIdx.x%R;
    int z = (floorf(blockIdx.x/R) * max_threads) + threadIdx.x;
    if(r==0 || r>=R || z>=L){
        return;
    }
    impurity_gpu[((R+1)*z)+r] = 0;
}

__global__ void restore_passivated_imp(int R, double *impurity_gpu){
    int r = blockIdx.x+1;
    if(r>=R){
        return;
    }
    impurity_gpu[((R+1)*1)+r] = impurity_gpu[((R+1)*0)+r];
}

/* 
    Performs relaxation on GPUs using Red- Black SOR method to calculate the new electric potentials
 */

extern "C" int ev_calc_gpu(int ev_calc, MJD_Siggen_Setup *setup, GPU_data *gpu_setup){
    int    iter;
    float  grid = setup->xtal_grid;
    int    L  = lrint(setup->xtal_length/grid)+2;
    int    R  = lrint(setup->xtal_radius/grid)+2;
    int  old_gpu_relax = 1, new_gpu_relax = 0;

    double e_over_E_gpu = 11.310; // e/epsilon; for 1 mm2, charge units 1e10 e/cm3, espilon = 16*epsilon0
    float vacuum_gap_gpu = setup->vacuum_gap;
  
    setup->fully_depleted = 1;
    setup->bubble_volts = 0;

    int num_threads = 300;
    int num_blocks = R * (ceil(L/num_threads)+1); //The +1 is just a precaution to make sure all R and Z values are included

    if(num_blocks+1<65535){
        set_bound_v<<<num_blocks+1,num_threads+1>>>(L, R, gpu_setup->v_gpu, setup->xtal_HV, gpu_setup->point_type_gpu, num_threads);
    }
    else{
        printf("----------------Pick a smaller block please----------------\n");
        return 0;
    }
    if (ev_calc) {
        set_passivated_imp<<<R,1>>>(R, gpu_setup->impurity_gpu);
    }
    else{
        set_passivated_wp_calc<<<num_blocks,num_threads>>>(L, R, gpu_setup->impurity_gpu, num_threads);
    }

    for (iter = 0; iter < setup->max_iterations; iter++) {

        /* the following definition of the factor for over-relaxation improves convergence
            time by a factor ~ 70-120 for a 2kg ICPC detector, grid = 0.1 mm
          OR_fact increases with increasing volxel count ((L+1)*(R+1))
                and with increasing iteration number
          0.997 is maximum asymptote for very large pixel count and iteration number */
        
        double OR_fact;
        if (ev_calc)  OR_fact = (1.991 - 1500.0/(L*R));
        else          OR_fact = (1.992 - 1500.0/(L*R));
        if (OR_fact < 1.4) OR_fact = 1.4;
        if (iter < 1) OR_fact = 1.0;
    
        old_gpu_relax = new_gpu_relax;
        new_gpu_relax = 1 - new_gpu_relax;   
    
        /*
        Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
        Max dimension size of a grid size (x,y,z): (65535, 65535, 65535)
        */
        int num_threads = 300;
        int num_blocks = R * (ceil(L/num_threads)+1); //The +1 is just a precaution to make sure all R and Z values are included
    
        //dividing into R blocks of L threads
        if (vacuum_gap_gpu > 0) {   // modify impurity value along passivated surface due to surface charge induced by capacitance
            vacuum_gap_calc<<<R-1,1>>>(gpu_setup->impurity_gpu, gpu_setup->v_gpu,L, R, grid, old_gpu_relax, new_gpu_relax, vacuum_gap_gpu, e_over_E_gpu);
          }
          cudaDeviceSynchronize();
      
          z_relection_set<<<R-1,1>>>(gpu_setup->v_gpu, L, R, old_gpu_relax, new_gpu_relax, gpu_setup->eps_dr_gpu, gpu_setup->eps_dr_gpu, gpu_setup->point_type_gpu);
          cudaDeviceSynchronize();
      
          reflection_symmetry_set<<<L-1,1>>>(gpu_setup->v_gpu, L, R, old_gpu_relax, new_gpu_relax);
          cudaDeviceSynchronize();
          
          // updates all the red cell in parallel
          relax_step<<<num_blocks,num_threads>>>(1,ev_calc, L, R, grid, OR_fact, gpu_setup->v_gpu, gpu_setup->point_type_gpu, gpu_setup->dr_gpu, gpu_setup->dz_gpu, gpu_setup->eps_dr_gpu, gpu_setup->eps_dz_gpu, 
            gpu_setup->s1_gpu, gpu_setup->s2_gpu, gpu_setup->impurity_gpu, gpu_setup->diff_array, old_gpu_relax, new_gpu_relax, num_threads, vacuum_gap_gpu);
          cudaDeviceSynchronize();
          // updates all the black cells in parallel
          relax_step<<<num_blocks,num_threads>>>(0,ev_calc, L, R, grid, OR_fact, gpu_setup->v_gpu, gpu_setup->point_type_gpu, gpu_setup->dr_gpu, gpu_setup->dz_gpu, gpu_setup->eps_dr_gpu, gpu_setup->eps_dz_gpu, 
            gpu_setup->s1_gpu, gpu_setup->s2_gpu, gpu_setup->impurity_gpu, gpu_setup->diff_array, old_gpu_relax, new_gpu_relax, num_threads, vacuum_gap_gpu);
               
        cudaDeviceSynchronize();

        // The Thrust library uses parallel reduction methods to find the maximum difference between old and new iteration and sum of all differences
        
        // thrust::host_vector<double> h_state(R*L);
        // thrust::device_vector<curandStatePhilox4_32_10_t> d_state = h_state;
        double sum_dif_thrust = thrust::reduce(thrust::device_pointer_cast(gpu_setup->diff_array), thrust::device_pointer_cast(gpu_setup->diff_array) + (L+1)*(R+1));
        double max_value_thrust = thrust::max_element(thrust::device_pointer_cast(gpu_setup->diff_array), thrust::device_pointer_cast(gpu_setup->diff_array)+(L+1)*(R+1))[0];
        // thrust::device_ptr<double> dev_ptr = thrust::device_pointer_cast(gpu_setup->diff_array);
        // double sum_dif_thrust = thrust::reduce(dev_ptr, dev_ptr + R*L);
        // thrust::device_ptr<double> max_ptr = thrust::max_element(dev_ptr, dev_ptr+R*L);
        // double max_value_thrust = max_ptr[0];
        cudaDeviceSynchronize();
    
        // Uncomment next four lines to print some results of relaxation
        // if (iter < 10 || (iter < 600 && iter%100 == 0) || iter%1000 == 0) {
        // print_output<<<1,1>>>(R, L, OR_fact, iter, ev_calc, old_gpu_relax, new_gpu_relax, gpu_setup->v_gpu, max_value_thrust, sum_dif_thrust);
        // }
        // cudaDeviceSynchronize();
    
        // check for convergence
        if ( ev_calc && max_value_thrust < 0.00000008) break;
        if ( ev_calc && max_value_thrust < 0.0008) break;  // comment out if you want convergence at the numerical error level
        if (!ev_calc && max_value_thrust < 0.0000000001) break;
        if (!ev_calc && max_value_thrust < 0.000001) break;  // comment out if you want convergence at the numerical error level
        
    } // end of iter loop

    printf("Iterations taken by R-B SOR to converge: %d\n", iter);

    if (setup->vacuum_gap > 0) {   // restore impurity value along passivated surface
        restore_passivated_imp<<<R,1>>>(R, gpu_setup->impurity_gpu);
      }

return 0;
}

