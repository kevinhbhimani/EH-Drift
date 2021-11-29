/*
  GPU struct keeps tract of variables on GPU memory. The struct must be passesed by pointer to a function.
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
#include <cuda.h>
#include <cuda_runtime.h>



//    #if __cplusplus
//   extern "C" {
//   #endif

typedef struct {

double *v_gpu;
char *point_type_gpu;
double *dr_gpu;
double *dz_gpu; 
double *eps_dr_gpu;
double *eps_dz_gpu;
double *s1_gpu; 
double *s2_gpu; 
double *impurity_gpu;
double *diff_array;

float *rho_e_gpu;
float *rho_h_gpu;
float *drift_offset_e_gpu;
float *drift_offset_h_gpu;
float *drift_slope_e_gpu;
float *drift_slope_h_gpu;
double *deltaez_array;
double *deltaer_array;
double *fr_array;
double *fz_array;
int *i_array;
int *k_array;

// __managed__ float vacuum_gap_gpu;
// __managed__ double e_over_E_gpu;
// __managed__ float fq;
// __managed__ int idid,idod,idd;
// __managed__ double f_drift;
// __managed__ float tstep;
// __managed__ float delta;
// __managed__ float delta_r;
// __managed__ float wrap_around_radius;
// __managed__ float ditch_thickness;
// __managed__ float ditch_depth;
// __managed__ float surface_drift_vel_factor;

} GPU_data;


extern __global__ void vacuum_gap_calc(double *impurity_gpu,double *v_gpu, int L, int R, float grid, int old_gpu_relax, int new_gpu_relax, float vacuum_gap_gpu, double e_over_E_gpu);
extern __global__ void z_relection_set(double* v_gpu, int L, int R, int old_gpu_relax, int new_gpu_relax, double *eps_dr_gpu, double *eps_dz_gpu, char *point_type_gpu);
extern __global__ void reflection_symmetry_set(double *v_gpu, int L, int R, int old_gpu_relax, int new_gpu_relax);
extern __global__ void relax_step(int is_red, int ev_calc, int L, int R, float grid, double OR_fact, double *v_gpu, char *point_type_gpu, double *dr_gpu, double *dz_gpu, 
              double *eps_dr_gpu, double *eps_dz_gpu, double *s1_gpu, double *s2_gpu, double *impurity_gpu, double *diff_array, int old_gpu_relax, int new_gpu_relax, int max_threads, float vacuum_gap_gpu);
extern __global__ void print_output(int R, int L, double OR_fact, int iter, int ev_calc, int old_gpu_relax, int new_gpu_relax, double *v_gpu, double max_dif_gpu, double sum_dif_gpu);

  // #if __cplusplus
  // }
  // #endif