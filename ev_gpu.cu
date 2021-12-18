/*
  This program is called to calculate electric field during initial setup of the detector
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

#include <time.h>

#include "gpu_vars.h"



extern "C" int ev_calc_gpu_initial(MJD_Siggen_Setup *setup, MJD_Siggen_Setup *old_setup, GPU_data *gpu_setup);
extern "C" int ev_calc_rb_cpu(MJD_Siggen_Setup *setup);

int do_relax_gpu(MJD_Siggen_Setup *setup, int ev_calc, GPU_data *gpu_setup);
int do_relax_rb(MJD_Siggen_Setup *setup, int ev_calc);
int interpolate_gpu(MJD_Siggen_Setup *setup, MJD_Siggen_Setup *old_setup);
int ev_relax_undep_gpu(MJD_Siggen_Setup *setup);
int write_ev_gpu(MJD_Siggen_Setup *setup);
int report_config_gpu(FILE *fp_out, char *config_file_name);



__global__ void z_relection_set(double* v_gpu, int L, int R, int old_gpu_relax, int new_gpu_relax, double *eps_dr_gpu, double *eps_dz_gpu, char *point_type_gpu){
  /* manage z=0 reflection symmetry   
setup->v[0][0]   = setup->v[0][2];
setup->v[1][0]   = setup->v[1][2];
setup->eps[0]    = setup->eps[1];
setup->eps_dr[0] = setup->eps_dr[1];
setup->eps_dz[0] = setup->eps_dz[1];
setup->point_type[0] = setup->point_type[1]; */
int r = blockIdx.x+1;
v_gpu[(old_gpu_relax*(L+1)*(R+1))+((R+1)*0)+r] = v_gpu[(old_gpu_relax*(L+1)*(R+1))+((R+1)*2)+r];
v_gpu[(new_gpu_relax*(L+1)*(R+1))+((R+1)*0)+r] = v_gpu[(new_gpu_relax*(L+1)*(R+1))+((R+1)*2)+r];
eps_dr_gpu[((R+1)*0)+r] = eps_dr_gpu[((R+1)*1)+r];
eps_dz_gpu[((R+1)*0)+r] = eps_dz_gpu[((R+1)*1)+r];
point_type_gpu[((R+1)*0)+r] = point_type_gpu[((R+1)*1)+r];
}
__global__ void vacuum_gap_calc(double *impurity_gpu,double *v_gpu, int L, int R, float grid, int old_gpu_relax, int new_gpu_relax, float vacuum_gap_gpu, double e_over_E_gpu){
int r = blockIdx.x+1;
  //printf("in vaccuum gap chunk \n");
impurity_gpu[((R+1)*1)+r] = impurity_gpu[((R+1)*0)+r] - v_gpu[(old_gpu_relax*(L+1)*(R+1))+((R+1)*1)+r] * 5.52e-4 * e_over_E_gpu * grid / vacuum_gap_gpu;
}
__global__ void reflection_symmetry_set(double *v_gpu, int L, int R, int old_gpu_relax, int new_gpu_relax){
int z = blockIdx.x+1;
/* manage r=0 reflection symmetry */
v_gpu[(old_gpu_relax*(L+1)*(R+1))+((R+1)*z)+0] = v_gpu[(new_gpu_relax*(L+1)*(R+1))+((R+1)*z)+0] = v_gpu[(old_gpu_relax*(L+1)*(R+1))+((R+1)*z)+2];
}
//Kernal for doing the relaxation step
// called as:     relax_step<<<R-1,L-1>>>(ev_calc, L, R, grid, OR_fact, v_gpu, point_type_gpu, dr_gpu, dz_gpu, eps_dr_gpu, eps_dz_gpu, 
//                  s1_gpu, s2_gpu, impurity_gpu, diff_array, old_gpu_relax, new_gpu_relax);   
__global__ void relax_step(int is_red, int ev_calc, int L, int R, float grid, double OR_fact, double *v_gpu, char *point_type_gpu, double *dr_gpu, double *dz_gpu, 
double *eps_dr_gpu, double *eps_dz_gpu, double *s1_gpu, double *s2_gpu, double *impurity_gpu, double *diff_array, int old_gpu_relax, int new_gpu_relax, int max_threads, float vacuum_gap_gpu){

int r = blockIdx.x%R;
int z = (floorf(blockIdx.x/R) * max_threads) + threadIdx.x;

/* start from z=1 and r=1 so that (z,r)=0 can be
used for reflection symmetry around r=0 or z=0 */
if(r==0 || z==0 || r>=R || z>=L){
  return;
}

int iter_var;

if(is_red){
  //is even, the node is red, use only even values
  iter_var = old_gpu_relax;
  if ((r+z)%2!=0){
    return;
  }
}
else{
  iter_var = new_gpu_relax;
  if ((r+z)%2!=1){
    return;
  }
}


double eps_sum, v_sum;

if (point_type_gpu[((R+1)*z)+r] < INSIDE){
  return;
}

if (point_type_gpu[((R+1)*z)+r] < DITCH) {       // normal bulk or passivated surface, no complications
  v_sum = (v_gpu[((iter_var*(L+1)*(R+1))+((R+1)*(z+1))+r)] + v_gpu[((iter_var*(L+1)*(R+1))+((R+1)*z)+r+1)]*s1_gpu[r] +
            v_gpu[((iter_var*(L+1)*(R+1))+((R+1)*(z-1))+r)] + v_gpu[((iter_var*(L+1)*(R+1))+((R+1)*z)+r-1)]*s2_gpu[r]);
  if (r > 1) {
    eps_sum = 4;  
  }
  else {
    eps_sum = 2 + s1_gpu[r] + s2_gpu[r];}
  }
  else if (point_type_gpu[((R+1)*z)+r] == CONTACT_EDGE) {  // adjacent to the contact
  v_sum = (v_gpu[((iter_var*(L+1)*(R+1))+((R+1)*(z+1))+r)]*dz_gpu[((1*(L+1)*(R+1))+((R+1)*z)+r)] + v_gpu[((iter_var*(L+1)*(R+1))+((R+1)*z)+r+1)]*dr_gpu[((1*(L+1)*(R+1))+((R+1)*z)+r)] +
            v_gpu[((iter_var*(L+1)*(R+1))+((R+1)*(z-1))+r)]*dz_gpu[((0*(L+1)*(R+1))+((R+1)*z)+r)] + v_gpu[((iter_var*(L+1)*(R+1))+((R+1)*z)+r-1)]*dr_gpu[((0*(L+1)*(R+1))+((R+1)*z)+r)]);
  eps_sum = dz_gpu[((1*(L+1)*(R+1))+((R+1)*z)+r)] + dr_gpu[((1*(L+1)*(R+1))+((R+1)*z)+r)] + dz_gpu[((0*(L+1)*(R+1))+((R+1)*z)+r)] + dr_gpu[((0*(L+1)*(R+1))+((R+1)*z)+r)];
} 
else if (point_type_gpu[((R+1)*z)+r] >= DITCH) {  // in or adjacent to the ditch
  v_sum = (v_gpu[((iter_var*(L+1)*(R+1))+((R+1)*(z+1))+r)]*eps_dz_gpu[((R+1)*z)+r] + v_gpu[((iter_var*(L+1)*(R+1))+((R+1)*z)+r+1)]*eps_dr_gpu[((R+1)*z)+r]*s1_gpu[r] +
            v_gpu[((iter_var*(L+1)*(R+1))+((R+1)*(z-1))+r)]*eps_dz_gpu[((R+1)*(z-1))+r] + v_gpu[((iter_var*(L+1)*(R+1))+((R+1)*z)+r-1)]*eps_dr_gpu[((R+1)*z)+r-1]*s2_gpu[r]);
  eps_sum = (eps_dz_gpu[((R+1)*z)+r]   + eps_dr_gpu[((R+1)*z)+r]  *s1_gpu[r] +
              eps_dz_gpu[((R+1)*(z-1))+r] + eps_dr_gpu[((R+1)*z)+r-1]*s2_gpu[r]);

}

 // calculate the interpolated mean potential and the effect of the space charge
if ((ev_calc || (vacuum_gap_gpu > 0 && z == 1)) && point_type_gpu[((R+1)*z)+r] < CONTACT_EDGE && r > 1 && z > 1) {   // normal bulk, no complications
v_gpu[((new_gpu_relax*(L+1)*(R+1))+((R+1)*z)+r)] = (1.0-OR_fact)*v_gpu[((old_gpu_relax*(L+1)*(R+1))+((R+1)*z)+r)] + OR_fact * (v_sum / eps_sum + impurity_gpu[((R+1)*z)+r]);
if(z==2){
  v_gpu[(old_gpu_relax*(L+1)*(R+1))+((R+1)*0)+r] = v_gpu[(old_gpu_relax*(L+1)*(R+1))+((R+1)*2)+r];
  v_gpu[(new_gpu_relax*(L+1)*(R+1))+((R+1)*0)+r] = v_gpu[(new_gpu_relax*(L+1)*(R+1))+((R+1)*2)+r];
  }
} 

else if (ev_calc || (vacuum_gap_gpu > 0 && z == 1)) {

v_gpu[((new_gpu_relax*(L+1)*(R+1))+((R+1)*z)+r)] = v_sum / eps_sum + impurity_gpu[((R+1)*z)+r];
if(z==2){
  v_gpu[(old_gpu_relax*(L+1)*(R+1))+((R+1)*0)+r] = v_gpu[(old_gpu_relax*(L+1)*(R+1))+((R+1)*2)+r];
  v_gpu[(new_gpu_relax*(L+1)*(R+1))+((R+1)*0)+r] = v_gpu[(new_gpu_relax*(L+1)*(R+1))+((R+1)*2)+r];
  }
} 
else if (point_type_gpu[((R+1)*z)+r] < CONTACT_EDGE && r > 1 && z > 1) {   // normal bulk, no complications
v_gpu[((new_gpu_relax*(L+1)*(R+1))+((R+1)*z)+r)] = (1.0-OR_fact)*v_gpu[((old_gpu_relax*(L+1)*(R+1))+((R+1)*z)+r)] + OR_fact * v_sum / eps_sum;
} 
else {                          // over-relaxation at the edges seems to make things worse
v_gpu[((new_gpu_relax*(L+1)*(R+1))+((R+1)*z)+r)] = v_sum / eps_sum;
}

// calculate difference from last iteration, for convergence check
diff_array[((R+1)*z)+r] = fabs(v_gpu[(old_gpu_relax*(L+1)*(R+1))+((R+1)*z)+r] - v_gpu[(new_gpu_relax*(L+1)*(R+1))+((R+1)*z)+r]);
//end of going through the detector
}

__global__ void print_output(int R, int L, double OR_fact, int iter, int ev_calc, int old_gpu_relax, int new_gpu_relax, double *v_gpu, double max_dif_gpu, double sum_dif_gpu){
  if (0 && ev_calc) {
    printf("%5d %d %d %.10f \n", iter, old_gpu_relax, new_gpu_relax, max_dif_gpu);
  } else {
    printf("GPU calculation: iter=%5d old=%d new=%d OR_fact=%f max_dif=%.10f sum_diff/(L-2)/(R-2)=%.10f; v_center=%.10f v_at_L/3_R/3=%.10f\n",
            iter, old_gpu_relax, new_gpu_relax, OR_fact, max_dif_gpu, sum_dif_gpu/(L-2)/(R-2),
            v_gpu[(new_gpu_relax*(L+1)*(R+1))+((R+1)*(L/2))+R/2], v_gpu[(new_gpu_relax*(L+1)*(R+1))+((R+1)*(L/3))+R/3]);
  } 
}
/* 
    Performs relaxation on GPUs using Red- Black SOR method to calculate the new electric potentials.
    This function is different that the one called in the time step loop because it uses different setups and grid sizes
 */
extern "C" int ev_calc_gpu_initial(MJD_Siggen_Setup *setup, MJD_Siggen_Setup *old_setup, GPU_data *gpu_setup) {
  int    i, j;
  float  grid = setup->xtal_grid;
  int    L  = lrint(setup->xtal_length/grid)+3;
  int    R  = lrint(setup->xtal_radius/grid)+3;

  if (!old_setup) {
    printf("\n\n ---- starting EV calculation --- \n");
    for (i = 1; i < L; i++) {
      for (j = 1; j < R; j++) {
        setup->v[0][i][j] = setup->v[1][i][j] = setup->xtal_HV/2.0;
      }
    }
  }
  if (old_setup) interpolate_gpu(setup, old_setup);
  setup->fully_depleted = 1;
  setup->bubble_volts = 0;

  /* set boundary voltages */
  for (i = 1; i < L; i++) {
    for (j = 1; j < R; j++) {
      if (setup->point_type[i][j] == HVC)
        setup->v[0][i][j] = setup->v[1][i][j] = setup->xtal_HV;
      if (setup->point_type[i][j] == PC)
        setup->v[0][i][j] = setup->v[1][i][j] = 0.0;
    }
  }

  if (!old_setup || !old_setup->fully_depleted) ev_relax_undep_gpu(setup);
  else {
    // printf("Performing calculations on GPU\n");
    // clock_t start, end;
    // double gpu_time_used;
    // start = clock();
    do_relax_gpu(setup, 1, gpu_setup);
    // end = clock();
    // gpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    // printf("Time used by GPU in seconds: %f for grid of %f \n\n", gpu_time_used, setup->xtal_grid);
  }  //else do_relax_rb(setup, 1);


  if (setup->write_field) write_ev_gpu(setup);

  if (setup->fully_depleted) {
    printf("Detector is fully depleted.\n");
    /* save potential close to point contact, to use later when calculating depletion voltage */
    for (i = 1; i < L; i++) {
      for (j = 1; j < R; j++) {
        setup->vsave[i][j] = fabs(setup->v[1][i][j]);
      }
    }
  } else {
    printf("Detector is not fully depleted.\n");
    if (setup->bubble_volts > 0)
      printf("Pinch-off bubble at %.1f V potential\n", setup->bubble_volts);
    if (!old_setup) {
      // write a little file that shows any undepleted voxels in the crystal
      FILE *file = fopen("undepleted.txt", "w");
      for (j = R-1; j > 0; j--) {
	setup->undepleted[j][L-1] = '\0';
	fprintf(file, "%s\n", setup->undepleted[j]+1);
      }
      fclose(file);
    }
  }

  return 0;
} /* ev_calc */


/* -------------------------------------- do_relax_gpu ------------------- */
int do_relax_gpu(MJD_Siggen_Setup *setup, int ev_calc, GPU_data *gpu_setup) {

  int    iter, r, z;
  float  grid = setup->xtal_grid;
  int    L  = lrint(setup->xtal_length/grid)+2;
  int    R  = lrint(setup->xtal_radius/grid)+2;
  int  old_gpu_relax = 1, new_gpu_relax = 0;

  double e_over_E_gpu = 11.310; // e/epsilon; for 1 mm2, charge units 1e10 e/cm3, espilon = 16*epsilon0
  float vacuum_gap_gpu = setup->vacuum_gap;
  

  if (ev_calc) {
    // for field calculation, save impurity value along passivated surface
    for (r = 1; r < R; r++)
      setup->impurity[0][r] = setup->impurity[1][r];
  } else {
    // for WP calculation, clear all impurity values
    for (z = 0; z < L; z++) {
      for (r = 1; r < R; r++) {
        setup->impurity[z][r] = 0;
      }
    }
  }

/*
Below we allocate and copy values to GPU. For better memory management and indexing, all multidemensional arrays are flattened.
The conversion for flattening array[i][j][k] = flat_array[(i*(L+1)*(R+1))+((R+1)*j)+k]
*/

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
point_type_flat = (char*)malloc(sizeof(char)*(L+1)*(R+1));
cudaMalloc((void**)&gpu_setup->point_type_gpu, sizeof(char)*(L+1)*(R+1));
for(int j=0; j<=L; j++){
    for(int k=0; k<=R; k++){
      point_type_flat[((R+1)*j)+k] = setup->point_type[j][k];
    }
  }
cudaMemcpy(gpu_setup->point_type_gpu, point_type_flat, sizeof(char)*(L+1)*(R+1), cudaMemcpyHostToDevice);

double *dr_flat; 
dr_flat = (double*)malloc(2*sizeof(double)*(L+1)*(R+1));
cudaMalloc((void**)&gpu_setup->dr_gpu, 2*sizeof(double)*(L+1)*(R+1));
  for(int i=0; i<2; i++) {
    for(int j=1; j<=L; j++){
        for(int k=0; k<=R; k++){
          dr_flat[(i*(L+1)*(R+1))+((R+1)*j)+k] = setup->dr[i][j][k];
        }
      }
    }
cudaMemcpy(gpu_setup->dr_gpu, dr_flat, 2*sizeof(double)*(L+1)*(R+1), cudaMemcpyHostToDevice);

double *dz_flat;
dz_flat = (double*)malloc(2*sizeof(double)*(L+1)*(R+1));
cudaMalloc((void**)&gpu_setup->dz_gpu, 2*sizeof(double)*(L+1)*(R+1));
  for(int i=0; i<2; i++) {
    for(int j=1; j<=L; j++){
        for(int k=0; k<=R; k++){
          dz_flat[(i*(L+1)*(R+1))+((R+1)*j)+k] = setup->dz[i][j][k];
        }
      }
    }
cudaMemcpy(gpu_setup->dz_gpu, dz_flat, 2*sizeof(double)*(L+1)*(R+1), cudaMemcpyHostToDevice);

cudaMalloc((void**)&gpu_setup->s1_gpu, sizeof(double)*(R+1));
cudaMemcpy(gpu_setup->s1_gpu, setup->s1, sizeof(double)*(R+1), cudaMemcpyHostToDevice);


cudaMalloc((void**)&gpu_setup->s2_gpu, sizeof(double)*(R+1));
cudaMemcpy(gpu_setup->s2_gpu, setup->s2, sizeof(double)*(R+1), cudaMemcpyHostToDevice);

double *eps_dr_flat;
eps_dr_flat = (double*)malloc(sizeof(double)*(L+1)*(R+1));
cudaMalloc((void**)&gpu_setup->eps_dr_gpu, sizeof(double)*(L+1)*(R+1));
for(int j=0; j<=L; j++){
  for(int k=0; k<=R; k++){
    eps_dr_flat[((R+1)*j)+k] = setup->eps_dr[j][k];
  }
}
cudaMemcpy(gpu_setup->eps_dr_gpu, eps_dr_flat, sizeof(double)*(L+1)*(R+1), cudaMemcpyHostToDevice);


double *eps_dz_flat;
eps_dz_flat = (double*)malloc(sizeof(double)*(L+1)*(R+1));
cudaMalloc((void**)&gpu_setup->eps_dz_gpu, sizeof(double)*(L+1)*(R+1));
for(int j=0; j<=L; j++){
  for(int k=0; k<=R; k++){
    eps_dz_flat[((R+1)*j)+k] = setup->eps_dz[j][k];
  }
}
cudaMemcpy(gpu_setup->eps_dz_gpu, eps_dz_flat, sizeof(double)*(L+1)*(R+1), cudaMemcpyHostToDevice);

double *impurity_flat;
impurity_flat = (double*)malloc(sizeof(double)*(L+1)*(R+1));
cudaMalloc((void**)&gpu_setup->impurity_gpu, sizeof(double)*(L+1)*(R+1));
for(int j=0; j<=L; j++){
  for(int k=0; k<=R; k++){
    impurity_flat[((R+1)*j)+k] = setup->impurity[j][k];
  }
}
cudaMemcpy(gpu_setup->impurity_gpu, impurity_flat, sizeof(double)*(L+1)*(R+1), cudaMemcpyHostToDevice);


double *diff_array_cpu;

diff_array_cpu = (double*)malloc(sizeof(double)*(L+1)*(R+1));
cudaMalloc((void**)&gpu_setup->diff_array, sizeof(double)*(L+1)*(R+1));

for (int i = 0; i<(L+1)*(R+1); i++){
  diff_array_cpu[i] =0.00;
}
cudaMemcpy(gpu_setup->diff_array, diff_array_cpu, sizeof(double)*(L+1)*(R+1), cudaMemcpyHostToDevice);

// thrust::device_ptr<double> dev_ptr;


for (iter = 0; iter < setup->max_iterations; iter++) {

/* the following definition of the factor for over-relaxation improves convergence
    time by a factor ~ 70-120 for a 2kg ICPC detector, grid = 0.1 mm
  OR_fact increases with increasing volxel count ((L+1)*(R+1))
        and with increasing iteration number
  0.997 is maximum asymptote for very large pixel count and iteration number */

// if(iter==1){
//   break;
// }

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

  if(num_blocks<=65535 && num_threads<=1024){
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
    
  }
  else{
    printf("--------Pick a smaller block please--------\n");
    return 0;
  }
  cudaDeviceSynchronize();

  // The Thrust library uses parallel reduction methods to find the maximum difference between old and new iteration and sum of all differences
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
  // if ( ev_calc && max_value_thrust < 0.00000008) break;
  if ( ev_calc && max_value_thrust < 0.0008) break;  // comment out if you want convergence at the numerical error level
  // if (!ev_calc && max_value_thrust < 0.0000000001) break;
  if (!ev_calc && max_value_thrust < 0.000001) break;  // comment out if you want convergence at the numerical error level

  /* every 100 iterations, check that detector is really depleted*/
  /*
  if (ev_calc && iter > 190 && iter%100 == 0) {
    for (z = 1; z < L; z++) {
      gpu_setup->v_gpu[old_gpu_relax][z][0] = gpu_setup->v_gpu[old_gpu_relax][z][2];
      for (r = 1; r < R; r++) {
        if (setup->point_type[z][r] < INSIDE) continue;   // HV or point contact
        if (v[new_gpu_relax][z][r] < 0 ||
            (v[new_gpu_relax][z][r] < v[new_gpu_relax][z][r] &&
              v[new_gpu_relax][z][r] < v[new_gpu_relax][z][r] &&
              v[new_gpu_relax][z][r] < v[new_gpu_relax][z][r] &&
              v[new_gpu_relax][z][r] < v[new_gpu_relax][z][r])) {
          printf("Detector may not be fully depleted. Switching to ev_relax_undep_gpu()\n");
          ev_relax_undep_gpu(setup);
          return 0;
        }
      }
    }
  }
  */
  } // end of iter loop
  printf("Iterations taken by R-B SOR to converge: %d\n", iter);

//copy values that were changed back to CPU
  cudaMemcpy(v_flat, gpu_setup->v_gpu, 2*sizeof(double)*(L+1)*(R+1), cudaMemcpyDeviceToHost);

  for(int i=0; i<2; i++) {
    for(int j=0; j<=L; j++){
      for(int k=0; k<=R; k++){
        memcpy(&setup->v[i][j][k], &v_flat[(i*(L+1)*(R+1))+((R+1)*j)+k], sizeof(double));
      }
    }
  }

  for (int z = 1; z < L; z++) {
    /* manage r=0 reflection symmetry */
    setup->v[old_gpu_relax][z][0] = setup->v[new_gpu_relax][z][0] = setup->v[old_gpu_relax][z][2];
  }


  setup->v[0][0] = setup->v[0][2];
  setup->v[1][0] = setup->v[1][2];

  if (setup->vacuum_gap > 0) {   // restore impurity value along passivated surface
    for (r = 1; r < R; r++)
      setup->impurity[1][r] = setup->impurity[0][r];
  }

// Free memory on GPU
cudaFree(gpu_setup->v_gpu);
cudaFree(gpu_setup->point_type_gpu);
cudaFree(gpu_setup->dr_gpu);
cudaFree(gpu_setup->dz_gpu);
cudaFree(gpu_setup->eps_dr_gpu);
cudaFree(gpu_setup->eps_dz_gpu);
cudaFree(gpu_setup->s1_gpu);
cudaFree(gpu_setup->s1_gpu);
cudaFree(gpu_setup->impurity_gpu);
cudaFree(gpu_setup->diff_array);

// Free memory on CPU
free(v_flat);
free(point_type_flat);
free(dr_flat);
free(dz_flat);
free(eps_dr_flat);
free(eps_dz_flat);
free(impurity_flat);
free(diff_array_cpu);

return 0;
} /* do_relax_gpu */

/* -------------------------------------- ev_calc_rb_cpu ------------------- */
extern "C" int ev_calc_rb_cpu(MJD_Siggen_Setup *setup) {
  int    i, j;
  float  grid = setup->xtal_grid;
  int    L  = lrint(setup->xtal_length/grid)+3;
  int    R  = lrint(setup->xtal_radius/grid)+3;

  setup->fully_depleted = 1;
  setup->bubble_volts = 0;

  /* set boundary voltages */
  for (i = 1; i < L; i++) {
    for (j = 1; j < R; j++) {
      if (setup->point_type[i][j] == HVC)
        setup->v[0][i][j] = setup->v[1][i][j] = setup->xtal_HV;
      if (setup->point_type[i][j] == PC)
        setup->v[0][i][j] = setup->v[1][i][j] = 0.0;
    }
  }

  do_relax_rb(setup, 1);
  // if (setup->write_field) write_ev(setup);

  if (setup->fully_depleted) {
    printf("Detector is fully depleted.\n");
    /* save potential close to point contact, to use later when calculating depletion voltage */
    for (i = 1; i < L; i++) {
      for (j = 1; j < R; j++) {
        setup->vsave[i][j] = fabs(setup->v[1][i][j]);
      }
    }
  } else {
    printf("Detector is not fully depleted.\n");
    if (setup->bubble_volts > 0)
      printf("Pinch-off bubble at %.1f V potential\n", setup->bubble_volts);
  }

  return 0;
} /* ev_calc_rb_cpu */




/* -------------------------------------- do_relax_rb ------------------- */
int do_relax_rb(MJD_Siggen_Setup *setup, int ev_calc) {
  int    old_rb = 1, new_rb = 0, iter, r, z;
  float  grid = setup->xtal_grid;
  int    L  = lrint(setup->xtal_length/grid)+2;
  int    R  = lrint(setup->xtal_radius/grid)+2;
  double eps_sum, v_sum, dif, sum_dif, max_dif;
  double ***v = setup->v, **eps_dr = setup->eps_dr, **eps_dz = setup->eps_dz;
  double ***dr = setup->dr, ***dz = setup->dz;
  double *s1 = setup->s1, *s2 = setup->s2;
  double e_over_E = 11.310; // e/epsilon; for 1 mm2, charge units 1e10 e/cm3, espilon = 16*epsilon0


  if (ev_calc) {
    // for field calculation, save impurity value along passivated surface
    for (r = 1; r < R; r++)
      setup->impurity[0][r] = setup->impurity[1][r];
  } else {
    // for WP calculation, clear all impurity values
    for (z = 0; z < L; z++) {
      for (r = 1; r < R; r++) {
        setup->impurity[z][r] = 0;
      }
    }
  }

  for (iter = 0; iter < setup->max_iterations; iter++) {

    /* the following definition of the factor for over-relaxation improves convergence
           time by a factor ~ 70-120 for a 2kg ICPC detector, grid = 0.1 mm
         OR_fact increases with increasing volxel count (L*R)
               and with increasing iteration number
         0.997 is maximum asymptote for very large pixel count and iteration number */
    double OR_fact;
    if (ev_calc)  OR_fact = (1.991 - 1500.0/(L*R));
    else          OR_fact = (1.992 - 1500.0/(L*R));
    if (OR_fact < 1.4) OR_fact = 1.4;
    // if (iter == 0) printf("OR_fact = %f\n", OR_fact);
    if (iter < 1) OR_fact = 1.0;

    old_rb = new_rb;
    new_rb = 1 - new_rb;
    sum_dif = 0;
    max_dif = 0;

    if (setup->vacuum_gap > 0) {   // modify impurity value along passivated surface
      for (r = 1; r < R; r++)      //   due to surface charge induced by capacitance
        setup->impurity[1][r] = setup->impurity[0][r] -
          v[old_rb][1][r] * 5.52e-4 * e_over_E * grid / setup->vacuum_gap;
    }

    /* start from z=1 and r=1 so that (z,r)=0 can be
       used for reflection symmetry around r=0 or z=0 */
    for (z = 1; z < L; z++) {
      /* manage r=0 reflection symmetry */
      setup->v[old_rb][z][0] = setup->v[new_rb][z][0] = setup->v[old_rb][z][2];
    }

    //iterate for red nodes
    for (z = 1; z < L; z++) {
      for (r = 1; r < R; r++) {
        if ((r+z)%2 == 0){
          if (setup->point_type[z][r] < INSIDE) continue;   // HV or point contact

          if (setup->point_type[z][r] < DITCH) {       // normal bulk or passivated surface, no complications
            v_sum = (v[old_rb][z+1][r] + v[old_rb][z][r+1]*s1[r] +
                    v[old_rb][z-1][r] + v[old_rb][z][r-1]*s2[r]);
            if (r > 1) eps_sum = 4;
            else       eps_sum = 2 + s1[r] + s2[r];
          } else if (setup->point_type[z][r] == CONTACT_EDGE) {  // adjacent to the contact
            v_sum = (v[old_rb][z+1][r]*dz[1][z][r] + v[old_rb][z][r+1]*dr[1][z][r] +
                    v[old_rb][z-1][r]*dz[0][z][r] + v[old_rb][z][r-1]*dr[0][z][r]);
            eps_sum = dz[1][z][r] + dr[1][z][r] + dz[0][z][r] + dr[0][z][r];
          } else if (setup->point_type[z][r] >= DITCH) {  // in or adjacent to the ditch
            v_sum = (v[old_rb][z+1][r]*eps_dz[z  ][r] + v[old_rb][z][r+1]*eps_dr[z][r  ]*s1[r] +
                    v[old_rb][z-1][r]*eps_dz[z-1][r] + v[old_rb][z][r-1]*eps_dr[z][r-1]*s2[r]);
            eps_sum = (eps_dz[z][r]   + eps_dr[z][r]  *s1[r] +
                      eps_dz[z-1][r] + eps_dr[z][r-1]*s2[r]);
          }

          // calculate the interpolated mean potential and the effect of the space charge

          if ((ev_calc || (setup->vacuum_gap > 0 && z == 1)) &&
              setup->point_type[z][r] < CONTACT_EDGE && r > 1 && z > 1) {   // normal bulk, no complications
            v[new_rb][z][r] = (1.0-OR_fact)*v[old_rb][z][r] + OR_fact * (v_sum / eps_sum + setup->impurity[z][r]);
          } else if (ev_calc || (setup->vacuum_gap > 0 && z == 1)) {
            v[new_rb][z][r] = v_sum / eps_sum + setup->impurity[z][r];
          } else if (setup->point_type[z][r] < CONTACT_EDGE && r > 1 && z > 1) {   // normal bulk, no complications
            v[new_rb][z][r] = (1.0-OR_fact)*v[old_rb][z][r] + OR_fact * v_sum / eps_sum;
          } else {                          // over-relaxation at the edges seems to make things worse
            v[new_rb][z][r] = v_sum / eps_sum;
          }

          // calculate difference from last iteration, for convergence check
          dif = fabs(v[old_rb][z][r] - v[new_rb][z][r]);
          sum_dif += dif;
          if (max_dif < dif) max_dif = dif;
        }
      }
    }

    //iterate for black nodes
    for (z = 1; z < L; z++) {
      for (r = 1; r < R; r++) {
        if ((r+z)%2 != 0){
          if (setup->point_type[z][r] < INSIDE) continue;   // HV or point contact
          if (setup->point_type[z][r] < DITCH) {       // normal bulk or passivated surface, no complications
            v_sum = (v[new_rb][z+1][r] + v[new_rb][z][r+1]*s1[r] +
                    v[new_rb][z-1][r] + v[new_rb][z][r-1]*s2[r]);
            if (r > 1) eps_sum = 4;
            else       eps_sum = 2 + s1[r] + s2[r];
          } else if (setup->point_type[z][r] == CONTACT_EDGE) {  // adjacent to the contact
            v_sum = (v[new_rb][z+1][r]*dz[1][z][r] + v[new_rb][z][r+1]*dr[1][z][r] +
                    v[new_rb][z-1][r]*dz[0][z][r] + v[new_rb][z][r-1]*dr[0][z][r]);
            eps_sum = dz[1][z][r] + dr[1][z][r] + dz[0][z][r] + dr[0][z][r];
          } else if (setup->point_type[z][r] >= DITCH) {  // in or adjacent to the ditch
            v_sum = (v[new_rb][z+1][r]*eps_dz[z  ][r] + v[new_rb][z][r+1]*eps_dr[z][r  ]*s1[r] +
                    v[new_rb][z-1][r]*eps_dz[z-1][r] + v[new_rb][z][r-1]*eps_dr[z][r-1]*s2[r]);
            eps_sum = (eps_dz[z][r]   + eps_dr[z][r]  *s1[r] +
                      eps_dz[z-1][r] + eps_dr[z][r-1]*s2[r]);
          }

          // calculate the interpolated mean potential and the effect of the space charge
          if ((ev_calc || (setup->vacuum_gap > 0 && z == 1)) &&
              setup->point_type[z][r] < CONTACT_EDGE && r > 1 && z > 1) {   // normal bulk, no complications
            v[new_rb][z][r] = (1.0-OR_fact)*v[old_rb][z][r] + OR_fact * (v_sum / eps_sum + setup->impurity[z][r]);
          } else if (ev_calc || (setup->vacuum_gap > 0 && z == 1)) {
            v[new_rb][z][r] = v_sum / eps_sum + setup->impurity[z][r];
          } else if (setup->point_type[z][r] < CONTACT_EDGE && r > 1 && z > 1) {   // normal bulk, no complications
            v[new_rb][z][r] = (1.0-OR_fact)*v[old_rb][z][r] + OR_fact * v_sum / eps_sum;
          } else {                          // over-relaxation at the edges seems to make things worse
            v[new_rb][z][r] = v_sum / eps_sum;
          }

          // calculate difference from last iteration, for convergence check
          dif = fabs(v[old_rb][z][r] - v[new_rb][z][r]);
          sum_dif += dif;
          if (max_dif < dif) max_dif = dif;
        }
      }
    }


    if (iter < 10 || (iter < 600 && iter%100 == 0) || iter%1000 == 0) {
      if (0 && ev_calc) {
        printf("%5d %d %d %.10f %.10f\n", iter, old_rb, new_rb, max_dif, sum_dif/(L-2)/(R-2));
      } else {
        printf("R-B calculation: iter=%5d old=%d new=%d OR_fact=%f max_dif=%.10f sum_dif/(L-2)/(R-2)=%.10f ; v_center=%.10f v_at_L/3_R/3=%.10f\n",
               iter, old_rb, new_rb, OR_fact, max_dif, sum_dif/(L-2)/(R-2),
               setup->v[new_rb][L/2][R/2], setup->v[new_rb][L/3][R/3]);
      }
    }
    // check for convergence
    if ( ev_calc && max_dif < 0.00000008) break;
    if ( ev_calc && max_dif < 0.0008) break;  // comment out if you want convergence at the numerical error level
    if (!ev_calc && max_dif < 0.0000000001) break;
    if (!ev_calc && max_dif < 0.000001) break;  // comment out if you want convergence at the numerical error level
  }
  printf(">> %d %.16f\n\n", iter, sum_dif);
  if (setup->vacuum_gap > 0) {   // restore impurity value along passivated surface
    for (r = 1; r < R; r++)
      setup->impurity[1][r] = setup->impurity[0][r];
  }
  return 0;
} /* do_relax_rb */

/* -------------------------------------- interpolate ------------------- */
int interpolate_gpu(MJD_Siggen_Setup *setup, MJD_Siggen_Setup *old_setup) {
  int    n, i, j, i2, j2, zmin, rmin, zmax, rmax;
  int    L  = lrint(old_setup->xtal_length/old_setup->xtal_grid)+3;
  int    R  = lrint(old_setup->xtal_radius/old_setup->xtal_grid)+3;
  int    L2 = lrint(setup->xtal_length/setup->xtal_grid)+3;
  int    R2 = lrint(setup->xtal_radius/setup->xtal_grid)+3;
  float  f, f1r, f1z, f2r, f2z;
  double ***v = setup->v, **ov = old_setup->v[1];

  /* the previous calculation was on a coarser grid...
     now copy/expand the potential to the new finer grid
  */
  n = (int) (old_setup->xtal_grid / setup->xtal_grid + 0.5);
  f = 1.0 / (float) n;
  printf("\ngrid %.4f -> %.4f; ratio = %d %.3f\n\n",
         old_setup->xtal_grid, setup->xtal_grid, n, f);
  for (i = i2 = 1; i < L-1; i++) {
    zmin = i2;
    zmax = i2 + n;
    if (zmax > L2-1) zmax = L2-1;
    for (j = j2 = 1; j < R-1; j++) {
      f1z = 0.0;
      rmin = j2;
      rmax = j2 + n;
      if (rmax > R2-1) rmax = R2-1;
      for (i2 = zmin; i2 < zmax; i2++) {
        f2z = 1.0 - f1z;
        f1r = 0.0;
        for (j2 = rmin; j2 < rmax; j2++) {
          f2r = 1.0 - f1r;
          v[0][i2][j2] = v[1][i2][j2] =      // linear interpolation
            f2z*f2r*ov[i][j  ] + f1z*f2r*ov[i+1][j  ] +
            f2z*f1r*ov[i][j+1] + f1z*f1r*ov[i+1][j+1];
          f1r += f;
        }
        f1z += f;
      }
      j2 = rmax;
    }
    i2 = zmax;
  }

  return 0;
} /* interpolate */



/* -------------------------------------- ev_relax_undep ------------------- */
/*  This function, unlike do_relax() above, properly handles undepleted detectors.
    Note that this function uses a modified sequential over-relaxtion algorithm,
    while do_relax() above uses a more standard text-book version.
 */
 int ev_relax_undep_gpu(MJD_Siggen_Setup *setup) {
  int    old = 1, new_undep = 0, iter, r, z, bvn;
  float  grid = setup->xtal_grid;
  int    L  = lrint(setup->xtal_length/grid)+2;
  int    R  = lrint(setup->xtal_radius/grid)+2;
  double eps_sum, v_sum, save_dif, min;
  double dif, sum_dif, max_dif, bubble_volts;
  double ***v = setup->v, **eps_dr = setup->eps_dr, **eps_dz = setup->eps_dz;
  double ***dr = setup->dr, ***dz = setup->dz;
  double *s1 = setup->s1, *s2 = setup->s2;
  char   **undep = setup->undepleted;
  double e_over_E = 11.310; // e/epsilon; for 1 mm2, charge units 1e10 e/cm3, espilon = 16*epsilon0


  // save impurity value along passivated surface
  for (r = 1; r < R; r++)
    setup->impurity[0][r] = setup->impurity[1][r];

  /* initialise the undepleted array for use with bubble depletion */
  for (z = 1; z < L; z++) {
    for (r = 1; r < R; r++) {
      if (setup->point_type[z][r] >= INSIDE) undep[r][z] = 0;
    }
  }

  for (iter = 0; iter < setup->max_iterations; iter++) {

    double OR_fact = ((0.997 - 300.0/((L+1)*(R+1))) * (1.0 - 0.9/(double)(1+iter/6)));
    if (300.0/((L+1)*(R+1)) > 0.5) OR_fact = (0.5 * (1.0 - 0.9/(double)(1+iter/6)));
    if (iter < 2) OR_fact = 0.0;

    old = new_undep;
    new_undep = 1 - new_undep;
    sum_dif = 0;
    max_dif = 0;
    bubble_volts = 0;
    bvn = 0;

    if (setup->vacuum_gap > 0) {   // modify impurity value along passivated surface
      for (r = 1; r < R; r++)      //   due to surface charge induced by capacitance
        setup->impurity[1][r] = setup->impurity[0][r] -
          v[old][1][r] * 5.52e-4 * e_over_E * grid / setup->vacuum_gap;
    }

    /* start from z=1 and r=1 so that (z,r)=0 can be
       used for reflection symmetry around r=0 or z=0 */
    for (z = 1; z < L; z++) {
      /* manage r=0 reflection symmetry */
      setup->v[old][z][0] = setup->v[old][z][2];

      for (r = 1; r < R; r++) {
        if (setup->point_type[z][r] < INSIDE) continue;   // HV or point contact
        save_dif = v[old][z][r] - v[new_undep][z][r];      // step difference from previous iteration

        if (setup->point_type[z][r] < DITCH) {       // normal bulk or passivated surface, no complications
          v_sum = (v[old][z+1][r] + v[old][z][r+1]*s1[r] +
                   v[old][z-1][r] + v[old][z][r-1]*s2[r]);
          if (r > 1) eps_sum = 4;
          else       eps_sum = 2 + s1[r] + s2[r];
        } else if (setup->point_type[z][r] == CONTACT_EDGE) {  // adjacent to the contact
          v_sum = (v[old][z+1][r]*dz[1][z][r] + v[old][z][r+1]*dr[1][z][r] +
                   v[old][z-1][r]*dz[0][z][r] + v[old][z][r-1]*dr[0][z][r]);
          eps_sum = dz[1][z][r] + dr[1][z][r] + dz[0][z][r] + dr[0][z][r];
        } else if (setup->point_type[z][r] >= DITCH) {  // in or adjacent to the ditch
          v_sum = (v[old][z+1][r]*eps_dz[z  ][r] + v[old][z][r+1]*eps_dr[z][r  ]*s1[r] +
                   v[old][z-1][r]*eps_dz[z-1][r] + v[old][z][r-1]*eps_dr[z][r-1]*s2[r]);
          eps_sum = (eps_dz[z][r]   + eps_dr[z][r]  *s1[r] +
                     eps_dz[z-1][r] + eps_dr[z][r-1]*s2[r]);
        }

        // calculate the interpolated mean potential and the effect of the space charge
        min = fminf(fminf(v[old][z+1][r], v[old][z][r+1]),
                    fminf(v[old][z-1][r], v[old][z][r-1]));
        v[new_undep][z][r] = v_sum / eps_sum + setup->impurity[z][r];

        undep[r][z] /= 2;
        if (v[new_undep][z][r] <= 0) {
          v[new_undep][z][r] = 0;
          undep[r][z] = 4;  // do not do over-relaxation for 3 iterations
        } else if (v[new_undep][z][r] <= min) {
          //if (bubble_volts == 0)
            bubble_volts = min + 0.2*grid*grid; // finer grids require smaller increment here
          v[new_undep][z][r] = bubble_volts;
          bvn++;
          undep[r][z] = 8;  // do not do over-relaxation for 4 iterations
        }

        // calculate difference from last iteration, for convergence check
        dif = v[old][z][r] - v[new_undep][z][r];
        if (dif < 0) dif = -dif;
        sum_dif += dif;
        if (max_dif < dif) max_dif = dif;
        // do over-relaxation
        if (!undep[r][z])  v[new_undep][z][r] += OR_fact*save_dif;
      }
    }

    // report results for some iterations
    if (iter < 10 || (iter < 600 && iter%100 == 0) || iter%1000 == 0) {
      if (0) {
        printf("%5d %d %d %.10f %.10f\n", iter, old, new_undep, max_dif, sum_dif/(L-2)/(R-2));
      } else {
        printf("%5d %d %d %.10f %.10f ; %.10f %.10f bubble %.2f %d\n",
               iter, old, new_undep, max_dif, sum_dif/(L-2)/(R-2),
               v[new_undep][L/2][R/2], v[new_undep][L/3][R/3], bubble_volts, bvn);
      }
    }
    // check for convergence
    if (max_dif < 0.00000008) break;
 
  }
  printf(">> %d %.16f\n\n", iter, sum_dif);

  setup->bubble_volts = bubble_volts;
  setup->fully_depleted = 1;
  for (r=1; r<R; r++) {
    for (z=1; z<L; z++) {
      if (setup->point_type[z][r] < INSIDE) {
        undep[r][z] = ' ';
      } else if (undep[r][z] == 0) {
        undep[r][z] = '.';
      } else {
        if (undep[r][z] > 4) undep[r][z] = 'B';  // identifies pinch-off
        else undep[r][z] = '*';
        setup->fully_depleted = 0;
      }
    }
  }

  if (setup->vacuum_gap > 0) {   // restore impurity value along passivated surface
    for (r = 1; r < R; r++)
      setup->impurity[1][r] = setup->impurity[0][r];
  }

  return 0;
} /* ev_relax_undep */


/* -------------------------------------- write_ev ------------------- */
int write_ev_gpu(MJD_Siggen_Setup *setup) {
  int    i, j, new_write_ev=1;
  float  grid = setup->xtal_grid;
  //int    L  = setup->xtal_length/grid + 2.99;
  //int    R  = setup->xtal_radius/grid + 2.99;
  int    L  = lrint(setup->xtal_length/grid)+2;
  int    R  = lrint(setup->xtal_radius/grid)+2;
  float  r, z, E;
  FILE   *file;
  double ***v = setup->v;
  cyl_pt **e;

  setup->Emin = 99999.9;
  setup->rmin = setup->zmin = 999.9;


  if (setup->impurity_z0 > 0) {
    // swap voltages back to negative for n-type material
    for (i=1; i<L; i++) {
      for (j=1; j<R; j++) {
        setup->v[new_write_ev][i][j] = -setup->v[new_write_ev][i][j];
      }
    }
  }

  /* write potential and field to output file */
  if (!(file = fopen(setup->field_name, "w"))) {
    printf("ERROR: Cannot open file %s for electric field...\n", setup->field_name);
    return 1;
  } else {
    printf("Writing electric field data to file %s\n", setup->field_name);
  }
  /* copy configuration parameters to output file */
  report_config_gpu(file, setup->config_file_name);
  fprintf(file, "#\n# HV bias in fieldgen: %.1f V\n", setup->xtal_HV);
  if (setup->fully_depleted) {
    fprintf(file, "# Detector is fully depleted.\n");
  } else {
    fprintf(file, "# Detector is not fully depleted.\n");
    if (setup->bubble_volts > 0.0f)
      fprintf(file, "# Pinch-off bubble at %.0f V potential\n", setup->bubble_volts);
  }
  
  if ((e = (cyl_pt **) malloc((R+1)*sizeof(*e))) == NULL) {
    printf("ERROR: Malloc failed\n");
    fclose(file);
    return 1;
  }
  for (i = 0; i < R; i++){
    if ((e[i] = (cyl_pt *) calloc(L, sizeof(*e[i]))) == NULL) {
      printf("ERROR: Calloc failed\n");
      fclose(file);
      return 1;
    }
  }

  for (j = 1; j < R; j++) {
    r = (j-1) * grid;
    for (i = 1; i < L; i++) {
      z = (i-1) * grid;
      // calc E in r-direction
      if (j == 1) {  // r = 0; symmetry implies E_r = 0
        e[j][i].r = 0;
      } else if (setup->point_type[i][j] == CONTACT_EDGE) {
        e[j][i].r = ((v[new_write_ev][i][j] - v[new_write_ev][i][j+1])*setup->dr[1][i][j] +
               (v[new_write_ev][i][j-1] - v[new_write_ev][i][j])*setup->dr[0][i][j]) / (0.2*grid);
      } else if (setup->point_type[i][j] < INSIDE &&
                 setup->point_type[i][j-1] == CONTACT_EDGE) {
        e[j][i].r =  (v[new_write_ev][i][j-1] - v[new_write_ev][i][j]) * setup->dr[1][i][j-1] / ( 0.1*grid) ;
      } else if (setup->point_type[i][j] < INSIDE &&
                 setup->point_type[i][j+1] == CONTACT_EDGE) {
        e[j][i].r =  (v[new_write_ev][i][j] - v[new_write_ev][i][j+1]) * setup->dr[0][i][j+1] / ( 0.1*grid) ;
      } else if (j == R-1) {
        e[j][i].r = (v[new_write_ev][i][j-1] - v[new_write_ev][i][j])/(0.1*grid);
      } else {
        e[j][i].r = (v[new_write_ev][i][j-1] - v[new_write_ev][i][j+1])/(0.2*grid);
      }
      // calc E in z-direction
      if (setup->point_type[i][j] == CONTACT_EDGE) {
        e[j][i].z = ((v[new_write_ev][i][j] - v[new_write_ev][i+1][j])*setup->dz[1][i][j] +
               (v[new_write_ev][i-1][j] - v[new_write_ev][i][j])*setup->dz[0][i][j]) / (0.2*grid);
      } else if (setup->point_type[i][j] < INSIDE &&
                 setup->point_type[i-1][j] == CONTACT_EDGE) {
        e[j][i].z =  (v[new_write_ev][i-1][j] - v[new_write_ev][i][j]) * setup->dz[1][i-1][j] / ( 0.1*grid) ;
      } else if (setup->point_type[i][j] < INSIDE &&
                 setup->point_type[i+1][j] == CONTACT_EDGE) {
        e[j][i].z =  (v[new_write_ev][i][j] - v[new_write_ev][i+1][j]) * setup->dz[0][i+1][j] / ( 0.1*grid) ;
      } else if (i == 1) {
        e[j][i].z = (v[new_write_ev][i][j] - v[new_write_ev][i+1][j])/(0.1*grid);
      } else if (i == L-1) {
        e[j][i].z = (v[new_write_ev][i-1][j] - v[new_write_ev][i][j])/(0.1*grid);
      } else {
        e[j][i].z = (v[new_write_ev][i-1][j] - v[new_write_ev][i+1][j])/(0.2*grid);
      }

      /* temporarily store E in e[j][i].phi */
      E = e[j][i].phi = sqrt(e[j][i].r*e[j][i].r + e[j][i].z*e[j][i].z);
      /* check for minimum field inside bulk of detector */
      int k = 3.0/grid;
      if (E > 0.1 && E < setup->Emin &&
          i > k+1 && j < R-k-1 && i < L-k-1 &&
          setup->point_type[i][j] == INSIDE &&
          setup->point_type[i + k][j] == INSIDE &&  // point is at least 3 mm from a boundary
          setup->point_type[i - k][j] == INSIDE &&
          setup->point_type[i][j + k] == INSIDE &&
          (j < k+1 || setup->point_type[i][j - k] == INSIDE)) {
        setup->Emin = E;
        setup->rmin = r;
        setup->zmin = z;
      }
    }
  }

  if (strstr(setup->field_name, "unf")) {
    fprintf(file, "#\n## start of unformatted data\n");
    i = R-1; j = L-1;
    fwrite(&i, sizeof(int), 1, file);
    fwrite(&j, sizeof(int), 1, file);
    for (i = 1; i < R; i++) {
      for (j = 1; j < L; j++) e[i][j].phi = 0;
      if (fwrite(&e[i][1], sizeof(cyl_pt), L-1, file) != L-1) {
        printf("ERROR while writing %s\n", setup->field_name);
      }
    }
  } else {
    fprintf(file, "#\n## r (mm), z (mm), V (V),  E (V/cm), E_r (V/cm), E_z (V/cm)\n");
    for (j = 1; j < R; j++) {
      r = (j-1) * grid;
      for (i = 1; i < L; i++) {
        z = (i-1) * grid;
        E = e[j][i].phi;
        fprintf(file, "%7.2f %7.2f %7.1f %7.1f %7.1f %7.1f\n",
                r, z, v[new_write_ev][i][j], E, e[j][i].r, e[j][i].z);
      }
      fprintf(file, "\n");
    }
  }
  fclose(file);
  for (i = 0; i < R; i++) free(e[i]);
  free(e);

  if (!setup->write_WP)
    printf("\n Minimum bulk field = %.2f V/cm at (r,z) = (%.1f, %.1f) mm\n\n",
           setup->Emin, setup->rmin, setup->zmin);

  if (0) { /* write point_type to output file */
    file = fopen("fields/point_type.dat", "w");
    for (j = 1; j < R; j++) {
      for (i = 1; i < L; i++)
        fprintf(file, "%7.2f %7.2f %2d\n",
                (j-1)*grid, (i-1)*grid, setup->point_type[i][j]);
      fprintf(file, "\n");
    }
    fclose(file);
  }

  return 0;
 } /* write_ev */

 /* -------------------------------------- report_config_gpu ------------------- */
int report_config_gpu(FILE *fp_out, char *config_file_name) {
  char  *c, line[256];
  FILE  *file;

  fprintf(fp_out, "# Config file: %s\n", config_file_name);
  if (!(file = fopen(config_file_name, "r"))) return 1;

  while (fgets(line, sizeof(line), file)) {
    if (strlen(line) < 3 || *line == ' ' || *line == '\t' || *line == '#') continue;
    if ((c = strchr(line, '#')) || (c = strchr(line, '\n'))) *c = '\0';
    fprintf(fp_out, "# %s\n", line);
  }
  fclose(file);
  return 0;
} /* report_config_gpu */