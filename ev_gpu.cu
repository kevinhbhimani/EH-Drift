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
Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
Max dimension size of a grid size (x,y,z): (65535, 65535, 65535)
*/

extern "C" int ev_calc_gpu(MJD_Siggen_Setup *setup);
int do_relax_gpu(MJD_Siggen_Setup *setup, int ev_calc);



/* -------------------------------------- ev_calc_gpu ------------------- */
extern "C" int ev_calc_gpu(MJD_Siggen_Setup *setup) {
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

  do_relax_gpu(setup, 1);
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
} /* ev_calc_gpu */


/* -------------------------------------- do_relax_gpu ------------------- */
int do_relax_gpu(MJD_Siggen_Setup *setup, int ev_calc) {
  int    old = 1, new_gpu = 0, iter, r, z;
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

    old = new_gpu;
    new_gpu = 1 - new_gpu;
    sum_dif = 0;
    max_dif = 0;

    if (setup->vacuum_gap > 0) {   // modify impurity value along passivated surface
      for (r = 1; r < R; r++)      //   due to surface charge induced by capacitance
        setup->impurity[1][r] = setup->impurity[0][r] -
          v[old][1][r] * 5.52e-4 * e_over_E * grid / setup->vacuum_gap;
    }

    /* start from z=1 and r=1 so that (z,r)=0 can be
       used for reflection symmetry around r=0 or z=0 */
    for (z = 1; z < L; z++) {
      /* manage r=0 reflection symmetry */
      setup->v[old][z][0] = setup->v[new_gpu][z][0] = setup->v[old][z][2];

      for (r = 1; r < R; r++) {
        if (setup->point_type[z][r] < INSIDE) continue;   // HV or point contact

        if (setup->point_type[z][r] < DITCH) {       // normal bulk or passivated surface, no complications
          v_sum = (v[old][z+1][r] + v[old][z][r+1]*s1[r] +
                   v[new_gpu][z-1][r] + v[new_gpu][z][r-1]*s2[r]);
          if (r > 1) eps_sum = 4;
          else       eps_sum = 2 + s1[r] + s2[r];
        } else if (setup->point_type[z][r] == CONTACT_EDGE) {  // adjacent to the contact
          v_sum = (v[old][z+1][r]*dz[1][z][r] + v[old][z][r+1]*dr[1][z][r] +
                   v[new_gpu][z-1][r]*dz[0][z][r] + v[new_gpu][z][r-1]*dr[0][z][r]);
          eps_sum = dz[1][z][r] + dr[1][z][r] + dz[0][z][r] + dr[0][z][r];
        } else if (setup->point_type[z][r] >= DITCH) {  // in or adjacent to the ditch
          v_sum = (v[old][z+1][r]*eps_dz[z  ][r] + v[old][z][r+1]*eps_dr[z][r  ]*s1[r] +
                   v[new_gpu][z-1][r]*eps_dz[z-1][r] + v[new_gpu][z][r-1]*eps_dr[z][r-1]*s2[r]);
          eps_sum = (eps_dz[z][r]   + eps_dr[z][r]  *s1[r] +
                     eps_dz[z-1][r] + eps_dr[z][r-1]*s2[r]);
        }

        // calculate the interpolated mean potential and the effect of the space charge

        if ((ev_calc || (setup->vacuum_gap > 0 && z == 1)) &&
            setup->point_type[z][r] < CONTACT_EDGE && r > 1 && z > 1) {   // normal bulk, no complications
          v[new_gpu][z][r] = (1.0-OR_fact)*v[old][z][r] + OR_fact * (v_sum / eps_sum + setup->impurity[z][r]);
        } else if (ev_calc || (setup->vacuum_gap > 0 && z == 1)) {
          v[new_gpu][z][r] = v_sum / eps_sum + setup->impurity[z][r];
        } else if (setup->point_type[z][r] < CONTACT_EDGE && r > 1 && z > 1) {   // normal bulk, no complications
          v[new_gpu][z][r] = (1.0-OR_fact)*v[old][z][r] + OR_fact * v_sum / eps_sum;
        } else {                          // over-relaxation at the edges seems to make things worse
          v[new_gpu][z][r] = v_sum / eps_sum;
        }

        // calculate difference from last iteration, for convergence check
        dif = fabs(v[old][z][r] - v[new_gpu][z][r]);
        sum_dif += dif;
        if (max_dif < dif) max_dif = dif;
      }
    }

    // report results for some iterations
    if (iter < 10 || (iter < 600 && iter%100 == 0) || iter%1000 == 0) {
      if (0 && ev_calc) {
        printf("%5d %d %d %.10f %.10f\n", iter, old, new_gpu, max_dif, sum_dif/(L-2)/(R-2));
      } else {
        printf("%5d %d %d %.10f %.10f ; %.10f %.10f\n",
               iter, old, new_gpu, max_dif, sum_dif/(L-2)/(R-2),
               v[new_gpu][L/2][R/2], v[new_gpu][L/3][R/3]);
      }
    }
    // check for convergence
    if ( ev_calc && max_dif < 0.00000008) break;
    if ( ev_calc && max_dif < 0.0008) break;  // comment out if you want convergence at the numerical error level
    if (!ev_calc && max_dif < 0.0000000001) break;
    if (!ev_calc && max_dif < 0.000001) break;  // comment out if you want convergence at the numerical error level

    /* every 100 iterations, check that detector is really depleted*/
    /*
    if (ev_calc && iter > 190 && iter%100 == 0) {
      for (z = 1; z < L; z++) {
        setup->v[old][z][0] = setup->v[old][z][2];
        for (r = 1; r < R; r++) {
          if (setup->point_type[z][r] < INSIDE) continue;   // HV or point contact
          if (v[new_gpu][z][r] < 0 ||
              (v[new_gpu][z][r] < v[new_gpu][z][r] &&
               v[new_gpu][z][r] < v[new_gpu][z][r] &&
               v[new_gpu][z][r] < v[new_gpu][z][r] &&
               v[new_gpu][z][r] < v[new_gpu][z][r])) {
            printf("Detector may not be fully depleted. Switching to ev_relax_undep()\n");
            ev_relax_undep(setup);
            return 0;
          }
        }
      }
    }
    */
  }

  printf(">> %d %.16f\n\n", iter, sum_dif);
  if (setup->vacuum_gap > 0) {   // restore impurity value along passivated surface
    for (r = 1; r < R; r++)
      setup->impurity[1][r] = setup->impurity[0][r];
  }

  return 0;
} /* do_relax_gpu */