/* program to evolve e and h charge distributions from surface-alpha interactions

   Calculates self-consistent electric field at each time step
   Uses a 2d approximation, so only (r,z), where the charge could is simulated
     by a ring, with phi symmetry
   The diffusion speed was checked to make sure it roughly matches known values
     for 3D charge clouds
   The total charge created was reduced to make the self-repulsion effects
     roughly match previous simulations for alpha particle interactions

   author:           D.C. Radford
   first written:    Oct 2020

   ***** Places where some options may be hardcoded are flagged with // CHANGEME comments *****

   Grid size and time step are taken as specified in detector config file
   Unknowns/guesses:
     - Surface charge density (specified in config file)
         ~ With the point-charge simulations, we adjusted the surface charges to
           approximately reproduce the observed effects
         ~ Those numbers will now need to be changed
     - Surface drift speed (ratio specified in config file)
         ~ Again, we adjusted the previous surface velocity factors to reproduce
           observations
     - Surface diffusion speed
         ~ Should radial diffusion along the surface be reduced by the same factor
           as the velocity?
         ~ But transverse diffusion will dominate and tend to dramatically change
           the radial velocity. By what factor should that be reduced?
     - Effective passivated-surface thickness
         ~ The passivation layer is typically 0.1 μm thick
         ~ But surface roughness and damage from the passivation process could be
           2-10 μm thick
  Can be run as ./ehdrift config_files/P42575A.config -a 25.00 -z 0.10 -g P42575A -s 0.00
  WP can be calculated as ./ehdrift config_files/P42575A_calc_wp.config -a 15.00 -z 0.10 -g P42575A -s 0.00
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "mjd_siggen.h"
#include "detector_geometry.h"
#include "gpu_vars.h"

#include <cuda.h>
#include <cuda_runtime.h>


#define MAX_ITS 50000     // default max number of iterations for relaxation

int report_config(FILE *fp_out, char *config_file_name);
int grid_init(MJD_Siggen_Setup *setup);
int ev_calc(MJD_Siggen_Setup *setup, MJD_Siggen_Setup *old_setup);
int ev_calc2(MJD_Siggen_Setup *setup);
int wp_calc(MJD_Siggen_Setup *setup, MJD_Siggen_Setup *old_setup);
int write_ev(MJD_Siggen_Setup *setup);
int write_wp(MJD_Siggen_Setup *setup);
int do_relax(MJD_Siggen_Setup *setup, int ev_calc);
int ev_relax_undep(MJD_Siggen_Setup *setup);
int wp_relax_undep(MJD_Siggen_Setup *setup);
int interpolate(MJD_Siggen_Setup *setup, MJD_Siggen_Setup *old_setup);

int write_rho(int L, int R, float grid, float **rho, char *fname);
int drift_rho(MJD_Siggen_Setup *setup, int L, int R, float grid, float ***rho,
              int q, double *gone);
int read_rho(int L, int R, float grid, float **rho, char *fname);

int gpu_drift(MJD_Siggen_Setup *setup, int L, int R, float grid, float ***rho, int q, GPU_data *gpu_setup);
void set_rho_zero_gpu(GPU_data *gpu_setup, int L, int R, int num_blocks, int num_threads);
void update_impurities_gpu(GPU_data *gpu_setup, int L, int R, int num_blocks, int num_threads, double e_over_E, float grid);

/* -------------------------------------- main ------------------- */
int main(int argc, char **argv)
{
  int write_densities = 1;

  MJD_Siggen_Setup setup, setup1, setup2;
  GPU_data gpu_setup;

  float BV;      // bias voltage
  int   WV = 0;  // 0: do not write the V and E values to ppc_ev.dat
                 // 1: write the V and E values to ppc_ev.dat
                 // 2: write the V and E values for both +r, -r (for gnuplot, NOT for siggen)
  int   WD = 0;  // 0: do not write out depletion surface
                 // 1: write out depletion surface to depl_<HV>.dat

  int   i, j;
  FILE  *fp;
  double e_over_E = 11.310; // e/epsilon; for 1 mm2, charge units 1e10 e/cm3, espilon = 16*epsilon0
  float **rho_e[4], **rho_h[3];
  double egone=0, hgone=0;
  float  alpha_r_mm = 10.0;  // impact radius of alpha on passivated surface; change with -a option
  float  alpha_z_mm = 0.1;
  char det_name[8];
  if (argc < 2 || argc%2 != 0 || read_config(argv[1], &setup)) {
    printf("Usage: %s <config_file_name> [options]\n"
           "   Possible options:\n"
	   "      -b bias_volts\n"
	   "      -w {0,1}  (do_not/do write the field file)\n"
	   "      -d {0,1}  (do_not/do write the depletion surface)\n"
	   "      -p {0,1}  (do_not/do write the WP file)\n"
	   "      -a <alpha radius in mm>\n"
     "      -z <z position in mm>\n"
     "      -r rho_spectrum_file_name\n", argv[0]);
    return 1;
  }
  strncpy(setup.config_file_name, argv[1], sizeof(setup.config_file_name));

  if (setup.xtal_grid < 0.001) setup.xtal_grid = 0.5;
  BV = setup.xtal_HV;
  WV = setup.write_field;
  setup.rho_z_spe[0] = 0;

  for (i=2; i<argc-1; i++) {
    if (strstr(argv[i], "-b")) {
      BV = setup.xtal_HV = atof(argv[++i]);   // bias volts
    } else if (strstr(argv[i], "-w")) {
      WV = atoi(argv[++i]);               // write-out options
    } else if (strstr(argv[i], "-d")) {
      WD = atoi(argv[++i]);               // write-out options
    } else if (strstr(argv[i], "-p")) {
      setup.write_WP = atoi(argv[++i]);   // weighting-potential options
    } else if (strstr(argv[i], "-a")) {
      alpha_r_mm = atof(argv[++i]);       // alpha impact radius
    } else if (strstr(argv[i], "-z")) {
      alpha_z_mm = atof(argv[++i]);       // alpha impact z position
    } else if (strstr(argv[i], "-g")) {
      strcpy(det_name, argv[++i]);        // name of the detector
    } else if (strstr(argv[i], "-s")) {
      setup.impurity_surface =  atof(argv[++i]);  // surface charge
    } else if (strstr(argv[i], "-r")) {
      if (!(fp = fopen(argv[++i], "r"))) {   // impurity-profile-spectrum file name
        printf("\nERROR: cannot open impurity profile spectrum file %s\n\n", argv[i+1]);
        return 1;
      }
      fread(setup.rho_z_spe, 36, 1, fp);
      for (j=0; j<1024; j++) setup.rho_z_spe[i] = 0;
      fread(setup.rho_z_spe, sizeof(setup.rho_z_spe), 1, fp);
      fclose(fp);
      printf(" z(mm)   rho\n");
      for (j=0; j < 200 && setup.rho_z_spe[j] != 0.0f; j++)
        printf(" %3d  %7.3f\n", j, setup.rho_z_spe[j]);
    } else {
      printf("Possible options:\n"
	     "      -b bias_volts\n"
	     "      -w {0,1,2} (for WV options)\n"
	     "      -p {0,1}   (for WP options)\n"
             "      -r rho_spectrum_file_name\n");
      return 1;
    }
  }
 /*
  if (setup.xtal_length/setup.xtal_grid * setup.xtal_radius/setup.xtal_grid > 2500*2500) {
    printf("Error: Crystal size divided by grid size is too large!\n");
    return 1;
  }
  */
  if (WV < 0 || WV > 2) WV = 0;

  /* -------------- give details of detector geometry */
  if (setup.verbosity >= CHATTY) {
    printf("\n\n"
           "      Crystal: Radius x Length: %.1f x %.1f mm\n",
	   setup.xtal_radius, setup.xtal_length);
    if (setup.hole_length > 0) {
      if (setup.inner_taper_length > 0)
        printf("    Core hole: Radius x length: %.1f x %.1f mm,"
               " taper %.1f x %.1f mm (%2.f degrees)\n",
               setup.hole_radius, setup.hole_length,
               setup.inner_taper_width, setup.inner_taper_length, setup.taper_angle);
      else
        printf("    Core hole: Radius x length: %.1f x %.1f mm\n",
               setup.hole_radius, setup.hole_length);
    }
    printf("Point contact: Radius x length: %.1f x %.1f mm\n",
           setup.pc_radius, setup.pc_length);
    if (setup.ditch_depth > 0) {
      printf("  Wrap-around: Radius x ditch x gap:  %.1f x %.1f x %.1f mm\n",
             setup.wrap_around_radius, setup.ditch_depth, setup.ditch_thickness);
    }
    printf("         Bias: %.0f V\n", BV);
  }
    
  if ((BV < 0 && setup.impurity_z0 < 0) || (BV > 0 && setup.impurity_z0 > 0)) {
    printf("ERROR: Expect bias and impurity to be opposite sign!\n");
    return 1;
  } 
  if (setup.impurity_z0 > 0) {
    // swap polarity for n-type material; this lets me assume all voltages are positive
    BV = -BV;
    setup.xtal_HV *= -1.0;
    setup.impurity_z0         *= -1.0;
    setup.impurity_gradient   *= -1.0;
    setup.impurity_quadratic  *= -1.0;
    setup.impurity_surface    *= -1.0;
    setup.impurity_radial_add *= -1.0;
  }
  /* use an adaptive grid; start out coarse and then refine the grid */
  memcpy(&setup1, &setup, sizeof(setup));
  memcpy(&setup2, &setup, sizeof(setup));
  setup1.xtal_grid *= 9.0;
  setup2.xtal_grid *= 3.0;
  if (grid_init(&setup1) != 0 ||
      grid_init(&setup2) != 0 ||
      grid_init(&setup)  != 0) {
    printf("failed to init field calculations\n");
    return 1;
  }


  /* add electrons and holes at a specific point location near passivated surface */
  double e_dens, esum1=0, esum2=0, ecentr=0, ermsr=0, ecentz=0, ermsz=0;
  double hsum1=0, hsum2=0, hcentr=0, hrmsr=0, hcentz=0, hrmsz=0;
  float  grid = setup.xtal_grid;
  int    L  = lrint(setup.xtal_length/grid)+2;
  int    LL = lrint(setup.xtal_length/grid)+2;
  // int    LL = L/8;
  int    R  = lrint(setup.xtal_radius/grid)+2;
  int    r, z, rr,zz, k;


  /* malloc and clear space for electron density arrays */
  for (j=0; j<4; j++) {
    if ((rho_e[j] = malloc(LL * sizeof(*rho_e[j])))   == NULL) {
      printf("malloc failed\n");
      return -1;
    }
    for (i = 0; i < LL; i++) {
      if ((rho_e[j][i] = malloc(R * sizeof(**rho_e[j])))   == NULL) {
        printf("malloc failed\n");
        return -1;
      }
    }
  }
  for (i = 0; i < LL; i++) {
    for (j = 0; j < R; j++) {
      rho_e[0][i][j] = rho_e[1][i][j] = rho_e[2][i][j] = 0;
      rho_e[3][i][j] = setup.impurity[i][j]; // save impurity values for use later
    }
  }
  /* malloc and clear space for hole density arrays */
  for (j=0; j<3; j++) {
    if ((rho_h[j] = malloc(LL * sizeof(*rho_h[j])))   == NULL) {
      printf("malloc failed\n");
      return -1;
    }
    for (i = 0; i < LL; i++) {
      if ((rho_h[j][i] = malloc(R * sizeof(**rho_h[j])))   == NULL) {
        printf("malloc failed\n");
        return -1;
      }
    }
  }
  for (i = 0; i < LL; i++) {
    for (j = 0; j < R; j++) {
      rho_h[0][i][j] = rho_h[1][i][j] = rho_h[2][i][j] = 0;
    }
  }

  /* add electrons and holes */
  e_dens = 5.0e-7/0.003 / (grid*grid*grid); // / 1000.0);  // 5 MeV, units of 1e10/cm3
  e_dens *= 20.0;                           // these numbers are fudged to get the self-repulsion about right (2D vs. 3D)
  r = alpha_r_mm/grid + 1;  // CHANGEME : starting radius, converted from mm to grid points; see -a command line option
  z = alpha_z_mm/grid + 1; // CHANGEME currently z = 0.1 mm

  /* CHANGEME?
       at this point, you can either read in some starting charge distribution
       or just put charge at the surface, at some specified radius r
   */
  if (1) {  // CHANGEME : change 1 to 0 if you want to read in some starting charge
            //              distribution, e.g. to continue a previous calculation
    // use initial alpha interaction at (r, z~0) as starting distribution
    rho_e[0][z][r] = rho_e[0][z+1][r] = e_dens/2.0;
    rho_h[0][z][r] = rho_h[0][z+1][r] = e_dens/2.0;
  } else {
    // read starting rho_e and rho_h values
    if (read_rho(LL, R, grid, rho_e[0], "start_ed.dat") ||
        read_rho(LL, R, grid, rho_h[0], "start_hd.dat")) return 1;
  }
  
  for (zz=1; zz<LL; zz++) {
    for (rr=1; rr<R; rr++) {
      esum1 += rho_e[0][zz][rr] * (double) rr;
      esum2 += rho_e[2][zz][rr] * (double) rr;
      ecentr += rho_e[0][zz][rr] * (double) (rr * rr);
      ecentz += rho_e[0][zz][rr] * (double) (rr * zz);
      ermsr += rho_e[0][zz][rr] * (double) (rr * rr * rr);
      ermsz += rho_e[0][zz][rr] * (double) (rr * zz * zz) ;
      hsum1 += rho_h[0][zz][rr] * (double) rr;
      hsum2 += rho_h[2][zz][rr] * (double) rr;
      hcentr += rho_h[0][zz][rr] * (double) (rr * rr);
      hcentz += rho_h[0][zz][rr] * (double) (rr * zz);
      hrmsr += rho_h[0][zz][rr] * (double) (rr * rr * rr);
      hrmsz += rho_h[0][zz][rr] * (double) (rr * zz * zz) ;
      setup.impurity[zz][rr] = rho_e[3][zz][rr] +
        (rho_h[0][zz][rr] - rho_e[0][zz][rr]) * e_over_E * grid*grid/2.0;
    }
  }
  printf("n: %3d  esums: %.0f %.0f", 0, esum1, esum2);
  ecentr /= esum1;
  ermsr -= esum1 * ecentr*ecentr;
  ermsr /= esum1;
  ecentz /= esum1;
  ermsz -= esum1 * ecentz*ecentz;
  ermsz /= esum1;
  printf(" |  ecentr,z: %.2f %.2f   ermsr,z:  %.2f %.2f\n",
         grid*(ecentr-1.0), grid*(ecentz-1.0), grid*sqrt(ermsr), grid*sqrt(ermsz));

  printf("n: %3d  hsums: %.0f %.0f", 0, hsum1, hsum2);
  hcentr /= hsum1;
  hrmsr -= hsum1 * hcentr*hcentr;
  hrmsr /= hsum1;
  hcentz /= hsum1;
  hrmsz -= hsum1 * hcentz*hcentz;
  hrmsz /= hsum1;
  printf(" |  hcentr,z: %.2f %.2f   hrmsr,z:  %.2f %.2f\n\n",
         grid*(hcentr-1.0), grid*(hcentz-1.0), grid*sqrt(hrmsr), grid*sqrt(hrmsz));

  /* -------------- calculate electric potential/field */
  if (setup.write_field) {
    setup1.write_field = 0; // no need to save intermediate calculations
    setup2.write_field = 0;
    if (setup.xtal_grid > 0.4) {
      ev_calc_gpu_initial(&setup2, NULL, &gpu_setup);
    } else {
      ev_calc_gpu_initial(&setup1, NULL, &gpu_setup);
      ev_calc_gpu_initial(&setup2, &setup1, &gpu_setup);
    }
    ev_calc_gpu_initial(&setup, &setup2, &gpu_setup);
  }

   /* -------------- calculate weighting potential */
  if (setup.write_WP) {
    setup1.write_WP = 0; // no need to save intermediate calculations
    setup2.write_WP = 0;
    if (setup.xtal_grid > 0.4) {
      wp_calc(&setup2, NULL);
    } else {
      wp_calc(&setup1, NULL);
      wp_calc(&setup2, &setup1);
    }
    wp_calc(&setup, &setup2);
  }

  /* -------------- calculate capacitance
     1/2 * epsilon * integral(E^2) = 1/2 * C * V^2
     so    C = epsilon * integral(E^2) / V^2    V = 1 volt
  */
  double esum, pi=3.14159, Epsilon=(8.85*16.0/1000.0);  // permittivity of Ge in pF/mm
  float  E_r, E_z;
  int    test;
  int    LC = lrint(setup.pc_length/grid)+1;
  int    RC = lrint(setup.pc_radius/grid)+1;

  if (setup.write_WP) {
    esum = esum2 = test = 0;
    for (z=1; z<L-1; z++) {
      for (r=2; r<R-1; r++) {
        E_r = setup.eps_dr[z][r]/16.0 * (setup.v[1][z][r] - setup.v[1][z][r+1])/(0.1*grid);
        E_z = setup.eps_dz[z][r]/16.0 * (setup.v[1][z][r] - setup.v[1][z+1][r])/(0.1*grid);
        esum += (E_r*E_r + E_z*E_z) * (double) (r-1);
        if ((r == RC   && z <= LC)   || (r <= RC   && z == LC)   ||
            (r == RC+1 && z <= LC+1) || (r <= RC+1 && z == LC+1)) { // average over two different surfaces
          if (setup.point_type[z+1][r+1] == PC) test = 1;
          esum2 += 0.5 * sqrt(E_r*E_r + E_z*E_z) * (double) (r-1);  // 0.5 since averaging over 2 surfaces
        }
      }
    }
    esum  *= 2.0 * pi * 0.01 * Epsilon * pow(grid, 3.0);
    // Epsilon is in pF/mm
    // 0.01 converts (V/cm)^2 to (V/mm)^2, pow() converts to grid^3 to mm3
    esum2 *= 2.0 * pi * 0.1 * Epsilon * pow(grid, 2.0);
    // 0.1 converts (V/cm) to (V/mm),  grid^2 to  mm2
    printf("  >>  Calculated capacitance at %.0f V: %.3lf pF\n", BV, esum);
    if (!test)
      printf("  >>  Alternative calculation of capacitance: %.3lf pF\n", esum2);
  }

  /* -------------- estimate depletion voltage */
  double min = BV, min2 = BV, dV, dW, testv;
  int    vminr=0, vminz=0;
  int    dz[4] = {1, -1, 0, 0}, dr[4] = {0, 0, 1, -1};
  if (setup.write_WP) {
    if (setup.fully_depleted) {
      // find minimum potential
      for (z=1; z<LC+2; z++) {
        for (r=1; r<RC+2; r++) {
          if (setup.vsave[z][r] > 0 &&
              min > setup.vsave[z][r] / (1.0 - setup.v[1][z][r])) {
            min = setup.vsave[z][r] / (1.0 - setup.v[1][z][r]);
          }
        }
      }
      /* check for bubble depletion / pinch-off by seeing how much the bias
         must be reduced for any pixel to be in a local potential minimum  */
      for (z=LC+2; z<L-3; z++) {
        for (r=1; r<R-3; r++) {
          if (setup.point_type[z][r] == INSIDE && setup.v[1][z][r] > 0.0001) {
            testv = -1;
            for (i=0; i<4; i++) {
              if (r==1 && i==2) break;  // do not check dr for r=1 (=0.0)
              dV = setup.vsave[z+dz[i]][r+dr[i]]  - setup.vsave[z][r];  // potential
              dW = setup.v[1][z+dz[i]][r+dr[i]]   - setup.v[1][z][r];   // WP
              if (dW*grid > 0.00001 && dV < 0 && testv < -dV/dW) testv = -dV/dW;
            }
            if (testv >= 0 && min2 > testv) {
              min2 = testv;
              vminr = r; vminz = z;
            }
          }
        }
      }
      if (min2 < min) {
        printf("Estimated pinch-off voltage = %.0f V\n", BV - min);
        printf(" min2 = %.1f at (r,z) = (%.1f, %.1f), so\n",
               min2, (vminr-1)*grid, (vminz-1)*grid);
        printf("   Full depletion (max pinch-off voltage) = %.0f\n", BV - min2);
      } else {
        printf("Estimated depletion voltage = %.0f V\n", BV - min);
      }
    }

    printf("Minimum bulk field = %.2f V/cm at (r,z) = (%.1f, %.1f) mm\n\n",
           setup.Emin, setup.rmin, setup.zmin);
  }

  if (setup.write_WP) return 0; //CHANGED : Why are we quiting here?
    /* we need **v to have electric potential, not WP, so we quit here */

    if(write_densities){
    //get_densities(L, R, rho_e, rho_h, &gpu_setup);
    char fn_1[256], fn_2[256];
    sprintf(fn_1, "/pine/scr/k/b/kbhimani/siggen_sims/%s/q=%.2f/drift_data_r=%.2f_z=%.2f/ed000.dat", det_name, setup.impurity_surface, alpha_r_mm, alpha_z_mm);
    sprintf(fn_2, "/pine/scr/k/b/kbhimani/siggen_sims/%s/q=%.2f/drift_data_r=%.2f_z=%.2f/hd000.dat", det_name, setup.impurity_surface, alpha_r_mm, alpha_z_mm);
    write_rho(LL/8, R, grid, rho_e[0], fn_1);
    write_rho(LL/8, R, grid, rho_h[0], fn_2);
  }

  gpu_init(&setup, rho_e, rho_h, &gpu_setup);
  gpu_drift(&setup, L, R, grid, rho_e, -1, &gpu_setup);
  gpu_drift(&setup, L, R, grid, rho_h, 1, &gpu_setup);

  /* -----------------------------------------
   *   This loop starting here is crucial.
   *   It loops over time steps (size = time_steps_calc in the config file) calculating
   *    the self-consistent field and letting the charge densities diffuse and drift.
   * ----------------------------------------- */
  int n;
  int num_threads = 300;
  int num_blocks = R * (ceil(LL/num_threads)+1);

  for (n=1; n<=4000; n++) {   // CHANGEME : 4000 time steps of size time_steps_calc (0.02) thus simulating 800ns
    
    printf("\n\n -=-=-=-=-=-=-=-=-=-=-=- n = %3d  -=-=-=-=-=-=-=-=-=-=-=-\n\n", n);
    cudaDeviceSynchronize();
    set_rho_zero_gpu(&gpu_setup, LL, R, num_blocks, num_threads);
    cudaDeviceSynchronize();
    update_impurities_gpu(&gpu_setup, LL, R, num_blocks, num_threads, e_over_E, grid);

    // get_densities(L, R, rho_e, rho_h, &gpu_setup);
    // // copy new values of rho_e
    // esum1 = esum2 = ecentr = ermsr = ecentz = ermsz = 0;
    // hsum1 = hsum2 = hcentr = hrmsr = hcentz = hrmsz = 0;
    // for (z=1; z<LL; z++) {
    //   for (r=1; r<R; r++) {
    //     // rho_e[0][z][r] = rho_e[2][z][r];
    //     // rho_h[0][z][r] = rho_h[2][z][r];
    //     esum1 += rho_e[0][z][r] * (double) r;
    //     esum2 += rho_e[0][z][r] * (double) r;
    //     hsum1 += rho_h[0][z][r] * (double) r;
    //     hsum2 += rho_h[0][z][r] * (double) r;
    //     ecentr += rho_e[0][z][r] * (double) (r * r);
    //     ecentz += rho_e[0][z][r] * (double) (r * z);
    //     ermsr += rho_e[0][z][r] * (double) (r * r * r);
    //     ermsz += rho_e[0][z][r] * (double) (r * z * z) ;
    //     hcentr += rho_h[0][z][r] * (double) (r * r);
    //     hcentz += rho_h[0][z][r] * (double) (r * z);
    //     hrmsr += rho_h[0][z][r] * (double) (r * r * r);
    //     hrmsz += rho_h[0][z][r] * (double) (r * z * z) ;
    //     // setup.impurity[z][r] = rho_e[3][z][r] +
    //     //   (rho_h[0][z][r] - rho_e[0][z][r]) * e_over_E * grid*grid/2.0;
    //   }
    // }

    // printf("n: %3d  esums: %.0f %.0f", n, esum1, esum2);
    // ecentr /= esum2;
    // ermsr -= esum2 * ecentr*ecentr;
    // ermsr /= esum2;
    // ecentz /= esum2;
    // ermsz -= esum2 * ecentz*ecentz;
    // ermsz /= esum2;
    // printf(" |  ecentr,z: %.2f %.2f   ermsr,z:  %.2f %.2f\n",
    //        grid*(ecentr-1.0), grid*(ecentz-1.0), grid*sqrt(ermsr), grid*sqrt(ermsz));

    // printf("n: %3d  hsums: %.0f %.0f", n, hsum1, hsum2);
    // hcentr /= hsum2;
    // hrmsr -= hsum2 * hcentr*hcentr;
    // hrmsr /= hsum2;
    // hcentz /= hsum2;
    // hrmsz -= hsum2 * hcentz*hcentz;
    // hrmsz /= hsum2;
    // printf(" |  hcentr,z: %.2f %.2f   hrmsr,z:  %.2f %.2f\n\n",
    //        grid*(hcentr-1.0), grid*(hcentz-1.0), grid*sqrt(hrmsr), grid*sqrt(hrmsz));

    cudaDeviceSynchronize();
    ev_calc_gpu(1, &setup, &gpu_setup);

    cudaDeviceSynchronize();
    gpu_drift(&setup, L, R, grid, rho_e, -1, &gpu_setup);
    cudaDeviceSynchronize();
    gpu_drift(&setup, L, R, grid, rho_h, 1, &gpu_setup);

    if (write_densities && n%10 == 0) {
      cudaDeviceSynchronize();
      get_densities(L, R, rho_e, rho_h, &gpu_setup);
          // copy new values of rho_e
      esum1 = esum2 = ecentr = ermsr = ecentz = ermsz = 0;
      hsum1 = hsum2 = hcentr = hrmsr = hcentz = hrmsz = 0;
      for (z=1; z<LL; z++) {
        for (r=1; r<R; r++) {
          // rho_e[0][z][r] = rho_e[2][z][r];
          // rho_h[0][z][r] = rho_h[2][z][r];
          esum1 += rho_e[0][z][r] * (double) r;
          esum2 += rho_e[0][z][r] * (double) r;
          hsum1 += rho_h[0][z][r] * (double) r;
          hsum2 += rho_h[0][z][r] * (double) r;
          ecentr += rho_e[0][z][r] * (double) (r * r);
          ecentz += rho_e[0][z][r] * (double) (r * z);
          ermsr += rho_e[0][z][r] * (double) (r * r * r);
          ermsz += rho_e[0][z][r] * (double) (r * z * z) ;
          hcentr += rho_h[0][z][r] * (double) (r * r);
          hcentz += rho_h[0][z][r] * (double) (r * z);
          hrmsr += rho_h[0][z][r] * (double) (r * r * r);
          hrmsz += rho_h[0][z][r] * (double) (r * z * z) ;
          // setup.impurity[z][r] = rho_e[3][z][r] +
          //   (rho_h[0][z][r] - rho_e[0][z][r]) * e_over_E * grid*grid/2.0;
        }
      }

      printf("n: %3d  esums: %.0f %.0f", n, esum1, esum2);
      ecentr /= esum2;
      ermsr -= esum2 * ecentr*ecentr;
      ermsr /= esum2;
      ecentz /= esum2;
      ermsz -= esum2 * ecentz*ecentz;
      ermsz /= esum2;
      printf(" |  ecentr,z: %.2f %.2f   ermsr,z:  %.2f %.2f\n",
            grid*(ecentr-1.0), grid*(ecentz-1.0), grid*sqrt(ermsr), grid*sqrt(ermsz));

      printf("n: %3d  hsums: %.0f %.0f", n, hsum1, hsum2);
      hcentr /= hsum2;
      hrmsr -= hsum2 * hcentr*hcentr;
      hrmsr /= hsum2;
      hcentz /= hsum2;
      hrmsz -= hsum2 * hcentz*hcentz;
      hrmsz /= hsum2;
      printf(" |  hcentr,z: %.2f %.2f   hrmsr,z:  %.2f %.2f\n\n",
            grid*(hcentr-1.0), grid*(hcentz-1.0), grid*sqrt(hrmsr), grid*sqrt(hrmsz));

      char fn[256];
      sprintf(fn, "/pine/scr/k/b/kbhimani/siggen_sims/%s/q=%.2f/drift_data_r=%.2f_z=%.2f/ed%3.3d.dat", det_name, setup.impurity_surface, alpha_r_mm, alpha_z_mm, n/10);
      if (esum2 > 0.1) write_rho(LL/8, R, grid, rho_e[0], fn);
      sprintf(fn, "/pine/scr/k/b/kbhimani/siggen_sims/%s/q=%.2f/drift_data_r=%.2f_z=%.2f/hd%3.3d.dat", det_name, setup.impurity_surface, alpha_r_mm, alpha_z_mm, n/10);
      if (hsum2 > 0.1) write_rho(LL/8, R, grid, rho_h[0], fn);
    }
  }

  cudaDeviceSynchronize();
  get_potential(L, R, &setup, &gpu_setup);

  strncpy(setup.field_name, "fields/ev_fin.dat", 64);
  write_ev(&setup);
  free_gpu_mem(&gpu_setup);
  
  return 0;
} /* main */

/* -------------------------------------- write_rho ------------------- */
// writes charge density to a file

int write_rho(int L, int R, float grid, float **rho, char *fname) {

  int    i, j;
  float  r, z;
  FILE *file;

  if (!(file = fopen(fname, "w"))) {
    printf("ERROR: Cannot open file %s for electron density...\n", fname);
    return 1;
  } else {
    printf("Writing electron density to file %s\n\n", fname);
  }

  fprintf(file, "#\n## r (mm), z (mm), ED\n");
  for (j = 1; j < R; j++) {
    r = (j-1) * grid;
    for (i = 1; i < L; i++) {
      z = (i-1) * grid;
      fprintf(file, "%7.2f %7.2f %12.6e\n", r, z, rho[i][j]);
    }
    fprintf(file, "\n");
  }
  fclose(file);

  return 0;
}

/* -------------------------------------- read_rho ------------------- */
// reads charge density from a file

int read_rho(int L, int R, float grid, float **rho, char *fname) {

  int    i, j;
  float  r, z;
  FILE  *file;
  char   line[256];

  if (!(file = fopen(fname, "r"))) {
    printf("ERROR: Cannot open file %s for electron density...\n", fname);
    return 1;
  } else {
    printf("Reading electron density from file %s\n\n", fname);
  }

  fgets(line, 256, file);  // header line
  fgets(line, 256, file);  // header line
  for (j = 1; j < R; j++) {
    for (i = 1; i < L; i++) {
      fgets(line, 256, file);  // data line
      sscanf(line, "%f %f %e", &r, &z, &rho[i][j]);
      if (fabs(r - (j-1) * grid) > 1e-5 || fabs(z - (i-1) * grid) > 1e-5) {
        printf("ERROR: i, j = %d %d  r, z = %f %f  expect %f %f\n"
               "       diff: %e  %e\n"
               "       L, R: %d %d\n"
               "       line: %s\n",
               i, j, r, z, (j-1) * grid, (i-1) * grid, r - (j-1) * grid, z - (i-1) * grid, L, R, line);
        return 1;
      }
    }
    fgets(line, 256, file);  // blank line
  }
  fclose(file);

  return 0;
}

/* -------------------------------------- drift_rho ------------------- */
// do the diffusion and drifting of the charge cloud densities

int drift_rho(MJD_Siggen_Setup *setup, int L, int R, float grid, float ***rho, int q, double *gone) {

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
      } else if (setup->point_type[z][r] == CONTACT_EDGE) {
        E_r = ((v[new_var][z][r] - v[new_var][z][r+1])*setup->dr[1][z][r] +
               (v[new_var][z][r-1] - v[new_var][z][r])*setup->dr[0][z][r]) / (0.2*grid);
      } else if (setup->point_type[z][r] < INSIDE &&
                 setup->point_type[z][r-1] == CONTACT_EDGE) {
        E_r =  (v[new_var][z][r-1] - v[new_var][z][r]) * setup->dr[1][z][r-1] / ( 0.1*grid) ;
      } else if (setup->point_type[z][r] < INSIDE &&
                 setup->point_type[z][r+1] == CONTACT_EDGE) {
        E_r =  (v[new_var][z][r] - v[new_var][z][r+1]) * setup->dr[0][z][r+1] / ( 0.1*grid) ;
      } else if (r == R-1) {
        E_r = (v[new_var][z][r-1] - v[new_var][z][r])/(0.1*grid);
      } else {
        E_r = (v[new_var][z][r-1] - v[new_var][z][r+1])/(0.2*grid);
      }
      // calc E in z-direction
      // enum point_types{PC, HVC, INSIDE, PASSIVE, PINCHOFF, DITCH, DITCH_EDGE, CONTACT_EDGE};
      if (setup->point_type[z][r] == CONTACT_EDGE) {
        E_z = ((v[new_var][z][r] - v[new_var][z+1][r])*setup->dz[1][z][r] +
               (v[new_var][z-1][r] - v[new_var][z][r])*setup->dz[0][z][r]) / (0.2*grid);
      } else if (setup->point_type[z][r] < INSIDE &&
                 setup->point_type[z-1][r] == CONTACT_EDGE) {
        E_z =  (v[new_var][z-1][r] - v[new_var][z][r]) * setup->dz[1][z-1][r] / ( 0.1*grid) ;
      } else if (setup->point_type[z][r] < INSIDE &&
                 setup->point_type[z+1][r] == CONTACT_EDGE) {
        E_z =  (v[new_var][z][r] - v[new_var][z+1][r]) * setup->dz[0][z+1][r] / ( 0.1*grid) ;
      } else if (z == 1) {
        E_z = (v[new_var][z][r] - v[new_var][z+1][r])/(0.1*grid);
      } else if (z == L-1) {
        E_z = (v[new_var][z-1][r] - v[new_var][z][r])/(0.1*grid);
      } else {
        E_z = (v[new_var][z-1][r] - v[new_var][z+1][r])/(0.2*grid);
      }

      /* do diffusion to neighboring pixels */
      deltaez = deltaer = ve_z = ve_r = 0;
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

      // enum point_types{PC, HVC, INSIDE, PASSIVE, PINCHOFF, DITCH, DITCH_EDGE, CONTACT_EDGE};
      rho[1][z][r]   += rho[0][z][r];
      if (0 && z == 1) printf("r,z = %d, %d E_r,z = %f, %f  deltaer,z = %f, %f  s1,s2 = %f, %f\n",
                         r, z, E_r, E_z, deltaer, deltaez, setup->s1[r], setup->s2[r]);
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

      //-----------------------------------------------------------
    }
  }
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
      } else if (setup->point_type[z][r] == CONTACT_EDGE) {
        E_r = ((v[new_var][z][r] - v[new_var][z][r+1])*setup->dr[1][z][r] +
               (v[new_var][z][r-1] - v[new_var][z][r])*setup->dr[0][z][r]) / (0.2*grid);
      } else if (setup->point_type[z][r] < INSIDE &&
                 setup->point_type[z][r-1] == CONTACT_EDGE) {
        E_r =  (v[new_var][z][r-1] - v[new_var][z][r]) * setup->dr[1][z][r-1] / ( 0.1*grid) ;
      } else if (setup->point_type[z][r] < INSIDE &&
                 setup->point_type[z][r+1] == CONTACT_EDGE) {
        E_r =  (v[new_var][z][r] - v[new_var][z][r+1]) * setup->dr[0][z][r+1] / ( 0.1*grid) ;
      } else if (r == R-1) {
        E_r = (v[new_var][z][r-1] - v[new_var][z][r])/(0.1*grid);
      } else {
        E_r = (v[new_var][z][r-1] - v[new_var][z][r+1])/(0.2*grid);
      }
      // calc E in z-direction
      // enum point_types{PC, HVC, INSIDE, PASSIVE, PINCHOFF, DITCH, DITCH_EDGE, CONTACT_EDGE};
      if (setup->point_type[z][r] == CONTACT_EDGE) {
        E_z = ((v[new_var][z][r] - v[new_var][z+1][r])*setup->dz[1][z][r] +
               (v[new_var][z-1][r] - v[new_var][z][r])*setup->dz[0][z][r]) / (0.2*grid);
      } else if (setup->point_type[z][r] < INSIDE &&
                 setup->point_type[z-1][r] == CONTACT_EDGE) {
        E_z =  (v[new_var][z-1][r] - v[new_var][z][r]) * setup->dz[1][z-1][r] / ( 0.1*grid) ;
      } else if (setup->point_type[z][r] < INSIDE &&
                 setup->point_type[z+1][r] == CONTACT_EDGE) {
        E_z =  (v[new_var][z][r] - v[new_var][z+1][r]) * setup->dz[0][z+1][r] / ( 0.1*grid) ;
      } else if (z == 1) {
        E_z = (v[new_var][z][r] - v[new_var][z+1][r])/(0.1*grid);
      } else if (z == L-1) {
        E_z = (v[new_var][z-1][r] - v[new_var][z][r])/(0.1*grid);
      } else {
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
      } else {
        dre =  TSTEP*ve_r;
      }
      if (E_z > 0) {
        dze = -TSTEP*ve_z;
      } else {
        dze =  TSTEP*ve_z;
      }

      if (dre == 0.0) {
        i = r;
        fr = 1.0;
      } else {
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
      } else {
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
