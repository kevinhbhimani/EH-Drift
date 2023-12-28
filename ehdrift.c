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

   Modified in Nov 2021 by Kevin Bhimani to perform parallel calculations using GPUs

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
  Can be run as ./ehdrift config_files/P42575A.config -r 15.00 -z 0.10 -g P42575A -s 0.00 -e 5000 -v 0 -f 1 -h 0.0200
  ./ehdrift config_files/OPPI.config -r 15.00 -z 0.10 -g OPPI -s -0.30 -e 30.20 -v 0 -f 1 -h 0.0200
  ICPC detector: ./ehdrift config_files/V01386A.config -r 15.00 -z 5.00 -g V01386A -s 0.00 -e 5000 -v 0 -f 1 -h 0.0200
  WP can be calculated as ./ehdrift config_files/P42575A_calc_wp.config -r 15.00 -z 0.10 -g P42575A -s 0.00 -e 5000 -v 0 -f 1 -h 0.0200
  ./ehdrift config_files/V01386A_calc_wp.config -r 15.00 -z 5.00 -g V01386A -s 0.00 -e 5000 -v 0 -f 1 -h 0.0200
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <stdbool.h>  
#include <math.h>
#include <time.h>
#include "mjd_siggen.h"
#include "detector_geometry.h"
#include "gpu_vars.h"
#include "calc_signal.h"
#include "cyl_point.h"
#include <readline/readline.h>
#include <readline/history.h>
#include <ctype.h>
#include <string.h>
#define MAX_ITS 50000     // default max number of iterations for relaxation
#include <cuda.h>
#include <cuda_runtime.h>
#include "hdf5.h"
#include <sys/stat.h> // For checking and creating directories
#include <errno.h>    // For error numbers

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
int write_rho(int L, int R, float grid, double **rho, char *fname);
int drift_rho(MJD_Siggen_Setup *setup, int L, int R, float grid, double ***rho, int q, double *gone);
int read_rho(int L, int R, float grid, double **rho, char *fname);
void tell_ehd_2(const char *format, ...);

int gpu_drift(MJD_Siggen_Setup *setup, int L, int R, float grid, double ***rho, int q, GPU_data *gpu_setup, int n_iter);
void set_rho_zero_gpu(GPU_data *gpu_setup, int L, int R, int num_blocks, int num_threads);
void update_impurities_gpu(GPU_data *gpu_setup, int L, int R, int num_blocks, int num_threads, double e_over_E, float grid);
double get_signal_gpu(MJD_Siggen_Setup *setup, GPU_data *gpu_setup, int L, int R, double actual_time_elapsed, double grid, int save_time, int num_threads, bool save_rho_init);
double get_total_hole_density(GPU_data *gpu_setup, int L, int R, int n_iter, int save_time, int num_blocks, int num_threads);
double get_hole_density_surface(GPU_data *gpu_setup, int L, int R, int n_iter, int save_time, int num_blocks, int num_threads);
double calculate_courant_cpu(MJD_Siggen_Setup *setup, int L, int R, float grid, int q);
int rc_integrate_ehd(double *s_in, double *s_out, float tau, int time_steps);
int ehd_field_setup_2(MJD_Siggen_Setup *setup);
int setup_wp_ehd(MJD_Siggen_Setup *setup);
int ev_calc_gpu(int ev_calc, MJD_Siggen_Setup *setup, GPU_data *gpu_setup);
int ev_calc_gpu_initial(MJD_Siggen_Setup *setup, MJD_Siggen_Setup *old_setup, GPU_data *gpu_setup);
void gpu_init(MJD_Siggen_Setup *setup, int LL_rho, double ***rho_e, double ***rho_h, GPU_data *gpu_setup);
void get_densities(int L, int R, double ***rho_e, double ***rho_h, GPU_data *gpu_setup);
void get_potential(int L, int R, MJD_Siggen_Setup *setup, GPU_data *gpu_setup);
#define TELL_NORMAL_EHD if (setup->verbosity >= NORMAL) tell_ehd_2
#define TELL_CHATTY_EHD if (setup->verbosity >= CHATTY) tell_ehd_2

/* -------------------------------------- main ------------------- */
int main(int argc, char **argv)
{
    // Variable declarations
  int write_densities = 0;
  int do_self_repulsion = 1;
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
  setup.energy = 5000.00; //default energy is 5000 kev
  double e_over_E = 11.310; // e/epsilon; for 1 mm2, charge units 1e10 e/cm3, espilon = 16*epsilon0
  double **rho_e[4], **rho_h[3];
  double egone=0, hgone=0;
  double  alpha_r_mm = 10.0;  // impact radius of alpha on passivated surface; change with -a option
  double  alpha_z_mm = 0.1;
  char fn[512]; // Ensure this is large enough
        // Configuration setup and command-line argument processing

  if (argc < 2 || argc%2 != 0 || read_config(argv[1], &setup)) {
    printf("Usage: %s <config_file_name> [options]\n"
           "   Possible options:\n"
	   "      -b bias_volts\n"
	   "      -w {0,1}  (do_not/do write the field file)\n"
	   "      -d {0,1}  (do_not/do write the depletion surface)\n"
	   "      -p {0,1}  (do_not/do write the WP file)\n"
	   "      -r <r position in mm>\n"
     "      -z <z position in mm>\n"
     "      -g detector name\n"
     "      -s <surface charge in in 1e10 e/cm2>"
     "      -e <Intereaction energy in KeV>"
     "      -v {0,1}  (do_not/do write the density files)"
     "      -f {0,1}  (do_not/do re-calculate field)"
     "      -h <grid size in mm>"
     "      -m <passivated surface depth size in mm>"
     "      -a rho_spectrum_file_name\n", argv[0]);
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
      WD = atoi(argv[++i]);               // densities write-out options
    } else if (strstr(argv[i], "-v")) {
      write_densities = atoi(argv[++i]);               // write-out options
    } else if (strstr(argv[i], "-p")) {
      setup.write_WP = atoi(argv[++i]);   // weighting-potential options
    } else if (strstr(argv[i], "-r")) {
      alpha_r_mm = atof(argv[++i]);       // alpha impact radius
    } else if (strstr(argv[i], "-z")) {
      alpha_z_mm = atof(argv[++i]);       // alpha impact z position
    } else if (strstr(argv[i], "-h")) {
      setup.xtal_grid = atof(argv[++i]);  // grid size
    } else if (strstr(argv[i], "-g")) {
      strcpy(setup.detector_name, argv[++i]);        // name of the detector
    } else if (strstr(argv[i], "-s")) {
      setup.impurity_surface =  atof(argv[++i]);  // surface charge
    } else if (strstr(argv[i], "-e")) {      // set the energy of interaction in KeV
      setup.energy = atof(argv[++i]);
    } else if (strstr(argv[i], "-f")) {      // flag to turn off or on field recalculation
      do_self_repulsion = atof(argv[++i]);
    } else if (strstr(argv[i], "-m")) {
      setup.passivated_thickness=atof(argv[++i]);  // passivated surface thickness in mm
    }else if (strstr(argv[i], "-a")) {
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
  if(setup.passivated_thickness < 0.0001){
    setup.passivated_thickness =0.002;
  } //defualt passivation is 2 micron

    if(setup.passivated_thickness>=setup.xtal_grid){
    printf("Grid is %f\n", setup.xtal_grid);
    printf("Passivated surface depth is %f\n",setup.passivated_thickness);
    printf("\n\nPassivated surface cannot be thicker than the grid\n\n");
    return 0;
  }
    
  printf("\nHome dir is %s\n", setup.home_dir);
  printf("\nScratch dir is %s\n", setup.scratch_dir);
  printf("\nPassivated surface thickness is %f\n", setup.passivated_thickness);
  printf("\nDetector name is %s\n", setup.detector_name);
  printf("\nEnergy of Interaction in KeV is %f\n", setup.energy);



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
    
  char fn_ev[256];
  sprintf(fn_ev, "%s/fields/ev_fin_grid=%.4f_sc=%.4f.dat",setup.scratch_dir, setup.xtal_grid, setup.impurity_surface); 
  strncpy(setup.field_name, fn_ev, 256);

  /* add electrons and holes at a specific point location near passivated surface */
  double e_dens, esum1=0, esum2=0, ecentr=0, ermsr=0, ecentz=0, ermsz=0;
  double hsum1=0, hsum2=0, hcentr=0, hrmsr=0, hcentz=0, hrmsz=0;
  double  grid = setup.xtal_grid;
  int    L  = lrint(setup.xtal_length/grid)+2; 
  // int    LL_rho = lrint(setup.xtal_length/grid)+2; // 3 so that last z is traced as passivated surface
  // int    LL_rho = (lrint(setup.xtal_length/grid)+3)/8;
  int    LL_rho = L;
  int    R  = lrint(setup.xtal_radius/grid)+2;
  int    r, z, rr,zz, k;

  /* malloc and clear space for electron density arrays */
  for (j=0; j<4; j++) {
    if ((rho_e[j] = malloc(LL_rho * sizeof(*rho_e[j])))   == NULL) {
      printf("malloc failed\n");
      return -1;
    }
    for (i = 0; i < LL_rho; i++) {
      if ((rho_e[j][i] = malloc(R * sizeof(**rho_e[j])))   == NULL) {
        printf("malloc failed\n");
        return -1;
      }
    }
  }
  for (i = 0; i < LL_rho; i++) {
    for (j = 0; j < R; j++) {
      rho_e[0][i][j] = rho_e[1][i][j] = rho_e[2][i][j] = 0;
      rho_e[3][i][j] = setup.impurity[i][j]; // save impurity values for use later
    }
  }
  /* malloc and clear space for hole density arrays */
  for (j=0; j<3; j++) {
    if ((rho_h[j] = malloc(LL_rho * sizeof(*rho_h[j])))   == NULL) {
      printf("malloc failed\n");
      return -1;
    }
    for (i = 0; i < LL_rho; i++) {
      if ((rho_h[j][i] = malloc(R * sizeof(**rho_h[j])))   == NULL) {
        printf("malloc failed\n");
        return -1;
      }
    }
  }
  for (i = 0; i < LL_rho; i++) {
    for (j = 0; j < R; j++) {
      rho_h[0][i][j] = rho_h[1][i][j] = rho_h[2][i][j] = 0;
    }
  }

  /* add electrons and holes */
  //.003: [E in keV]/[.0027 keV per charge pair] = [number of charge pairs] [# of charge pairs]/[grid size^3] = [density of charge pairs]

  e_dens = (setup.energy / 1000.0 * 1e-7) / 0.003 / (grid * grid * grid); // energy in keV, converted to MeV, units of 1e10/cm3
  e_dens *= 20.0;                           // these numbers are fudged to get the self-repulsion about right (2D vs. 3D)
  r = alpha_r_mm/grid + 1;  // CHANGEME : starting radius, converted from mm to grid points; see -a command line option
  z = alpha_z_mm/grid + 1; // CHANGEME currently z = 0.1 mm
// NOT(A or B) = NOT(A) and NOT(B)
//enum point_types{PC, HVC, INSIDE, PASSIVE, PINCHOFF, DITCH, DITCH_EDGE, CONTACT_EDGE};

  // if (setup.point_type[z][r] != PASSIVE && setup.point_type[z][r] != INSIDE && setup.point_type[z][r] != PC && setup.point_type[z][r] != HVC) {
  if (setup.point_type[z][r] != PASSIVE && setup.point_type[z][r] != INSIDE) {      
const char *pointTypeNames[] = {"PC", "HVC", "INSIDE", "PASSIVE", "PINCHOFF", "DITCH", "DITCH_EDGE", "CONTACT_EDGE"};
if (setup.point_type[z][r] >= 0 && setup.point_type[z][r] <= CONTACT_EDGE) {
  printf("Point is of type %s\n", pointTypeNames[setup.point_type[z][r]]);
} else {
  printf("Unknown point type\n");
}
printf("EVENT LOCATION NOT INSIDE ACTIVE VOLUME OF THE DECTOR\n");
    return 0;
  }



  /* CHANGEME?
       at this point, you can either read in some starting charge distribution
       or just put charge at the surface, at some specified radius r
   */
  if (1) {  // CHANGEME : change 1 to 0 if you want to read in some starting charge
            //              distribution, e.g. to continue a previous calculation
    // use initial alpha interaction at (r, z~0) as starting distribution
    rho_e[0][z][r] = rho_e[0][z+1][r] = e_dens/2.0;
    rho_h[0][z][r] = rho_h[0][z+1][r] = e_dens/2.0; 
  }
   else {
    // read starting rho_e and rho_h values
    if (read_rho(LL_rho, R, grid, rho_e[0], "start_ed.dat") ||
        read_rho(LL_rho, R, grid, rho_h[0], "start_hd.dat")) return 1;
  }
  
  for (zz=1; zz<LL_rho; zz++) {
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

  if (setup.write_WP) return 0;
    /* we need **v to have electric potential, not WP, so we quit here */

    if(write_densities){
      char fn_1[256], fn_2[256];
      sprintf(fn_1, "%s/%.2f_keV/grid_%.4f/self_repulsion_%d/%s/q=%.2f/drift_data_r=%.2f_z=%.2f/ed000.dat", setup.scratch_dir, setup.energy, grid, do_self_repulsion, setup.detector_name, setup.impurity_surface, alpha_r_mm, alpha_z_mm);
      sprintf(fn_2, "%s/%.2f_keV/grid_%.4f/self_repulsion_%d/%s/q=%.2f/drift_data_r=%.2f_z=%.2f/hd000.dat", setup.scratch_dir, setup.energy, grid, do_self_repulsion, setup.detector_name, setup.impurity_surface, alpha_r_mm, alpha_z_mm);
      write_rho(LL_rho/8, R, grid, rho_e[0], fn_1);
      write_rho(LL_rho/8, R, grid, rho_h[0], fn_2);
    }

  /* -------------- read weighting potential */
  if (!setup.write_WP) {
    if (ehd_field_setup_2(&setup)) return 1;
  }

  printf("\nCalculating Courant number\n");
  double courant_number = calculate_courant_cpu(&setup, LL_rho, R, grid, -1);
  printf("The courant number is %f\n", courant_number);
  if(courant_number >0.f){
    setup.step_time_calc = setup.step_time_calc/ courant_number;
  } 
  printf("Time step is set at %f\n", setup.step_time_calc);    
    
  gpu_init(&setup, LL_rho, rho_e, rho_h, &gpu_setup);

  gpu_drift(&setup, LL_rho, R, grid, rho_e, -1, &gpu_setup, 0);
  gpu_drift(&setup, LL_rho, R, grid, rho_h, 1, &gpu_setup, 0);

  /* -----------------------------------------
   *   This loop starting here is crucial.
   *   It loops over time steps (size = time_steps_calc in the config file) calculating
   *    the self-consistent field and letting the charge densities diffuse and drift.
   * ----------------------------------------- */


  // Below we use modular arithymatic to divide r and z value into blocks and grids.
  int n_iter=1;
  double sig[100000] = {0};
  int sim_time = 8000; //run time in nano seconds

  double courant_num_vec[100000] = {0};
  int num_threads = 1024;
  int num_blocks = R * (ceil(LL_rho/num_threads)+1);
  int save_time = setup.step_time_out; // Time interval to save the signal in nanoseconds
  //for 0.2 time step, 10 would correspond to interval of 2 ns
  double total_hole_density[100000] = {0};
  double hole_density_surface[100000] = {0};
  double signal_time_n;  
  bool change_time_step = false;  // Flag to indicate if time step should be changed
  double new_time_step_ns = 0.5;  // New time step in ns after conditions are met
  double signal_threshold = 0.95;
  double time_threshold_ns = 400.0;
  int save_init_rho = 1; //time in ns to save the inital number of densiies for siganl calculations  
  int last_signal_pos = 0;  
  double actual_time_elapsed = 0.0; // To track the actual simulation time
  double last_save_time = 0.0;
  int save_index = 0;
  bool save_rho_init = true;  // Flag to indicate if time step should be changed
  int write_rho_counter = 1;
    
  double prev_signal = 0;
  double signal_diff_threshold = 0.001; // 0.1%
  double max_time_step = 5.0; // maximum time step in ns
  //save_time*2 to just make sure we run enough time steps to generate time of total time
    
  printf("\n\n -=-=-=-=-=-Starting Signal Calculations\n-=-=-=-=-=-");
  printf("\n\n -=-=-=-=-=- Running at time %.2f out of %d\n\n-=-=-=-=-=-", 0.0, sim_time);
    
  for (n_iter = 1; actual_time_elapsed <= sim_time+(save_time*2); n_iter++) {
      
    if(n_iter%1000==0){
        printf("\n\n -=-=-=-=-=- Running at time %.2f out of %d", actual_time_elapsed, sim_time);
        printf(" -=-=-=-=-=- Signal collected=%.4f -=-=-=-=-=-\n\n", sig[save_index-1]/1000);
    }    
    cudaDeviceSynchronize();
    set_rho_zero_gpu(&gpu_setup, LL_rho, R, num_blocks, num_threads);
    cudaDeviceSynchronize();

    update_impurities_gpu(&gpu_setup, LL_rho, R, num_blocks, num_threads, e_over_E, grid);
    cudaDeviceSynchronize();
      
      
    if (actual_time_elapsed - last_save_time >= save_rho_init) {
        if(save_rho_init){
            get_signal_gpu(&setup, &gpu_setup, LL_rho, R, actual_time_elapsed, grid, save_time, num_threads, true);
            save_rho_init = false;
        }
    }
      
    signal_time_n = get_signal_gpu(&setup, &gpu_setup, LL_rho, R, actual_time_elapsed, grid, save_time, num_threads, false);        

    // Save signal logic
    if (actual_time_elapsed - last_save_time >= save_time) {
   
        signal_time_n = get_signal_gpu(&setup, &gpu_setup, LL_rho, R, actual_time_elapsed, grid, save_time, num_threads, false);        
        if (isnan(signal_time_n)) {  // Check if signal is NaN
            printf("\n\n Error: Signal collected is NaN at iteration %d. Exiting the program.\n\n", n_iter);
            return -1; // Exit with an error code
        }
        
            // Check and adjust the signal to ensure it's never decreasing
        if (signal_time_n < prev_signal) {
            signal_time_n = prev_signal; // Set current signal to previous value if it's less
        }
        
        // Check if the signal is greater than 1 and adjust if necessary
        if (signal_time_n > 1000) {
            signal_time_n = 1000;
            setup.step_time_calc = 2.0;
        }
        
        sig[save_index] = signal_time_n;
        last_signal_pos = save_index;
        save_index++;
        last_save_time = actual_time_elapsed;
        
      if(write_densities){
        cudaDeviceSynchronize();
        get_densities(LL_rho, R, rho_e, rho_h, &gpu_setup);
        // Use sprintf to format the entire string, including scratch_dir
        sprintf(fn, "%s/%.2f_keV/grid_%.4f/self_repulsion_%d/%s/q=%.2f/drift_data_r=%.2f_z=%.2f/ed%3.3d.dat", 
                setup.scratch_dir, setup.energy, grid, do_self_repulsion, setup.detector_name, 
                setup.impurity_surface, alpha_r_mm, alpha_z_mm, write_rho_counter);
        write_rho(LL_rho/8, R, grid, rho_e[0], fn);
          
        // Use sprintf to format the entire string, including home_dir
        sprintf(fn, "%s/%.2f_keV/grid_%.4f/self_repulsion_%d/%s/q=%.2f/drift_data_r=%.2f_z=%.2f/hd%3.3d.dat", 
                setup.scratch_dir, setup.energy, grid, do_self_repulsion, setup.detector_name, 
                setup.impurity_surface, alpha_r_mm, alpha_z_mm, write_rho_counter);
        write_rho(LL_rho/8, R, grid, rho_h[0], fn);
        write_rho_counter +=1;
      }
    }
    if(do_self_repulsion){
      cudaDeviceSynchronize();
      ev_calc_gpu(1, &setup, &gpu_setup);
    }
    cudaDeviceSynchronize();
    gpu_drift(&setup, LL_rho, R, grid, rho_e, -1, &gpu_setup, n_iter);
    cudaDeviceSynchronize();
    gpu_drift(&setup, LL_rho, R, grid, rho_h, 1, &gpu_setup, n_iter);
      
    // After performing necessary operations, update actual_time_elapsed
    actual_time_elapsed += setup.step_time_calc;
    // Check if it's time to change the time step
    // if (!change_time_step && 
    //     (sig[last_signal_pos] / 1000 > signal_threshold || actual_time_elapsed > time_threshold_ns)) {
    //     change_time_step = true;
    //     printf("\n\n-=-=-=-=-=-Updating time step to %f-=-=-=-=-=-\n\n", new_time_step_ns);
    //     setup.step_time_calc = new_time_step_ns;
    // }
        // Check the difference in signal and adjust time step
    if (prev_signal > 0) { // Skip the first iteration as there's no previous signal
        double signal_diff = fabs(signal_time_n - prev_signal) / prev_signal;
        if (signal_diff < signal_diff_threshold && setup.step_time_calc < max_time_step) {
            setup.step_time_calc = fmin(setup.step_time_calc * 2, max_time_step);
            printf("\n\n-=-=-=-=-=-Changing time step to %f", setup.step_time_calc);
            printf(" -=-=-=-=-=- Signal collected=%.4f -=-=-=-=-=-\n\n", signal_time_n/1000);

        }
    }
    prev_signal = signal_time_n; // Update the previous signal for the next iteration
  }
  cudaDeviceSynchronize();
  get_potential(L, R, &setup, &gpu_setup);

  /* do RC integration for preamp risetime */
  if (setup.preamp_tau > 1) {
    rc_integrate_ehd(sig, sig, setup.preamp_tau/setup.step_time_out, n_iter/save_time);
    for (n_iter=0; n_iter<sim_time/(setup.step_time_calc*save_time)-1; n_iter++) sig[n_iter] = sig[n_iter+1];
  }

  printf("signal: \n");
  for (i = 0; i < (sim_time/save_time); i++){
    printf("%.5f ", sig[i]/1000);
    if (i%20 == 0) printf("\n");
  }
  printf("\n");
    
// Use code below to save to text file instead    
//   printf("Saving waveform\n");
//   int written = 0;
//   char filename[1024];
//   sprintf(filename, "%s/waveforms/%.2f_keV/grid_%.4f/self_repulsion_%d/%s/q=%.2f/signal_r=%.2f_phi=0.00_z=%.2f.txt",setup.home_dir, setup.energy, grid, do_self_repulsion, setup.detector_name, setup.impurity_surface, alpha_r_mm, alpha_z_mm);
//   // printf("The file name is %s\n", filename);
//   FILE *f = fopen(filename,"w");
//   //written = fwrite(sig, sizeof(float), sizeof(sig), f);
//   for(i = 0; i < (sim_time/save_time); i++){
//     written = fprintf(f,"%f\n",sig[i]/1000);
//     if (written == 0) {
//     printf("Error during writing to file!");
//     break;
//     }
//   }
//   fclose(f);
//   printf("Done writting waveform\n");
    


printf("Saving waveform in HDF5 format\n");

// Construct the directory path based on the detector name
char dir_path[1024];
sprintf(dir_path, "%s/waveforms/%s", setup.home_dir, setup.detector_name);

// Check if the directory exists
struct stat st = {0};
if (stat(dir_path, &st) == -1) {
    // Directory does not exist, create it
    if (mkdir(dir_path, 0700) == -1) {
        fprintf(stderr, "Error creating directory %s: %s\n", dir_path, strerror(errno));
        exit(EXIT_FAILURE);
    }
}

// Construct the file path
char hdf5_filename[1024];
sprintf(hdf5_filename, "%s/signal_data_%.2f_keV_grid%.4f_%s_q%.2f_r%.2f_z%.2f.h5", 
        dir_path, setup.energy, grid, setup.detector_name, setup.impurity_surface, alpha_r_mm, alpha_z_mm);

hid_t file_id, dataset_id, dataspace_id, attr_id, attr_dataspace_id;  // HDF5 identifiers
hsize_t dims[1]; // Dimensions of the dataset

// Create a new file using the default properties
file_id = H5Fcreate(hdf5_filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

// Prepare scaled waveform data
dims[0] = (sim_time/save_time);
double scaled_sig[dims[0]];
for (int i = 0; i < dims[0]; i++) {
    scaled_sig[i] = sig[i] / 1000.0; // Divide each signal value by 1000
}

// ---- Save Waveform ----
dataspace_id = H5Screate_simple(1, dims, NULL);
dataset_id = H5Dcreate(file_id, "/waveform", H5T_NATIVE_DOUBLE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, scaled_sig);
H5Dclose(dataset_id);
H5Sclose(dataspace_id);

// ---- Save Energy, r, z, grid, and detector name as individual datasets ----
// Assuming energy, alpha_r_mm, and alpha_z_mm are double
dims[0] = 1;

// Energy
dataspace_id = H5Screate_simple(1, dims, NULL);
dataset_id = H5Dcreate(file_id, "/energy", H5T_NATIVE_DOUBLE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &setup.energy);
H5Dclose(dataset_id);

// r
dataset_id = H5Dcreate(file_id, "/r", H5T_NATIVE_DOUBLE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &alpha_r_mm);
H5Dclose(dataset_id);

// z
dataset_id = H5Dcreate(file_id, "/z", H5T_NATIVE_DOUBLE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &alpha_z_mm);
H5Dclose(dataset_id);

// Grid
dataset_id = H5Dcreate(file_id, "/grid", H5T_NATIVE_DOUBLE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &grid);
H5Dclose(dataset_id);

// Detector name
dims[0] = strlen(setup.detector_name) + 1; // Include space for null terminator
dataspace_id = H5Screate_simple(1, dims, NULL);
dataset_id = H5Dcreate(file_id, "/detector_name", H5T_C_S1, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
H5Dwrite(dataset_id, H5T_C_S1, H5S_ALL, H5S_ALL, H5P_DEFAULT, setup.detector_name);
H5Dclose(dataset_id);
H5Sclose(dataspace_id);

// ---- Save Attributes ----
attr_dataspace_id = H5Screate(H5S_SCALAR);

// Surface charge
attr_id = H5Acreate(file_id, "surface_charge", H5T_NATIVE_DOUBLE, attr_dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
H5Awrite(attr_id, H5T_NATIVE_DOUBLE, &setup.impurity_surface);
H5Aclose(attr_id);

// Self repulsion
attr_id = H5Acreate(file_id, "self_repulsion", H5T_NATIVE_INT, attr_dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
H5Awrite(attr_id, H5T_NATIVE_INT, &do_self_repulsion);
H5Aclose(attr_id);

H5Sclose(attr_dataspace_id);

// Close the file
H5Fclose(file_id);

printf("Done writing waveform in HDF5 format\n");

  return 0;
} /* main */

int rc_integrate_ehd(double s_in[], double s_out[], float tau, int time_steps){
  int   j;
  float s_in_old, s;  /* DCR: added so that it's okay to
			 call this function with s_out == s_in */
  
  if (tau < 1.0f) {
    for (j = time_steps-1; j > 0; j--) s_out[j] = s_in[j-1];
    s_out[0] = 0.0;
  } else {
    s_in_old = s_in[0];
    s_out[0] = 0.0;
    for (j = 1; j < time_steps; j++) {
      s = s_out[j-1] + (s_in_old - s_out[j-1])/tau;
      s_in_old = s_in[j];
      s_out[j] = s;
    }
  }
  return 0;
}

int ehd_field_setup_2(MJD_Siggen_Setup *setup){

  setup->rmin  = 0;
  setup->rmax  = setup->xtal_radius;
  setup->rstep = setup->xtal_grid;
  setup->zmin  = 0;
  setup->zmax  = setup->xtal_length;
  setup->zstep = setup->xtal_grid;
  if (setup->xtal_temp < MIN_TEMP) setup->xtal_temp = MIN_TEMP;
  if (setup->xtal_temp > MAX_TEMP) setup->xtal_temp = MAX_TEMP;

  printf("rmin: %.2f rmax: %.2f, rstep: %.2f\n"
         "zmin: %.2f zmax: %.2f, zstep: %.2f\n"
         "Detector temperature is set to %.1f K\n",
         setup->rmin, setup->rmax, setup->rstep,
         setup->zmin, setup->zmax, setup->zstep,
         setup->xtal_temp);

  if (setup_wp_ehd(setup) != 0){
    printf("Failed to read weighting potential from file %s\n", setup->wp_name);
    return -1;
  }

  return 0;
}
/*setup_wp_ehd
  read weighting potential values from files. returns 0 on success*/
int setup_wp_ehd(MJD_Siggen_Setup *setup){
  FILE   *fp;
  char   line[MAX_LINE], *cp;
  int    i, j, lineno;
  cyl_pt cyl;
  double  wp, **wpot;

  setup->rlen = lrintf((setup->rmax - setup->rmin)/setup->rstep) + 1;
  setup->zlen = lrintf((setup->zmax - setup->zmin)/setup->zstep) + 1;
  TELL_CHATTY_EHD("rlen, zlen: %d, %d\n", setup->rlen, setup->zlen);

  //assuming rlen, zlen never change as for setup_efld
  if ((wpot = (double **) malloc(setup->rlen*sizeof(*wpot))) == NULL){
    error("Malloc failed in setup_wp\n");
    return 1;
  }
  for (i = 0; i < setup->rlen; i++){
    if ((wpot[i] = (double *) malloc(setup->zlen*sizeof(*wpot[i]))) == NULL){  
      error("Malloc failed in setup_wp\n");
      //NB: memory leak here.
      return 1;
    }
    memset(wpot[i], 0, setup->zlen*sizeof(*wpot[i]));
  }
  if ((fp = fopen(setup->wp_name, "r")) == NULL){
    error("failed to open file: %s\n", setup->wp_name);
    return -1;
  }
  lineno = 0;
  TELL_NORMAL_EHD("Reading weighting potential from file: %s\n", setup->wp_name);

  if (strstr(setup->wp_name, "unf")) {
    /* try to read from unformatted file */
    while (fgets(line, sizeof(line), fp) && line[0] == '#' &&
           !strstr(line, "start of unformatted data"))
      ;
    if (line[0] != '#') rewind(fp);
    fread(&i, sizeof(int), 1, fp);
    fread(&j, sizeof(int), 1, fp);
    if (i != setup->rlen || j != setup->zlen) {
      error("Error in WP dimensions: %d != %d, or %d != %d\n", i, setup->rlen, j, setup->zlen);
      fclose(fp);
      return -1;
    }
    for (i = 0; i < setup->rlen; i++) {
      if (fread(wpot[i], sizeof(double), setup->zlen, fp) != setup->zlen) {
        error("Error while reading %s\n", setup->wp_name);
        return -1;
      }
    }
    TELL_NORMAL_EHD("Done reading field, %d x %d points\n", setup->rlen, setup->zlen);

  } else {

    /* read the table from a text file*/
    while (fgets(line, MAX_LINE, fp) != NULL){
      lineno++;
      for (cp = line; isspace(*cp) && *cp != '\0'; cp++);
      if (*cp == '#' || !strlen(cp)) continue;
      if (sscanf(line, "%f %f %lf\n",&cyl.r, &cyl.z, &wp) != 3){ 
        error("failed to read weighting potential from line %d\n"
              "line: %s", lineno, line);
        fclose(fp);
        return 1;
      }
      i = lrintf((cyl.r - setup->rmin)/setup->rstep);
      j = lrintf((cyl.z - setup->zmin)/setup->zstep);
      if (i < 0 || i >= setup->rlen || j < 0 || j >= setup->zlen) continue;
      if (outside_detector_cyl(cyl, setup)) continue;
      wpot[i][j] = wp;
    }
    TELL_NORMAL_EHD("Done reading %d lines of WP data\n", lineno);
  }

  fclose(fp);
  setup->wpot = wpot;
  for (i = 0; i < setup->rlen; i++) setup->wpot[i] = wpot[i];

  return 0;
}

void tell_ehd_2(const char *format, ...){
  va_list ap;

  va_start(ap, format);
  vprintf(format, ap);
  va_end(ap);
  return;
}

/* -------------------------------------- write_rho ------------------- */
// writes charge density to a file

int write_rho(int L, int R, float grid, double **rho, char *fname) {

  int    i, j;
  float  r, z;
  FILE *file;

  if (!(file = fopen(fname, "w"))) {
    printf("ERROR: Cannot open file %s for electron density...\n", fname);
    return 1;
  } else {
    printf("Writing electron density to file %s\n", fname);
  }

  fprintf(file, "#\n## r (mm), z (mm), ED\n");
  for (j = 1; j < R; j++) {
    r = (j-1) * grid;
    for (i = 1; i < L; i++) {
      z = (i-1) * grid;
      fprintf(file, "%7.4f %7.4f %14.8e\n", r, z, rho[i][j]);
    }
    fprintf(file, "\n");
  }
  fclose(file);

  return 0;
}

/* -------------------------------------- read_rho ------------------- */
// reads charge density from a file

int read_rho(int L, int R, float grid, double **rho, char *fname) {

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


double calculate_courant_cpu(MJD_Siggen_Setup *setup, int L, int R, float grid, int q) {

  #define TSTEP   setup-> step_time_calc  // time step of calculation, in ns; do not exceed 2.0 !!!
  #define DELTA   (0.07*TSTEP) /* Prob. of a charge moving to next 20-micron bin during a time step */
  #define DELTA_R (0.07*TSTEP) /* ... in r-direction */

  /* ASSUMPTIONS:
     0.1 mm grid size, 1 ns steps
     detector temperature = REF_TEMP = 77 K
     dealing only with electrons (no holes)
  */
  double cournat_numb=0;
  int    i, k, r, z, new = 0;
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
    /* NOTE that impurity and field arrays in setup start at (i,j)=(1,1) for (r,z)=(0,0) */
  int idid = lrint((setup->wrap_around_radius - setup->ditch_thickness)/grid) + 1; // ditch ID
  int idod = lrint(setup->wrap_around_radius/grid) + 1; // ditch OD
  int idd =  lrint(setup->ditch_depth/grid) + 1;        // ditch depth

  for (r=1; r<R; r++) {
    for (z=1; z<L-2; z++) {

      // need to r-calculate all the fields
      // calc E in r-direction
      if (r == 1) {  // r = 0; symmetry implies E_r = 0
        E_r = 0;
      } else if (setup->point_type[z][r] == CONTACT_EDGE) {
        E_r = ((v[new][z][r] - v[new][z][r+1])*setup->dr[1][z][r] +
               (v[new][z][r-1] - v[new][z][r])*setup->dr[0][z][r]) / (0.2*grid);
      } else if (setup->point_type[z][r] < INSIDE &&
                 setup->point_type[z][r-1] == CONTACT_EDGE) {
        E_r =  (v[new][z][r-1] - v[new][z][r]) * setup->dr[1][z][r-1] / ( 0.1*grid) ;
      } else if (setup->point_type[z][r] < INSIDE &&
                 setup->point_type[z][r+1] == CONTACT_EDGE) {
        E_r =  (v[new][z][r] - v[new][z][r+1]) * setup->dr[0][z][r+1] / ( 0.1*grid) ;
      } else if (r == R-1) {
        E_r = (v[new][z][r-1] - v[new][z][r])/(0.1*grid);
      } else {
        E_r = (v[new][z][r-1] - v[new][z][r+1])/(0.2*grid);
      }
      // calc E in z-direction
      // enum point_types{PC, HVC, INSIDE, PASSIVE, PINCHOFF, DITCH, DITCH_EDGE, CONTACT_EDGE};
      if (setup->point_type[z][r] == CONTACT_EDGE) {
        E_z = ((v[new][z][r] - v[new][z+1][r])*setup->dz[1][z][r] +
               (v[new][z-1][r] - v[new][z][r])*setup->dz[0][z][r]) / (0.2*grid);
      } else if (setup->point_type[z][r] < INSIDE &&
                 setup->point_type[z-1][r] == CONTACT_EDGE) {
        E_z =  (v[new][z-1][r] - v[new][z][r]) * setup->dz[1][z-1][r] / ( 0.1*grid) ;
      } else if (setup->point_type[z][r] < INSIDE &&
                 setup->point_type[z+1][r] == CONTACT_EDGE) {
        E_z =  (v[new][z][r] - v[new][z+1][r]) * setup->dz[0][z+1][r] / ( 0.1*grid) ;
      } else if (z == 1) {
        E_z = (v[new][z][r] - v[new][z+1][r])/(0.1*grid);
      } else if (z == L-1) {
        E_z = (v[new][z-1][r] - v[new][z][r])/(0.1*grid);
      } else {
        E_z = (v[new][z-1][r] - v[new][z+1][r])/(0.2*grid);
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
        ve_z *= setup->surface_drift_vel_factor;  // assume 2-micron-thick roughness/passivation in z
      }
      // ve_r=ve_r*grid*grid/(setup->passivated_thickness*setup->passivated_thickness);
      // ve_z=ve_z*grid/(setup->passivated_thickness);
      // ve_r=ve_r*grid;
      // ve_z=ve_z*grid;
      // double cournat= ((ve_r+ve_z)*TSTEP)/grid;
      double cournat= (ve_r+ve_z)*setup->step_time_calc;
      if(cournat-cournat_numb>0.000001){
        cournat_numb= cournat;
      }

    }
    }
  return cournat_numb;
}