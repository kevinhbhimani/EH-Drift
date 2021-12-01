/* program to calculate signals from resuts of ehdrift simulation results
   author:           D.C. Radford
   first written:         Oct 2020
   Can be ran as ./ehd_siggen config_files/P42575A.config -a 25.00 -z 0.10 -g P42575A -s 0.00
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdarg.h>
#include "mjd_siggen.h"
#include "calc_signal.h"
#include "fields.h"
#include "detector_geometry.h"


#include "cyl_point.h"
#include <readline/readline.h>
#include <readline/history.h>
#include <ctype.h>
#include <string.h>

#define MAX_ITS 50000     // default max number of iterations for relaxation

int report_config(FILE *fp_out, char *config_file_name);
int grid_init(MJD_Siggen_Setup *setup);

int read_rho(int L, int R, float grid, float **rho, char *fname);
int ehd_field_setup(MJD_Siggen_Setup *setup);
int setup_wp(MJD_Siggen_Setup *setup);
int rc_integrate(float *s_in, float *s_out, float tau, int time_steps);
int write_spectrum(float *spec, int nchs, char *name);

void tell(const char *format, ...){
  va_list ap;

  va_start(ap, format);
  vprintf(format, ap);
  va_end(ap);
  return;
}
void error(const char *format, ...) {
  va_list ap;

  va_start(ap, format);
  vfprintf(stderr, format, ap);
  va_end(ap);
  return;
}

/* -------------------------------------- main ------------------- */
int main(int argc, char **argv)
{

  MJD_Siggen_Setup setup;

  float BV;      // bias voltage
  int   WV = 0;  // 0: do not write the V and E values to ppc_ev.dat
                 // 1: write the V and E values to ppc_ev.dat
                 // 2: write the V and E values for both +r, -r (for gnuplot, NOT for siggen)
  int   WD = 0;  // 0: do not write out depletion surface
                 // 1: write out depletion surface to depl_<HV>.dat

  int   i, j;
  FILE  *fp;
  float **rho_e, **rho_h;

  float  alpha_r_mm = 10.0;  // impact radius of alpha on passivated surface; change with -a option
  float  alpha_z_mm = 0.1;  // impact z position of alpha on passivated surface; change with -z option
  char det_name[8];

  if (argc < 2 || argc%2 != 0 || read_config(argv[1], &setup)) {
    printf("Usage: %s <config_file_name> [options]\n"
           "   Possible options:\n"
	   "      -b bias_volts\n"
	   "      -w {0,1}  do_not/do write the field file)\n"
	   "      -d {0,1}  do_not/do write the depletion surface)\n"
	   "      -p {0,1}  do_not/do write the WP file)\n"     
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
      alpha_z_mm = atof(argv[++i]);       // alpha impact z
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
    
  /* add electrons and holes at a specific point location near passivated surface */
  double esum01=0, esum02=0, esum1, esum2, ecentr, ermsr, ecentz, ermsz;
  double hsum01=0, hsum02=0, hsum1, hsum2, hcentr, hrmsr, hcentz, hrmsz;
  float  grid = setup.xtal_grid;
  int    L  = lrint(setup.xtal_length/grid)+2;
  int    LL = L/8;
  int    R  = lrint(setup.xtal_radius/grid)+2;
  int    r, z, n;
  char   fn[256];
  float  sig[1024] = {0};

  /* malloc and clear space for electron and hole density arrays */
  if ((rho_e = malloc(LL * sizeof(*rho_e)))   == NULL ||
      (rho_h = malloc(LL * sizeof(*rho_h)))   == NULL) {
    printf("malloc failed\n");
    return -1;
  }
  for (i = 0; i < LL; i++) {
    if ((rho_e[i] = malloc(R * sizeof(**rho_e)))   == NULL ||
        (rho_h[i] = malloc(R * sizeof(**rho_h)))   == NULL) {
      printf("malloc failed\n");
      return -1;
    }
  }
  for (i = 0; i < LL; i++) {
    for (j = 0; j < R; j++) {
      rho_e[i][j] = rho_h[i][j] = 0;
    }
  }

  /* -------------- read weighting potential */
  if (ehd_field_setup(&setup)) return 1;

  /* -------------- calculate signal */
  for (n=1; n<=400; n+=1) {
    sprintf(fn, "/pine/scr/k/b/kbhimani/siggen_sims/%s/q=%.2f/drift_data_r=%.2f_z=%.2f/ed%3.3d.dat",det_name,setup.impurity_surface, alpha_r_mm, alpha_z_mm, n); 
    if (read_rho(LL, R, grid, rho_e, fn)) break;
    sprintf(fn, "/pine/scr/k/b/kbhimani/siggen_sims/%s/q=%.2f/drift_data_r=%.2f_z=%.2f/hd%3.3d.dat",det_name,setup.impurity_surface, alpha_r_mm, alpha_z_mm, n); 
    if (read_rho(LL, R, grid, rho_h, fn)) break;

    esum1 = esum2 = ecentr = ermsr = ecentz = ermsz = 0;
    hsum1 = hsum2 = hcentr = hrmsr = hcentz = hrmsz = 0;
    for (z=1; z<LL; z++) {
      for (r=1; r<R-1; r++) {
        esum1 += rho_e[z][r] * (double) (r-1) * setup.wpot[r-1][z-1];
        esum2 += rho_e[z][r] * (double) (r-1);
        hsum1 += rho_h[z][r] * (double) (r-1) * setup.wpot[r-1][z-1];
        hsum2 += rho_h[z][r] * (double) (r-1);
        ecentr += rho_e[z][r] * (double) ((r-1) * (r-1));
        ecentz += rho_e[z][r] * (double) ((r-1) * (z-1));
        ermsr  += rho_e[z][r] * (double) ((r-1) * (r-1) * (r-1));
        ermsz  += rho_e[z][r] * (double) ((r-1) * (z-1) * (z-1)) ;
        hcentr += rho_h[z][r] * (double) ((r-1) * (r-1));
        hcentz += rho_h[z][r] * (double) ((r-1) * (z-1));
        hrmsr  += rho_h[z][r] * (double) ((r-1) * (r-1) * (r-1));
        hrmsz  += rho_h[z][r] * (double) ((r-1) * (z-1) * (z-1)) ;
      }
    }

    // assume that any holes that have disappeared went to the point contact, so WP = 1
    if (n > 1 && hsum02 > hsum2) hsum1 += hsum02 - hsum2;
    
    printf("n: %3d  esums: %6.0f %6.0f ratio: %.5f %.5f", n, esum1, esum2, esum1/esum2, esum1/esum02);
    ecentr /= esum2;
    ermsr -= esum2 * ecentr*ecentr;
    ermsr /= esum2;
    ecentz /= esum2;
    ermsz -= esum2 * ecentz*ecentz;
    ermsz /= esum2;
    printf(" |  ecentr,z: %.2f %.2f   ermsr,z:  %.2f %.2f\n",
           grid*ecentr, grid*ecentz, grid*sqrt(ermsr), grid*sqrt(ermsz));

    printf("n: %3d  hsums: %6.0f %6.0f ratio: %.5f %.5f", n, hsum1, hsum2, hsum1/hsum2, hsum1/hsum02);
    hcentr /= hsum2;
    hrmsr -= hsum2 * hcentr*hcentr;
    hrmsr /= hsum2;
    hcentz /= hsum2;
    hrmsz -= hsum2 * hcentz*hcentz;
    hrmsz /= hsum2;
    printf(" |  hcentr,z: %.2f %.2f   hrmsr,z:  %.2f %.2f\n\n",
           grid*hcentr, grid*hcentz, grid*sqrt(hrmsr), grid*sqrt(hrmsz));
    if (n==1) {
      esum01 = esum1; esum02 = esum2;
      hsum01 = hsum1; hsum02 = hsum2;
    }
    sig[n]     = 1000.0 * ((hsum1 - hsum01) / hsum02 - (esum1 - esum01) / esum02);
    //sig[800+n/5] = 1000.0 * (hsum1 - hsum01) / hsum02;
    //sig[900+n/5] = 1000.0 * (esum01 - esum1) / esum02;
    printf("Signal collected is %.4f\n\n", sig[n]/1000);

  }

  /* do RC integration for preamp risetime */
  if (setup.preamp_tau > 1) {
    rc_integrate(sig, sig, setup.preamp_tau/setup.step_time_out, n);
    for (n=0; n<1024-1; n++) sig[n] = sig[n+1];
  }

  //write_spectrum(sig, 1024, "ehds.spe");

//modified by Kevin Bhimani to write data to a text file

  printf("signal: \n");
  for (i = 0; i < 400; i++){
    printf("%.3f ", sig[i]/1000);
    if (i%10 == 9) printf("\n");
    }
  printf("\n");
  
  printf("Done printing. Attempting to write\n");

  int written = 0;
  char filename[1000];
  sprintf(filename, "/nas/longleaf/home/kbhimani/siggen_ccd/waveforms/%s/q=%.2f/signal_r=%.2f_phi=0.00_z=%.2f.txt",det_name,setup.impurity_surface, alpha_r_mm, alpha_z_mm);
  //printf("The file name is %s\n", filename);
  FILE *f = fopen(filename,"w");
  //written = fwrite(sig, sizeof(float), sizeof(sig), f);
  for(i = 0; i <400; i++){
    written = fprintf(f,"%f\n",sig[i]/1000);
   }  
  if (written == 0) {
    printf("Error during writing to file !");
  }
  fclose(f);
  //Kevin's modifications end here

printf("Done writting\n");

  return 0;
  
}


/* -------------------------------------- read_rho ------------------- */
int read_rho(int L, int R, float grid, float **rho, char *fname) {

  int    i, j;
  float  r, z;
  FILE  *file;
  char   line[256];

  if (!(file = fopen(fname, "r"))) {
    printf("ERROR: Cannot open file %s for electron density...\n", fname);
    return 1;
  } else {
    printf("Reading electron density from file %s\n", fname);
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

int ehd_field_setup(MJD_Siggen_Setup *setup){

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

  if (setup_wp(setup) != 0){
    printf("Failed to read weighting potential from file %s\n", setup->wp_name);
    return -1;
  }

  return 0;
}

int rc_integrate(float *s_in, float *s_out, float tau, int time_steps){
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

struct spe_header{
  int reclA;            /* 24 */
  unsigned int title[2]; /*don't know why this is uint, but seems to
                           work, so...*/ 
  int dim;
  int a1;               /*  1 */
  int a2;               /*  1 */
  int a3;               /*  1 */
  int reclB;            /* 24 */
};
int write_spectrum(float *spec, int nchs, char *name){  
  FILE *fp;
  int record_length;
  struct spe_header header;
  char *suffix_start;
  char *fname_start;

  header.reclA = header.reclB = 24; 
  header.title[0] = header.title[1] = 0;
  header.a1 = header.a2 = header.a3 = 1;

  fp = fopen(name,"w");
  if (fp == NULL){
    fprintf(stderr,"Error! Unable to open spectrum file %s \n",name);
    return 0;
  }
  header.dim = nchs;
  if ((suffix_start = strstr(name,".spe")) == NULL){
    suffix_start = name + strlen(name);
  }
  if ((fname_start = rindex(name,'/')) == NULL){
    fname_start = name; 
  }else{
    fname_start++;/*get rid of the '/'*/
  }
  if (suffix_start - fname_start < 8){
    memcpy(header.title,"       ",8);/*blank the title*/
    memcpy(header.title,fname_start,suffix_start - fname_start);
  }else{ 
    memcpy(header.title,suffix_start - 8,8);
  }
  record_length = sizeof(float)*header.dim;

  fwrite(&header, sizeof(header), 1, fp);
  fwrite(&record_length, sizeof(record_length), 1, fp);
  fwrite(spec, sizeof(float), nchs, fp); 
  fwrite(&record_length, sizeof(record_length), 1,fp);
  fclose(fp);

  return 1;
}
