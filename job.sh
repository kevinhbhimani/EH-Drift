#!/bin/bash

# Longleaf slurm submission script            
# MAKE SURE THAT ALL NUMBERS ARE AT TWO DECIMAL POINTS, GRID HAS TO BE TO FOUR DECIMALS
# Check job status: squeue -u kbhimani (ONYEN)
#--------------------------------------------------------------------------      
#            
#SBATCH --job-name=khb-siggen2d                                                               
#SBATCH --output=/nas/longleaf/home/kbhimani/siggen_ccd/logs/job_log_.sh.o%j       
#SBATCH --error=/nas/longleaf/home/kbhimani/siggen_ccd/logs/job_error_log_.sh.o%j           
#SBATCH --partition=volta-gpu # a100-gpu or volta-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
# SBATCH --mem-per-cpu=32g
#SBATCH --time=0-12:00:00                                       
#SBATCH --nodes=1
# uncomment the lines below to get email notification about your job                       
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=kevin_bhimani@unc.edu
#---------------------------------------------------------------------------   
source /nas/longleaf/home/kbhimani/.bash_profile
make clean
make
# detector="UNC_scanner"
detector="P42575A"
save_rho=0
self_repulsion=1
write_wp=0
grid=0.0200

config_file="config_files/$detector.config"
dir_save="/work/users/k/b/kbhimani/siggen_ccd_data"
dir_run="/nas/longleaf/home/kbhimani/siggen_ccd"

# e_vals=($(seq 1500.00 50.00 5500.00))
# e_vals=(30.20 7.50)
# r_vals=($(seq 3.00 2.00 27.00))
r_vals=($(seq 27.00 2.00 33.00))
# e_vals=(60.00 5000.00)
# e_vals=($(seq 1500.00 50.00 5500.00))
# z_vals=($(seq 0.02 0.08 3.00))
# z_vals=(0.02 0.03 0.04 0.05 0.20 0.30 0.40 0.50 0.60 0.70 0.75 0.80 0.90 1.00 2.00 3.00)
# surface_charge_vals=(-0.050 -0.500 0.00 0.010 0.05 0.500)
# sd_vals=(0.0050 0.0030 0.0020 0.0010 0.0100)
# sd_vals=(0.0050)
# grid_array=(0.0550 0.0600 0.0700 0.0800 0.0500 0.0300 0.0350 0.0400 0.0450 0.0500 0.0250 0.0200 0.0100 0.0050 0.0150)
#1> /dev/null supresses output from the command
sim_count=0


# for sur_ch in "${surface_charge_vals[@]}"; do
#     echo "Calculating weighting potential, wp=1, det=$detector, sc=$sur_ch, grid=$grid"
#     $dir_run/ehdrift $config_file -p 1 -g $detector -s $sur_ch -h $grid 1> /dev/null
#     for e_var in "${e_vals[@]}"; do
#         for sd_var in "${sd_vals[@]}"; do
#             for z_var in "${z_vals[@]}"; do
#                 for r_var in "${r_vals[@]}"; do
#                     echo "Running simulation at r=$r_var, z=$z_var, det=$detector, sc=$sur_ch, eng=$e_var, grid=$grid, sd=$sd_var, wd=$save_rho, sr=$self_repulsion, count=$sim_count"
#                     $dir_run/ehdrift $config_file -r $r_var -z $z_var -p 0 -g $detector -s $sur_ch -e $e_var -c $sd_var -h $grid -v $save_rho -f $self_repulsion 1> /dev/null
#                     sim_count=$((sim_count+1))
#                 done
#             done
#         done
#     done
# done

z_vals=(0.02)
e_vals=(5000.00)
# surface_charge_vals=(-0.450 -0.500 -0.550 -0.600 -0.650 -0.700 -0.750)
# surface_charge_vals=(-0.800 -0.850 -0.900 -0.950 -1.000 -1.050 -1.100 -1.150 -1.200 -1.300 -1.400 -1.500)
surface_charge_vals=(0.550 0.600 0.650 0.700 0.800 0.900 1.000 1.100 1.500 2.000)
sd_vals=(0.0030 0.0025 0.0020 0.0025 0.0015 0.0010)
for sur_ch in "${surface_charge_vals[@]}"; do
    echo "Calculating weighting potential, wp=1, det=$detector, sc=$sur_ch, grid=$grid"
    $dir_run/ehdrift $config_file -p 1 -g $detector -s $sur_ch -h $grid 1> /dev/null
    for e_var in "${e_vals[@]}"; do
        for sd_var in "${sd_vals[@]}"; do
            for z_var in "${z_vals[@]}"; do
                for r_var in "${r_vals[@]}"; do
                    echo "Running simulation at r=$r_var, z=$z_var, det=$detector, sc=$sur_ch, eng=$e_var, grid=$grid, sd=$sd_var, wd=$save_rho, sr=$self_repulsion, count=$sim_count"
                    $dir_run/ehdrift $config_file -r $r_var -z $z_var -p 0 -g $detector -s $sur_ch -e $e_var -c $sd_var -h $grid -v $save_rho -f $self_repulsion 1> /dev/null
                    sim_count=$((sim_count+1))
                done
            done
        done
    done
done

echo "Calculations finished. Number of sims ran was $sim_count!"
