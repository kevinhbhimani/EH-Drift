#!/bin/bash
#
# Longleaf slurm submission script            
# execute as:                                           
#      sbatch set_slurm.slr <radius> <z Position> <detector ID> <Surface_Charge>
#       <energy in Kev><{0,1} (do_not/do write the density files) <{0,1} (do_not/do re-calculate field) <grid size>
#      MAKE SURE THAT ALL NUMBERS ARE AT TWO DECIMAL POINTS, GRID HAS TO BE TO FOUR DECIMALS
#      for example sbatch set_slurm_all.slr 15.00 0.10 P42575A 0.00 5000.00 0 1 0.0200

# Check job status: squeue -u kbhimani (ONYEN)
#--------------------------------------------------------------------------      
#SBATCH --job-name=khb_ehd_P42575A
#SBATCH --output=/nas/longleaf/home/kbhimani/siggen_ccd/logs/job_log_.sh.o%j       
#SBATCH --error=/nas/longleaf/home/kbhimani/siggen_ccd/logs/job_error_log_.sh.o%j     
#                                                                                                  
# directories to put log files ... so you know what is happening to your jobs   
#                                                                  
#SBATCH --output=/nas/longleaf/home/kbhimani/siggen_ccd/logs/job_log_.sh.o%j       
#SBATCH --error=/nas/longleaf/home/kbhimani/siggen_ccd/logs/job_error_log_.sh.o%j     
#             
#SBATCH --partition=volta-gpu # a100-gpu or volta-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
# SBATCH --mem-per-cpu=32g
#SBATCH --time=6-00:00:00                                       
# set the time for your job here. Job will terminate if run time exceeds it.
#SBATCH --nodes=1
# uncomment the lines below to get email notification about your job                       
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=kevin_bhimani@unc.edu
#---------------------------------------------------------------------------   
source /nas/longleaf/home/kbhimani/.bash_profile
module load cuda/11.2
make clean
make
# radius=$1 #make sure to keep these at 1 decimal point
# zPos=$2
detector="OPPI"
grid=0.0200
save_rho=0
self_repulsion=1

config_file="config_files/$detector.config"
dir_save="/work/users/k/b/kbhimani/siggen_ccd_data"
dir_run="/nas/longleaf/home/kbhimani/siggen_ccd"

# # Define energy, radius, height, and surface charge values
# e_vals=($(seq 5000.00 -300.00 2000.00))
# # r_vals=($(seq 2.00 2.00 16.00))
# # z_vals=($(seq 0.02 0.02 0.1))
# z_vals=(0.02 0.04 0.08 0.16 0.32 0.64 1.28 2.56)
# r_vals=(2.00 3.00 4.50 6.75 10.13 15.19)
# surface_charge_vals=(0.00 -0.10 -0.50)

e_vals=(17.80)
z_vals=(0.02 0.03 0.04 0.06 0.08 0.11 0.16 0.22 0.32 0.45 0.63 0.89 1.25 1.77 2.50)
r_vals=(0.30 2.10 3.90 5.70 7.50 9.30 11.10 12.90 14.70 16.50 18.30 20.10 21.90 23.70 25.50 27.30)
surface_charge_vals=(-0.03)


for sur_ch in "${surface_charge_vals[@]}"
do
    echo "Calculating weighting potential for detector $detector with surface charge $sur_ch and grid $grid repulsion=$self_repulsion"
    # echo $dir_run/ehdrift $config_file -p 1 -g $detector -s $sur_ch -h $grid
    $dir_run/ehdrift $config_file -p 1 -g $detector -s $sur_ch -h $grid 1> /dev/null
    for e_var in "${e_vals[@]}"
    do
        for z_var in "${z_vals[@]}"
        do
            for r_var in "${r_vals[@]}"
            do
                echo "Running simulation at radius $r_var and z position $z_var for detector $detector with surface charge $sur_ch with energy $e_var and grid $grid. Write densities=$save_rho and Self repulsion=$self_repulsion"
                # echo $dir_run/ehdrift $config_file -r $r_var -z $z_var -p 0 -g $detector -s $sur_ch -e $e_var -v $save_rho -f $self_repulsion -h $grid
                $dir_run/ehdrift $config_file -r $r_var -z $z_var -p 0 -g $detector -s $sur_ch -e $e_var -v $save_rho -f $self_repulsion -h $grid 1> /dev/null
                sim_count=$((sim_count+1))
            done
        done
    done
done

echo "Total number of simulations ran $sim_count"
