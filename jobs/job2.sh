#!/bin/bash

# Longleaf slurm submission script            
# Check job status: squeue -u kbhimani (ONYEN)
#--------------------------------------------------------------------------      
#            
#SBATCH --job-name=ehdrift2
#SBATCH --output=/nas/longleaf/home/kbhimani/siggen_ccd/jobs/logs/job_log.sh.o%j     
#SBATCH --error=/nas/longleaf/home/kbhimani/siggen_ccd/jobs/logs/job_error_log.sh.o%j 
#SBATCH --partition=a100-gpu # a100-gpu or volta-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH --time=1-00:00:00                                       
#SBATCH --nodes=1
# uncomment the lines below to get email notification about your job                       
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=kevin_bhimani@unc.edu
#---------------------------------------------------------------------------   
source /nas/longleaf/home/kbhimani/.bash_profile
module load cuda/11.2
detector=$1
save_rho=0
self_repulsion=1
grid=0.0200
sd_val=0.001
run_time=16000
sample_rate=16

dir_save="/work/users/k/b/kbhimani/siggen_ccd_data"
dir_run="/nas/longleaf/home/kbhimani/siggen_ccd"

# config_file = "/nas/longleaf/home/kbhimani/siggen_ccd/config_files/siggen_configs/P00662C_siggen.config"

config_file="$dir_run/config_files/siggen_configs/${detector}_siggen.config"
echo "Config file:" $config_file

z_vals=(0.02 0.04 0.08 0.16 0.32 0.64 1.28 2.56)
# r_vals=(2.00 5.00 8.00 11.00 14.00 17.00 20.00 23.00 26.00 29.00 32.00 35.00 38.00 41.00)
r_vals=(11.75) #for ICPC ditch
surface_charge_vals=(-0.30 0.30 0.00)
# to add: 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2
e_vals=(5000) #2039
#1> /dev/null supresses output from the command
sim_count=0

for sur_ch in "${surface_charge_vals[@]}"; do
    echo "Calculating weighting potential, wp=1, det=$detector, sc=$sur_ch, grid=$grid"
    # echo "$dir_run/ehdrift $config_file -p 1 -g $detector -s $sur_ch -h $grid"
    $dir_run/ehdrift $config_file -p 1 -g $detector -s $sur_ch -h $grid 1> /dev/null
    for e_var in "${e_vals[@]}"; do
        for z_var in "${z_vals[@]}"; do
            for r_var in "${r_vals[@]}"; do
                echo "Running simulation at r=$r_var, z=$z_var, det=$detector, sc=$sur_ch, eng=$e_var, grid=$grid, sd=$sd_val, wd=$save_rho, sr=$self_repulsion, count=$sim_count"
                # echo "$dir_run/ehdrift $config_file -r $r_var -z $z_var -p 0 -g $detector -s $sur_ch -e $e_var -c $sd_val -h $grid -v $save_rho -f $self_repulsion -t $run_time -u $sample_rate"
                $dir_run/ehdrift $config_file -r $r_var -z $z_var -p 0 -g $detector -s $sur_ch -e $e_var -c $sd_val -h $grid -v $save_rho -f $self_repulsion -t $run_time -u $sample_rate 1> /dev/null
                sim_count=$((sim_count+1))
                
            done
        done
    done
done

echo "Calculations finished. Number of sims ran was $sim_count!"
