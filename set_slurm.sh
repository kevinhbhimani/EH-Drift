#!/bin/bash
#
# Perlmutter cluster job slurm submission script            
# execute as:                                           
#      sbatch set_slurm.slr <radius> <z Position> <detector ID> <Surface_Charge>
#       <energy in Kev><{0,1} (do_not/do write the density files) <{0,1} (do_not/do re-calculate field) <grid size>
#      MAKE SURE THAT ALL NUMBERS ARE AT TWO DECIMAL POINTS, GRID HAS TO BE TO FOUR DECIMALS
#      for example sbatch set_slurm.slr 15.00 0.10 P42575A 0.00 5000.00 0 1 0.0200
#       ICPC sbatch set_slurm.slr 15.00 5.00 V01386A 0.00 5000.00 0 1 0.0200
# check job detail: jobstats or scontrol show job <JobID>  or squeue --job 37985561  <JobID> or sqs                 
#--------------------------------------------------------------------------      
#            
#SBATCH --job-name=khb-siggen2d
#                                                                                             
# directories to put log files ... so you know what is happening to your jobs   
#                                                                  
#SBATCH --output=/nas/longleaf/home/kbhimani/siggen_ccd/logs/job_log_.sh.o%j       
#SBATCH --error=/nas/longleaf/home/kbhimani/siggen_ccd/logs/job_error_log_.sh.o%j         
#SBATCH --account=majorana_g
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=regular
#SBATCH --time=1:00:00 
#SBATCH --ntasks=1                                        
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-task=1
# uncomment the lines below to get email notification about your job                       
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=kevin_bhimani@unc.edu
#---------------------------------------------------------------------------   
#!/bin/bash

radius=$1 #make sure to keep these at 1 decimal point
zPos=$2
detector=$3
surface_charge=$4
energy=$5
save_rho=$6
self_repulsion=$7
grid=$8

config_file="config_files/$detector.config"
config_file_calc_wp="config_files/${detector}_calc_wp.config"

dir_save="/work/users/k/b/kbhimani/siggen_ccd_data"
dir_run="/nas/longleaf/home/kbhimani/siggen_ccd"

echo "Calculating weighting potential at radius $radius and z position $zPos for detector $detector with surface charge $surface_charge with energy $energy and grid $grid. Write densities=$save_rho and Self repulsion=$self_repulsion"

$dir_run/ehdrift $config_file_calc_wp -r $radius -z $zPos -g $detector -s $surface_charge -e $energy -v $save_rho -f $self_repulsion -h $grid

# echo "time $dir_run/ehdrift $config_file_calc_wp -r $radius -z $zPos -g $detector -s $surface_charge -e $energy -v $save_rho -f $self_repulsion"
# time $dir_run/ehdrift $config_file_calc_wp -r $radius -z $zPos -g $detector -s $surface_charge -e $energy -v $save_rho -f $self_repulsion

bash $dir_run/set_run.sh $radius $zPos $detector $surface_charge $energy $save_rho $self_repulsion $grid
