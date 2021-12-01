#!/bin/bash
# source /global/homes/k/kbhimani/.bash_profile.ext

#make sure to keep these at 2 decimal point
radius=$1 
zPos=$2
detector=$3
surface_charge=$4

config_file="config_files/$detector.config"
config_file_calc_wp="config_files/${detector}_calc_wp.config"
dir_save="/pine/scr/k/b/kbhimani/siggen_sims"
dir_run="/nas/longleaf/home/kbhimani/siggen_ccd"

echo "running simulation at radius $radius and z position $zPos for detector $detector with surface charge $surface_charge"

#1> /dev/null supresses output from the command

mkdir $dir_save/$detector
mkdir $dir_save/$detector/q=${surface_charge}
mkdir $dir_save/$detector/q=${surface_charge}/drift_data_r=${radius}_z=${zPos}

mkdir $dir_run/waveforms/$detector
mkdir $dir_run/waveforms/$detector/q=${surface_charge}

time $dir_run/ehdrift $config_file_calc_wp -a $radius -z $zPos -g $detector -s $surface_charge
time $dir_run/ehdrift $config_file -a $radius -z $zPos -g $detector -s $surface_charge
time $dir_run/ehd_siggen $config_file -a $radius -z $zPos -g $detector -s $surface_charge

echo "done!"