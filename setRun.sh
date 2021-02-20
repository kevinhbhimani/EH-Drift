#!/bin/bash
source /global/homes/k/kbhimani/.bash_profile.ext

#make sure to keep these at 2 decimal point
radius=$1 
zPos=$2
detector=$3
surface_charge=$4

config_file="config_files/$detector.config"
config_file_calc_wp="config_files/${detector}_calc_wp.config"

echo "running simulation at radius $radius and z position $zPos for detector $detector with surface charge $surface_charge"

#1> /dev/null supresses output from the command

mkdir /global/cscratch1/sd/kbhimani/siggen_sims/$detector
mkdir /global/cscratch1/sd/kbhimani/siggen_sims/$detector/q=${surface_charge}
mkdir /global/cscratch1/sd/kbhimani/siggen_sims/$detector/q=${surface_charge}/drift_data_r=${radius}_z=${zPos}

mkdir /global/cscratch1/sd/kbhimani/siggen_sims/waveforms/$detector
mkdir /global/cscratch1/sd/kbhimani/siggen_sims/waveforms/$detector/q=${surface_charge}

/global/homes/k/kbhimani/siggen_2d/ehdrift $config_file -a $radius -z $zPos -g $detector -s $surface_charge 1> /dev/null
/global/homes/k/kbhimani/siggen_2d/ehd_siggen $config_file -a $radius -z $zPos -g $detector -s $surface_charge 1> /dev/null

echo "done!"