#!/bin/bash
# source /global/homes/k/kbhimani/.bash_profile.ext

#make sure to keep these at 2 decimal point
radius=$1 
zPos=$2
detector=$3
surface_charge=$4
energy=$5
save_rho=$6
self_repulsion=$7
grid=$8

config_file="config_files/$detector.config"
config_file_calc_wp="config_files/${detector}_calc_wp.config"

dir_save="/pscratch/sd/k/kbhimani/siggen_ccd_data"
dir_run="/global/homes/k/kbhimani/siggen_ccd"

echo "running simulation at radius $radius and z position $zPos for detector $detector with surface charge $surface_charge with energy $energy and grid $grid. Write densities=$save_rho and Self repulsion=$self_repulsion"

#1> /dev/null supresses output from the command

mkdir $dir_save/${energy}_keV > $null
mkdir $dir_save/${energy}_keV/grid_$grid > $null
mkdir $dir_save/${energy}_keV/grid_$grid/self_repulsion_$self_repulsion > $null
mkdir $dir_save/${energy}_keV/grid_$grid/self_repulsion_$self_repulsion/$detector > $null
mkdir $dir_save/${energy}_keV/grid_$grid/self_repulsion_$self_repulsion/$detector/q=${surface_charge} > $null
mkdir $dir_save/${energy}_keV/grid_$grid/self_repulsion_$self_repulsion/$detector/q=${surface_charge}/drift_data_r=${radius}_z=${zPos} > $null

mkdir $dir_run/waveforms > $null
mkdir $dir_run/waveforms/${energy}_keV > $null
mkdir $dir_run/waveforms/${energy}_keV/grid_$grid > $null
mkdir $dir_run/waveforms/${energy}_keV/grid_$grid/self_repulsion_$self_repulsion > $null
mkdir $dir_run/waveforms/${energy}_keV/grid_$grid/self_repulsion_$self_repulsion/$detector > $null
mkdir $dir_run/waveforms/${energy}_keV/grid_$grid/self_repulsion_$self_repulsion/$detector/q=${surface_charge} > $null

$dir_run/ehdrift $config_file -a $radius -z $zPos -g $detector -s $surface_charge -e $energy -v $save_rho -f $self_repulsion -h $grid 1> /dev/null
#time $dir_run/ehdrift $config_file -a $radius -z $zPos -g $detector -s $surface_charge -e $energy -v $save_rho -f $self_repulsion 
#time $dir_run/ehd_siggen $config_file -a $radius -z $zPos -g $detector -s $surface_charge