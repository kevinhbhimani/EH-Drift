#!/bin/bash
# Function to run a simulation
run_simulation() {
    local sur_ch=$1
    local sur_drift=$2
    local pass_thick=$3
    local variation_type=$4

    # Calculate weighting potential if varying surface charge
    if [ "$variation_type" == "varying_surface_charge" ]; then
        echo "Calculating weighting potential for surface charge $sur_ch."
        # Uncomment to run simulation
        $dir_run/ehdrift $config_file -p 1 -g $detector -s $sur_ch -h $grid
    fi

    echo "Running simulation for $variation_type: surface charge=$sur_ch, surface drift=$sur_drift, passivated thickness=$pass_thick."
    # Uncomment to run simulation
    $dir_run/ehdrift $config_file -r $r_val -z $z_val -p 0 -g $detector -s $sur_ch -c $sur_drift -m $pass_thick -e $e_val -v $save_rho -f $self_repulsion -h $grid # > /dev/null
    
    new_filename="r${r_val}_z${z_val}_${variation_type}_sd${sur_drift}_pt${pass_thick}_sc${sur_ch}.h5"
    move_to_dir="${dir_save_base}/${detector}/sc_pt_sc_study"
    mkdir -p "$move_to_dir"
    # Uncomment to move the file
    mv "${dir_run}/waveforms/${detector}/${detector}_waveforms.h5" "$move_to_dir/$new_filename"
    
    echo "Moving to: $move_to_dir/$new_filename"
    ((sim_count++))
}
detector="P42575A"
grid=0.0200
save_rho=0
self_repulsion=1

config_file="config_files/$detector.config"
dir_save_base="/nas/longleaf/home/kbhimani/siggen_ccd/waveforms"
dir_run="/nas/longleaf/home/kbhimani/siggen_ccd"

e_val=5000.00
z_val=0.02
r_val=15.00


# Fixed values
fixed_sur_drift=0.01
fixed_pass_thick=0.002
fixed_sur_ch=-0.30

surface_charge_vals=(0.00 -0.005 -0.010 -0.020 -0.030 -0.040 -0.050 -0.060 -0.070 -0.080 -0.100 -1.000 -2.000)
surface_drift_vals=(0.100 0.050 0.020 0.010 0.002 0.001 0.0005 1.0000)
pass_thickness_vals=(0.0010 0.0020 0.0030 0.0040 0.0005 0.0060 0.0080 0.0100 0.0120 0.0150)

sim_count=0
cd $dir_run

# # Varying surface charge
# echo "Varying Surface Charge"
# for sur_ch in "${surface_charge_vals[@]}"; do
#     run_simulation $sur_ch $fixed_sur_drift $fixed_pass_thick "varying_surface_charge"
# done

# # Varying surface drift factor
# echo "Varying Surface Drift Factor"
# for sur_drift in "${surface_drift_vals[@]}"; do
#     run_simulation $fixed_sur_ch $sur_drift $fixed_pass_thick "varying_surface_drift"
# done

# Varying passivated thickness
echo "Varying Passivated Thickness"
for pass_thick in "${pass_thickness_vals[@]}"; do
    run_simulation $fixed_sur_ch $fixed_sur_drift $pass_thick "varying_pass_thickness"
done

echo "Total number of simulations ran: $sim_count"

cd "/nas/longleaf/home/kbhimani/siggen_ccd/Analysis/sd_pt_sc_study"
