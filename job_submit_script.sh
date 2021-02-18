for radius in $(seq -f "%0.2f" 5.00 5.00 29.00)
do
  for z_pos in $(seq -f "%0.2f" 0.02 0.04 2.00)
  do
      echo sbatch set_Slurm.slr $radius $z_pos P42575A -0.5
  done
done

# The passivation layer is typically 0.1 μm thick​ but surface roughness and damage from the passivation process could be 2-10 μm
config_file_calc_wp="config_files/${detector}_calc_wp.config"

# Calculates  the weighting potential so we do have to do it all the time
/global/homes/k/kbhimani/siggen_2d/ehdrift config_files/P42575A_calc_wp.config -g P42575A -s 0.00
sbatch set_Slurm.slr 15.00 0.01 P42575A 0.00 
./ehd_siggen config_files/P42575A.config -a 15.00 -z 0.01 -g P42575A -s 0.00

# sbatch set_Slurm.slr <radius> <z Position> <detector ID> <Surface_Charge>
sbatch set_Slurm.slr 15.00 0.02 P42575A 0.50 
sbatch set_Slurm.slr 15.00 0.06 P42575A 0.50 
sbatch set_Slurm.slr 15.00 0.10 P42575A 0.50
sbatch set_Slurm.slr 15.00 0.14 P42575A 0.50 
sbatch set_Slurm.slr 15.00 0.18 P42575A 0.50 
sbatch set_Slurm.slr 15.00 0.20 P42575A 0.50

sbatch set_Slurm.slr 15.00 1.00 P42575A 0.50 
sbatch set_Slurm.slr 15.00 2.00 P42575A 0.50 
sbatch set_Slurm.slr 15.00 3.00 P42575A 0.50 
sbatch set_Slurm.slr 15.00 4.00 P42575A 0.50 
sbatch set_Slurm.slr 15.00 5.00 P42575A 0.50


-a $radius -z $zPos -g $detector -s $surface_charge
./ehd_siggen config_files/P42575A.config -a 15.00 -z 0.10 -g P42575A -s 0.00
./ehd_siggen config_files/P42575A.config -a 15.00 -z 1.00 -g P42575A -s 0.00
./ehd_siggen config_files/P42575A.config -a 15.00 -z 2.00 -g P42575A -s 0.00
./ehd_siggen config_files/P42575A.config -a 15.00 -z 3.00 -g P42575A -s 0.00
./ehd_siggen config_files/P42575A.config -a 15.00 -z 4.00 -g P42575A -s 0.00
./ehd_siggen config_files/P42575A.config -a 15.00 -z 5.00 -g P42575A -s 0.00