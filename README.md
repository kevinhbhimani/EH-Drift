
# `EH-Drift Simulations`

## Overview
EH-Drift simulations offer a novel approach to simulating surface events in High Purity Germanium (HPGe) detectors, initially developed by David Radford and further expanding upon the Siggen simulations for the Majorana Demonstrator (MJD). EH-Drift is distinguished by its ability to simulate diffusion and self-repulsion effects at a granular level, providing a pixel-by-pixel account of charge densities. This method enables the modeling of non-spherical charge clouds and the slow drift of charges on detector's passivated surfaces.

## Key Features

- **2D Simulation with φ Symmetry**: `ehdrift` operates in r and z dimensions while assuming φ symmetry. This approach simulates charge movement as a ring, providing a 2D representation that aligns well with the behavior of computationally expensive 3D charge clouds.

- **Diffusion and Self-Repulsion Incorporation**: The program models charge carrier movement in the detector by incorporating both diffusion and electrostatic self-repulsion effects. The diffusion coefficient is calibrated to match known values for 3D charge clouds. Additionally, the simulation modulates the total charge to align self-repulsion effects with established simulations, particularly for alpha particle interactions.

- **Charge Density Tracking**: `ehdrift` tracks charge densities at each grid point. This enables the accurate simulation of nonspherical charge cloud shapes and movements, offering a high-resolution view of charge distribution within the detector.

- **Surface Charge Effects Integration**: The impact of surface charges, crucial for accurately modeling near-surface events like surface alpha interactions, is fully integrated.

- **Field Recalculations**: The simulation continuously recalculates electric potentials in response to charge movements. This approach captures the changing electric field within the detector due to movement of large charge clouds.

- **Optimized CUDA C++ Utilization**: The simulation leverages the power of GPUs through CUDA C++, significantly enhancing computational efficiency. By running the entire simulation loop on the GPU, `ehdrift` minimizes the need for memory transfers between the CPU and GPU at each step. This GPU-powered approach ensures grid-independent runtime, facilitating the execution of thousands of events required for background modeling simulations.

- **Dynamic Time Step Utilization**: The time step is determined by the Courant number, satisfying the CFL condition. The time step is dynamically adjusted when there is no more significant charge collection, enabling the efficient simulation of long signal waveforms.

## Simulation Workflow
1. **Initial Setup**: Configure the detector grid based on factors such as detector geometry, impurity concentration, surface charge, and bias voltage.
2. **Grid Division and Charge Distribution**: Divide the detector into a fine grid and set initial charge densities based on the impact energy of the particle.
3. **Boundary Condition Setting**: Establishes conditions according to the detector geometry, impurity concentration, surface charge, and bias voltage.
4. **GPU Setup**: Initialize and transfer memory pointers to the GPU. Utilizes modular arithmetic to distribute and index grid points into blocks and threads.
5. **Potential Calculation**: Computes electric and weighting potentials using an over-relaxation algorithm, incorporating estimates for capacitance and depletion. This calculation is performed using the Red-Black Successive Over-Relaxation Algorithm on the GPU.
6. **Charge Drift and Diffusion**: Allows charges to diffuse and drift in the calculated electric field over small time steps. These operations are conducted on the GPU using Atomic Operations.
7. **Surface Drift Modeling**: Models charges that reach the passivated surface, drifting at a reduced velocity and influencing signal formation.
8. **Impurity Distribution Update**: Updates the net impurity distribution to reflect the movement of charges, necessitating a recalculation of electric potentials.
9. **Data Collection**: Generates signals on the GPU using parallel reduction techniques and stores them at iterations defined in the config file. Snapshots of charge densities at specific time steps can be recorded for creating GIFs of charge cloud movements.

## Configuration
The configuration file is required for setting up `ehdrift` simulations. It allows users to define a wide range of parameters including detector geometry, electric field characteristics, simulation settings, and file paths. The file uses a key-value format, with each line representing a distinct parameter and its corresponding value. Comments can be included using the `#` symbol. Example configuration files are provided to guide users in accurately setting up their simulations. Refer to these examples for a comprehensive understanding of how to customize your simulation environment.

## Hardware Requirements

- **CUDA-Enabled GPU**: GPU-accelerates the core computations in `ehdrift`, requiring a CUDA-enabled GPU.

- **RAM and Storage**: Adequate RAM is essential for efficient data processing and simulation. The amount of available memory will influence the minimum grid size that can be effectively utilized in simulations. Additionally, ensure sufficient storage capacity for saving the optional density snapshots.

## Compiling the Program

### Compilation Prerequisites
Before compiling `ehdrift`, ensure your system has the following prerequisites:

1. **GCC Compiler**: Necessary for compiling C/C++ code.
   - Installation can typically be completed using your operating system's package manager or by visiting the [GCC official website](https://gcc.gnu.org/).

2. **NVIDIA CUDA Toolkit**: Essential for compiling CUDA code and executing GPU-accelerated computations.
   - Download and installation instructions are available on [NVIDIA's official site](https://developer.nvidia.com/cuda-downloads).

### Compile the Program
Users must first specify their GPU architecture in the Makefile. Then compiling `ehdrift` is straightforward. Open a terminal, navigate to the directory containing the program's files, and run the following command:

```bash
make
```

## Running the program
Once compiled, the `ehdrift` program can be executed from the terminal. The following command-line flags are available to customize the simulation:

- `-r`: Set the r position of event in mm.
- `-z`: Set the z position of event in mm.
- `-g`: Define the detector name.
- `-s`: Set the surface charge in 1e10 e/cm².
- `-e`: Input the interaction energy in KeV.
- `-v {0,1}`: Decide whether to write the density files (0 = no, 1 = yes).
- `-f {0,1}`: Choose whether to re-calculate the field (0 = no, 1 = yes).
- `-w {0,1}`: Choose whether to write the field file (0 = no, 1 = yes).
- `-d {0,1}`: Decide whether to write the depletion surface (0 = no, 1 = yes).
- `-p {0,1}`: Choose whether to write the Weighting Potential (WP) file (0 = no, 1 = yes).
- `-b`: Set bias voltage in volts.
- `-h`: Specify the grid size in mm.
- `-m`: Define the passivated surface depth size in mm.
<<<<<<< HEAD
=======
- `-c`: Specify the velocity in surface compared to the bulk
>>>>>>> 7a51b11 (Updates on anlysis)
- `-a`: Input a custom impurity density profile file.

First, run the program to calculate the weighting potential (WP), which can then be reused for different r and z values. To do this:

1. Run the program with the WP file flag set to one to only calculate the WP and exit:

```bash
./ehdrift config_files/P42575A.config -p 1 -s -0.50 -h 0.0200
```
This process calculates and saves the detector's weighting potential with a surface charge of -0.50 using a 0.0500 grid. Weighting potential is unique for detector, surface charge and grid, and must be recalculated if any of the parameters are changed. 

2. Next, run the program to simulate a 5000 KeV event at r=15 mm and z=0.10 mm, and save the signal:
```bash
<<<<<<< HEAD
./ehdrift config_files/P42575A.config -r 15.00 -z 0.10 -p 0 -s -0.50 -e 5000 -h 0.0200
=======
./ehdrift config_files/P42575A.config -r 15.00 -z 0.02 -p 0 -s -0.50 -e 5000 -h 0.0200
>>>>>>> 7a51b11 (Updates on anlysis)
```

### Runtime Performance
The runtime performance of generating 8000ns waveforms on A100 GPUs is summarized below for different grid.

#### 20 Micron Grid
- **Calculate Weighting Potential**: 1 minute and 1 second
- **Generating Signal**: 5 minutes and 7 seconds

#### 10 Micron Grid
- **Calculate Weighting Potential**: 4 minutes and 41 seconds
- **Generating Signal**: 24 minutes and 37 seconds

## Saving Outputs

The output signal is stored in an HDF5 file format in the specified directory set in the configuration file. For each run, the program checks for an existing HDF5 file for the given detector. If the file exists, new events are appended to it; otherwise, a new file is created.

### HDF5 File Structure

The HDF5 file stores the simulation results in a structured format. The file contains datasets for event data and waveforms, along with attributes that provide additional context and settings.

#### Datasets

- `event_data`: A compound dataset that includes information about each event. Each entry in this dataset contains the following fields:
  - `energy`: Interaction energy of the event.
  - `r`: Radial position of the event in the detector.
  - `z`: Height of the event within the detector.
  - `surface_charge`: Surface charge considered in the simulation.
  - `waveform`: A 1D array storing the normalized signal values for the event.

#### Attributes

The file also contains the following attributes at the root level, providing context for the entire dataset:

- `grid`: Grid size used in the simulation.
- `passivated_thickness`: Width of the passivated surface.
<<<<<<< HEAD
=======
- `surface_bulk_vel_factor`: How fast the charges move on the passivated surface compared to the bulk.
>>>>>>> 7a51b11 (Updates on anlysis)
- `self_repulsion`: Indicates whether self-repulsion effects were considered.
- `detector_name`: Name of the detector used in the simulation.

### Accessing Data

You can access the event data and waveform for each event by iterating through the `event_data` dataset. Each entry in this dataset corresponds to a unique event, with its waveform and other parameters.


## Analysis
The HDF5 file structure allows for detailed and event-specific data analysis. The following Python code example demonstrates how to iterate over HDF5 files to extract waveform data and associated parameters for each event:

```python
import os
import h5py
import pandas as pd
from tqdm import tqdm

# Directory containing the waveform files
<<<<<<< HEAD
directory = '/nas/longleaf/home/kbhimani/siggen_ccd/waveforms/P42575A/'
=======
directory = '/path/to/waveforms'
>>>>>>> 7a51b11 (Updates on anlysis)

# Initialize an empty list to store data
waveforms_data = []

# Iterate over files in the directory
for filename in tqdm(os.listdir(directory)):
    if filename.endswith('.h5') or filename.endswith('.hdf5'):
        with h5py.File(os.path.join(directory, filename), 'r') as file:
            # Check for 'event_data' dataset and 'waveforms' dataset in the file
            event_data = file['event_data']
            waveforms = event_data['waveform']
            # Extract file-level attributes
            grid = file.attrs['grid']
            passivated_thickness = file.attrs['passivated_thickness']
            self_repulsion = file.attrs['self_repulsion']
            detector_name_bytes = file.attrs['detector_name'][:]
            detector_name = detector_name_bytes.tobytes().decode('utf-8')
<<<<<<< HEAD

=======
            surface_bulk_vel_factor= file.attrs['surface_bulk_vel_factor']
>>>>>>> 7a51b11 (Updates on anlysis)
            # Iterate through each event in the file
            for i in range(event_data.shape[0]):
                # Extract parameters for each event
                eng, r, z, surface_charge = event_data[i]['energy'], event_data[i]['radius'], event_data[i]['height'], event_data[i]['surface_charge']

                # Extract waveform for each event
                waveform = waveforms[i]

                # Append to the list as a dictionary
                waveforms_data.append({
                    'r': r, 
                    'z': z, 
                    'eng': eng, 
                    'sc': surface_charge, 
                    'grid': grid, 
                    'pass_thickness': passivated_thickness, 
                    'self_repulsion': self_repulsion, 
                    'det': detector_name, 
                    'wf': waveform
<<<<<<< HEAD
=======
                    'sf_drift':surface_bulk_vel_factor
>>>>>>> 7a51b11 (Updates on anlysis)
                })
                
# Convert the list of dictionaries to a DataFrame
waveforms_df = pd.DataFrame(waveforms_data)
# Remove null character from 'det' column
waveforms_df['det'] = waveforms_df['det'].apply(lambda x: x.strip('\x00'))
waveforms_df.head()
```
Waveform dataframe can now be used for plotting or quering
```python
# Example: Query for a specific waveform
r_exp= 15.00
<<<<<<< HEAD
z_exp=0.1
=======
z_exp=0.02
>>>>>>> 7a51b11 (Updates on anlysis)
sc_exp=-0.5
eng_exp= 5000
det_exp = 'P42575A'
grid_exp = 0.02

query = f"r == {r_exp} and z == {z_exp} and sc == {sc_exp} and eng == {eng_exp} and det == '{det_exp}'  and grid == {grid_exp}"
specific_waveform_row = waveforms_df.query(query).iloc[0]['wf']

sim_time=8000
step_time_out = 10
time = np.linspace(start=step_time_out, stop= sim_time, num= (int) (sim_time/step_time_out))

plt.plot(time, specific_waveform_row)
plt.xlabel('Time (ns)')
plt.ylabel('Normalized Signal')
```

## References

1. "Surface Characterization of P-Type Point Contact Germanium Detectors," F. Edzards et al., *Particles*, vol. 4, no. 4, pp. 489-511, 2021. [DOI:10.3390/particles4040036](https://doi.org/10.3390/particles4040036). [arXiv:2105.14487](https://arxiv.org/abs/2105.14487)

2. "Simulation of Semiconductor Detectors in 3D with SolidStateDetectors.jl," I. Abt et al., *Journal of Instrumentation*, vol. 16, no. 08, p. P08007, Aug. 2021. [DOI:10.1088/1748-0221/16/08/P08007](https://doi.org/10.1088/1748-0221/16/08/P08007). GitHub Repository: [SolidStateDetectors.jl](https://github.com/JuliaPhysics/SolidStateDetectors.jl)

3. "The Performance Model for a Parallel SOR Algorithm Using the Red-Black Scheme," I. Epicoco, S. Mocavero, and G. Aloisio, *International Journal of High Performance Systems Architecture*, vol. 4, no. 2, pp. 101-109, 2012. [DOI:10.1504/IJHPSA.2012.050989](https://www.inderscienceonline.com/doi/abs/10.1504/IJHPSA.2012.050989). [Eprint](https://www.inderscienceonline.com/doi/pdf/10.1504/IJHPSA.2012.050989)

4. D.C. Radford, mjd_fieldgen and mjd_siggen Software. GitHub Repository: [icpc_siggen](https://github.com/radforddc/icpc_siggen)


## Contact and Support

For questions, feedback, or contributions to the ehdrift project, please feel free to reach out. You can contact us via email:

- **Kevin Bhimani**
  - Email: [kevin_bhimani@unc.edu](mailto:kevin_bhimani@unc.edu)
  - For: Technical queries, bug reports, and development contributions.

- **Julieta Gruszko**
  - Email: [jgruszko@unc.edu](mailto:jgruszko@unc.edu)
  - For: General inquiries, research collaboration, and project insights.
