
# `EH-Drift Simulations`

## Overview
EH-Drift simulations is a novel approach to simulating surface events in High Purity Germanium (HPGe) detectors. Intially developed by David Radford, these simulations expand upon the Siggen simulations developed for the Majorana Demonstrator (MJD). EH-Drift is distinguished by its ability to simulate diffusion and self-repulsion effects at a granular level, maintaining a pixel-by-pixel account of charge densities. This method enables the modeling of non-spherical charge clouds and the slow drift of charges on detector's passivated surfaces.

## Key Features

- **2D Simulation with φ Symmetry**: `ehdrift` operates in r and z dimensions while assuming φ symmetry. This approach simulates charge movement as a ring, providing a 2D representation that aligns well with the behavior of computationally expensive 3D charge clouds.

- **Diffusion and Self-Repulsion Incorporation**: The program models charge carrier movement in the detector by incorporating both diffusion and electrostatic self-repulsion effects. The diffusion coefficient is calibrated to match known values for 3D charge clouds. Additionally, the total charge in the simulation is modulated to align self-repulsion effects with established simulations, particularly for alpha particle interactions.

- **Charge Density Tracking**: `ehdrift` tracks charge densities at each grid point. This enables the accurate simulation of nonspherical charge cloud shapes and movements, offering a high-resolution view of charge distribution within the detector.

- **Surface Charge Effects Integration**: The impact of surface charges, crucial for near-surface event simulations, is integrated. This feature is particularly important for accurately modeling the behavior of surface alpha events on detector's passivated surface.

- **Field Recalculations**: Electric potentials are continuously recalculated in response to charge movements. This approach captures the changing electric field within the detector due to movement of large charge clouds.

- **Optimized CUDA C++ Utilization**: The simulation leverages the power of GPUs through CUDA C++, significantly enhancing computational efficiency. By executing the entire simulation loop on the GPU, `ehdrift` minimizes the need for memory transfers between the CPU and GPU at each step. This GPU-powered approach ensures grid-independent runtime, facilitating the execution of thousands of events required for background modeling simulations.

- **Dynamic Time Step Utilization**: The time step is determined by the Courant number, satisfying the CFL condition. The time step is dynamically adjusted when there is no more significant charge collection, enabling the efficient simulation of long signal waveforms.


## Simulation Workflow
1. **Initial Setup**: Configures the detector grid based on factors such as detector geometry, impurity concentration, surface charge, and bias voltage.
2. **Grid Division and Charge Distribution**: Divides the detector into a fine grid and sets initial charge densities based on the impact energy of the particle.
3. **Boundary Condition Setting**: Establishes conditions according to the detector geometry, impurity concentration, surface charge, and bias voltage.
4. **GPU Setup**: Initializes memory for pointers and transfers them to GPU memory. Utilizes modular arithmetic to distribute and index grid points into blocks and threads.
5. **Potential Calculation**: Computes electric and weighting potentials using an over-relaxation algorithm, incorporating estimates for capacitance and depletion. This calculation is performed using the Red-Black Successive Over-Relaxation Algorithm on the GPU.
6. **Charge Drift and Diffusion**: Allows charges to diffuse and drift in the calculated electric field over small time steps. These operations are conducted on the GPU using Atomic Operations.
7. **Surface Drift Modeling**: Models charges that reach the passivated surface, drifting at a reduced velocity and influencing signal formation.
8. **Impurity Distribution Update**: Updates the net impurity distribution to reflect the movement of charges, necessitating a recalculation of electric potentials.
9. **Data Collection**: Generates signals on the GPU using parallel reduction techniques and stores them at iterations defined in the config file. Snapshots of charge densities at specific time steps can be recorded for creating GIFs of charge cloud movements.

## Configuration
The configuration file is required for setting up `siggen_ccd` simulations. It allows users to define a wide range of parameters including detector geometry, electric field characteristics, simulation settings, and file paths. The file adopts a key-value format where each line represents a distinct parameter paired with its corresponding value. Comments can be included using the `#` symbol. Example configuration files are provided to guide users in accurately setting up their simulations. Refer to these examples for a comprehensive understanding of how to customize your simulation environment.

## Hardware Requirements

- **CUDA-Enabled GPU**: The core computations in `siggen_ccd` are GPU-accelerated, requiring a CUDA-enabled GPU. Users must specify their GPU architecture in the Makefile to align with their specific hardware capabilities.

- **RAM and Storage**: Adequate RAM is essential for efficient data processing and simulation. The amount of available memory will influence the minimum grid size that can be effectively utilized in simulations. Additionally, ensure sufficient storage capacity for saving the optional density snapshots.


## Compiling the Program

### Compilation Prerequisites
Before compiling `siggen_ccd`, ensure your system has the following prerequisites:

1. **GCC Compiler**: Necessary for compiling C/C++ code.
   - Installation can typically be completed using your operating system's package manager or by visiting the [GCC official website](https://gcc.gnu.org/).

2. **NVIDIA CUDA Toolkit**: Essential for compiling CUDA code and executing GPU-accelerated computations.
   - Download and installation instructions are available on [NVIDIA's official site](https://developer.nvidia.com/cuda-downloads).

### Compile the Program
Compiling `siggen_ccd` is straightforward. Open a terminal, navigate to the directory containing the program's files, and run the following command:

```bash
make
```

## Running the program
Once compiled, the `siggen_ccd` program can be executed from the terminal. The following command-line flags are available to customize the simulation:

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
- `-a`: Input the rho_spectrum_file_name.

It is recommended to first run the program to calculate the weighting potential (WP) and then reuse it for different r and z values. To do this:

1. Run the program with the WP file flag set to one to only calculate the WP and exit:

```bash
./ehdrift config_files/P42575A.config -p 1 -s -0.50 -h 0.0200
```
This caculated and saves the weighting potential of the detector with surface charge of -0.50 using grid of 0.0500. Weighting potential is unique for detector, surface charge and grid, and must be recalculated if any of the parameters are changed.

2. Next, run the program to simulate a 5000 KeV event and save the signal:
```bash
./ehdrift config_files/P42575A.config -r 15.00 -z 0.10 -p 0 -s -0.50 -e 5000 -h 0.0200
```
## Saving Outputs
The output signal from `siggen_ccd` is saved as HDF5 files, with the paths specified in the configuration file. The files adhere to a structured format for easy data retrieval and analysis.

### Root Group
In the root group `/`, the HDF5 file contains several datasets:

- `/waveform`: This is a one-dimensional array that stores the scaled signal values, representing the waveform data.

- `/energy`: A single value dataset that represents the interaction energy of the event.

- `/r`: A single value indicating the radial position of the event within the detector.

- `/z`: A dataset storing a single value, denoting the height or depth of the event within the detector.

- `/grid`: Contains a single value representing the grid size used in the simulation.

- `/detector_name`: A string that identifies the name of the detector used in the simulation.

### Attributes
Attached to the root group are several attributes providing additional context and settings used during the simulation:

- `surface_charge`: A single value indicating the surface charge considered in the simulation.

- `self_repulsion`: This attribute holds a boolean value, indicating whether self-repulsion effects were taken into account in the simulation.



## Contact and Support

If you have any questions, feedback, or would like to contribute to the `siggen_ccd` project, please feel free to reach out. We value your input and collaboration. You can contact us via email:

- **Kevin Bhimani**
  - Email: [kevin_bhimani@unc.edu](mailto:kevin_bhimani@unc.edu)
  - For: Technical queries, bug reports, and development contributions.

- **Julieta Gruszko**
  - Email: [jgruszko@unc.edu](mailto:jgruszko@unc.edu)
  - For: General inquiries, research collaboration, and project insights.
