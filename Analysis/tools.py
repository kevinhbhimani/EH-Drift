import os
import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
# from dspeed.processors.trap_filters import trap_norm
import random

def trap_norm(w_in: np.ndarray, rise: int, flat: int, w_out: np.ndarray) -> None:
    w_out[:] = np.nan  # Initialize output waveform with NaNs

    # Check if input waveform is too short
    if len(w_in) < rise:
        return  # Possibly handle this case differently depending on needs

    # Initial processing
    w_out[0] = w_in[0] / rise
    end_idx = min(len(w_in), rise)
    for i in range(1, end_idx):
        w_out[i] = w_out[i - 1] + w_in[i] / rise

    # Handling the flat part
    end_idx = min(len(w_in), rise + flat)
    for i in range(rise, end_idx):
        w_out[i] = w_out[i - 1] + (w_in[i] - w_in[i - rise]) / rise

    # Decrease part
    end_idx = min(len(w_in), 2 * rise + flat)
    for i in range(rise + flat, end_idx):
        if i - rise - flat >= 0:
            w_out[i] = w_out[i - 1] + (w_in[i] - w_in[i - rise] - w_in[i - rise - flat]) / rise

    # Final part
    for i in range(2 * rise + flat, len(w_in)):
        w_out[i] = (w_out[i - 1] + (w_in[i] - w_in[i - rise] - w_in[i - rise - flat] + w_in[i - 2 * rise - flat]) / rise)

def load_waveform_data(directory):
    """
    Load waveform data from HDF5 files in the specified directory and remove duplicates,
    excluding the 'wf' column from duplicate consideration.

    Parameters:
    - directory: Path to the directory containing the HDF5 waveform files.

    Returns:
    - A pandas DataFrame containing the loaded waveform data without duplicates.
    """
    waveforms_data = []  # Initialize an empty list to store data
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
                # Iterate through each event in the file
                for i in range(event_data.shape[0]):
                    # Extract parameters for each event
                    eng, r, z, surface_charge, vel_fact = event_data[i]['energy'], event_data[i]['radius'], event_data[i]['height'], event_data[i]['surface_charge'], event_data[i]['surface_bulk_vel_factor']
                    # Extract waveform for each event
                    waveform = waveforms[i]
                    # Append to the list as a dictionary
                    waveforms_data.append({
                        'r': r, 
                        'z': z, 
                        'eng': eng, 
                        'sc': surface_charge,
                        'sf_drift': round(vel_fact,5),
                        'grid': grid, 
                        'pass_thickness': passivated_thickness, 
                        'self_repulsion': self_repulsion, 
                        'det': detector_name, 
                        'wf': waveform,
                    })
    
    # Convert the list of dictionaries to a DataFrame
    waveforms_df = pd.DataFrame(waveforms_data)
    # Handle null characters in 'det' column
    waveforms_df['det'] = waveforms_df['det'].apply(lambda x: x.strip('\x00'))
    # Remove duplicate rows considering all columns except 'wf'
    columns_to_consider = [col for col in waveforms_df.columns if col != 'wf']
    waveforms_df = waveforms_df.drop_duplicates(subset=columns_to_consider)
    return waveforms_df


# def trap_filter(waveform, rise, flat):
#     """
#     Apply a trapezoidal filter to the input waveform.
    
#     Parameters:
#     - waveform: Input waveform array.
#     - rise: Rise time of the trapezoidal filter.
#     - flat: Flat top width of the trapezoidal filter.
    
#     Returns:
#     - The filtered waveform.
#     """
#     trap = np.zeros_like(waveform)
#     acc = 0
#     for i in range(1, len(waveform)):
#         acc += waveform[i]
#         if i >= rise:
#             acc -= waveform[i-rise]
#             trap[i] = acc
#         if i >= rise + flat:
#             acc -= waveform[i-rise-flat]
#             trap[i] = acc - trap[i-rise]
#     return trap

def apply_trap_filter_to_df(df, rise, flat, pickoff, plot=False):
    """
    Applies a trapezoidal filter to all waveforms in the DataFrame and optionally plots the filtered waveform.

    Parameters:
    - df: DataFrame containing the waveforms in a column named 'wf'.
    - rise: Rise time of the trapezoidal filter.
    - flat: Flat top duration of the trapezoidal filter.
    - pickoff: Sample point to pick off the filtered signal.
    - plot: If True, plots the original and filtered waveform for the first waveform in the DataFrame.

    Returns:
    - A modified DataFrame with a new column 'energy_estimate' containing the energy estimates.
    """
    energy_estimates = []

    for waveform in tqdm(df['wf'], desc='Applying Trap Filter'):
        if len(waveform) > 2 * rise + flat:
            trap_out = np.zeros_like(waveform)
            trap_norm(waveform, rise, flat, trap_out)
            if pickoff < len(trap_out):
                energy_estimate = trap_out[pickoff]
            else:
                energy_estimate = np.nan  # or some error/default value
        else:
            energy_estimate = np.nan  # Handle short waveform
        
        energy_estimates.append(energy_estimate)

    df['energy_estimate'] = energy_estimates

    if plot:
        # Plot the first waveform and its filtered version
        plt.figure(figsize=(10, 6))
        plt.plot(df.iloc[0]['wf'], label='Original Waveform')
        plt.plot(trap_filter(df.iloc[0]['wf'], rise, flat), label='Filtered Waveform', linestyle='--')
        plt.axvline(pickoff, color='r', linestyle=':', label='Pick-off Point')
        plt.legend()
        plt.title('Original vs. Trapezoidal Filtered Waveform')
        plt.xlabel('Time Steps')
        plt.ylabel('Amplitude')
        plt.show()
    return df

def compute_pickoff_values_and_plot_random(df, rise, flat, pickoff, plot_random=True):
    """
    Computes pickoff values for each waveform in the DataFrame after applying a trapezoidal filter.
    Optionally, plots a randomly selected waveform with its filtered signal and pickoff point.
    
    Parameters:
    - df: DataFrame containing the waveforms.
    - rise: Rise time for the trapezoidal filter.
    - flat: Flat time for the trapezoidal filter.
    - pickoff: Index to pick off the value from the filtered waveform.
    - plot_random: If True, plots a randomly selected waveform with filter and pickoff. Default is True.
    """
    # Initialize a column for pickoff values
    df['energy_estimate'] = np.nan
    
    for index, row in df.iterrows():
        waveform = row['wf']
        # Initialize output array for the filtered waveform
        filtered_wf = np.zeros_like(waveform)
        # Apply the trapezoidal filter
        trap_norm(waveform, rise, flat, filtered_wf)
        
        # Check if pickoff index is within the bounds of the filtered waveform
        if pickoff < len(filtered_wf):
            pickoff_val = filtered_wf[pickoff]
        else:
            pickoff_val = np.nan  # Handle cases where pickoff is out of bounds
        
        # Update the DataFrame
        df.at[index, 'energy_estimate'] = pickoff_val

    if plot_random:
        # Randomly select a waveform and its filtered version for plotting
        random_index = np.random.randint(0, len(df))
        waveform = df.iloc[random_index]['wf']
        filtered_wf = np.zeros_like(waveform)
        trap_norm(waveform, rise, flat, filtered_wf)
        
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(waveform, label='Original Waveform')
        plt.plot(filtered_wf, label='Filtered Waveform', linestyle='--')
        plt.axvline(pickoff, color='r', linestyle=':', label=f'Pickoff at {pickoff}')
        plt.legend()
        plt.title('Randomly Selected Waveform and Filtered Waveform with Pickoff')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.show()


def calculate_tail_slope(waveform, ti, tf, window=100):
    """
    Calculates the tail slope of a waveform by averaging samples around specified points
    and computing the slope between these averages.
    """
    if ti - window // 2 < 0 or tf + window // 2 >= len(waveform):
        raise ValueError("Time indices and window size must be within the bounds of the waveform length.")
    
    avg_initial = np.mean(waveform[ti - window // 2 : ti + window // 2])
    avg_final = np.mean(waveform[tf - window // 2 : tf + window // 2])
    slope = (avg_final - avg_initial) / (tf - ti)
    return slope

def add_tail_slope_to_dataframe(df, ti, tf, window=100, plot_random_waveform=False):
    """
    Adds a tail slope column to the input DataFrame based on the waveform data and
    optionally plots a randomly selected waveform with its tail slope line.
    """
    tail_slopes = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        waveform = row['wf']
        slope = calculate_tail_slope(waveform, ti, tf, window)
        tail_slopes.append(slope)
    df['tail_slope'] = tail_slopes

    if plot_random_waveform:
        # Randomly select a waveform and plot it along with the tail slope line
        random_index = random.randint(0, len(df) - 1)
        waveform = df.iloc[random_index]['wf']
        slope = df.iloc[random_index]['tail_slope']
        plt.figure(figsize=(10, 6))
        plt.plot(waveform, label='Waveform')
        # Plotting the tail slope line
        y_initial = np.mean(waveform[ti - window // 2 : ti + window // 2])
        y_final = slope * (tf - ti) + y_initial
        plt.plot([ti, tf], [y_initial, y_final], 'r--', label='Tail Slope')
        plt.title(f"Random Waveform with Tail Slope (Index: {random_index})")
        plt.legend()
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.show()
        
def query_and_plot_waveform(df, time, r_exp, z_exp, sc_exp, sd_exp, eng_exp, det_exp, grid_exp):
    """
    Query a specific waveform based on provided criteria and plot the original waveform.
    
    Parameters:
    - df: DataFrame containing the waveform data.
    - r_exp, z_exp, sc_exp, eng_exp, det_exp, grid_exp: Parameters for querying the specific waveform.
    - rise, flat: Parameters for the trapezoidal filter.
    - time: Array representing the time steps of the waveform.
    """
    # Construct the query string
    query = f"r == {r_exp} and z == {z_exp} and sc == {sc_exp} and sf_drift == {sd_exp} and eng == {eng_exp} and det == '{det_exp}' and grid == {grid_exp}"
    
    # Perform the query
    specific_waveform_row = df.query(query)

    # if not specific_waveform_row.empty:
    #     specific_waveform = specific_waveform_row.iloc[0]['wf']
    #     # Plotting the original waveform
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(time, specific_waveform, label='Original Waveform')
    #     # Optionally, plot the trapezoidal filter output
    #     plt.xlabel('Time (ns)')
    #     plt.ylabel('Normalized Signal')
    #     plt.title(f'Waveform Plot for r={r_exp}, z={z_exp}, sc={sc_exp}, sd={sd_exp}, eng={eng_exp}, det={det_exp}, grid={grid_exp}')
    #     # plt.legend()
    #     plt.show()
    # else:
    #     print("No waveform found matching the criteria.")
    return specific_waveform