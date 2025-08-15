# %%
hdf5_data_path = '/scratch/ty296/hdf5_data/'
groupname = 'L'
p_fixed_name = 'p_ctrl'
p_fixed_value = 0.4




# %%
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

def von_neumann_entropy_sv(sv_arr: np.ndarray, n: int = 1, positivedefinite: bool = False, threshold: float = 1e-16) -> float:
    """
    Compute von Neumann entropy from singular values.
    
    Parameters:
    - sv_arr: array of singular values
    - n: Renyi entropy parameter (n=1 for von Neumann)
    - positivedefinite: if True, treat sv_arr as probabilities; if False, square them first
    - threshold: minimum value to avoid log(0)
    
    Returns:
    - Entropy value
    """
    if positivedefinite:
        p = np.maximum(sv_arr, threshold)
    else:
        p = np.maximum(sv_arr, threshold) ** 2
    
    if n == 1:
        # von Neumann entropy: -sum(p * log(p))
        SvN = -np.sum(p * np.log(p))
    elif n == 0:
        # Hartley entropy: log(number of non-zero elements)
        SvN = np.log(len(sv_arr))
    else:
        # Renyi entropy: log(sum(p^n)) / (1 - n)
        SvN = np.log(np.sum(p**n)) / (1 - n)
    
    return SvN


# %%
def read_hdf5_file(filename: str, load_sv_arrays: bool = True) -> List[Dict]:
    """
    Read a single HDF5 file and return list of results, based on Julia read_results_hdf5.
    
    Parameters:
    - filename: path to HDF5 file
    - load_sv_arrays: whether to load singular value arrays
    
    Returns:
    - List of result dictionaries
    """
    results = []
    
    with h5py.File(filename, 'r') as file:
        if 'metadata' in file and 'singular_values' in file:
            # New format with separate metadata and singular_values groups
            metadata_group = file['metadata']
            sv_arrays_group = file['singular_values']
            
            # Get all result groups from metadata
            result_groups = [k for k in metadata_group.keys() if k.startswith('result_')]
            
            # Sort by result number
            result_groups.sort(key=lambda x: int(x.split('_')[1]))
            
            for group_name in result_groups:
                result_dict = {}
                
                # Load metadata
                meta_group = metadata_group[group_name]
                for key in meta_group.keys():
                    if isinstance(meta_group[key], h5py.Dataset):
                        result_dict[key] = meta_group[key][()]
                    elif isinstance(meta_group[key], h5py.Group):
                        # Handle nested groups (like args)
                        subdict = {}
                        subgroup = meta_group[key]
                        for subkey in subgroup.keys():
                            subdict[subkey] = subgroup[subkey][()]
                        result_dict[key] = subdict
                
                # Load singular value arrays if requested
                if load_sv_arrays and group_name in sv_arrays_group:
                    result_dict['sv_arr'] = sv_arrays_group[group_name][:]
                
                results.append(result_dict)
        
        else:
            # Legacy format - results directly in file root
            result_groups = [k for k in file.keys() if k.startswith('result_')]
            result_groups.sort(key=lambda x: int(x.split('_')[1]))
            
            for group_name in result_groups:
                result_dict = {}
                group = file[group_name]
                
                for key in group.keys():
                    if isinstance(group[key], h5py.Dataset):
                        # Skip large arrays if requested
                        if not load_sv_arrays and key == 'sv_arr':
                            continue
                        result_dict[key] = group[key][()]
                    elif isinstance(group[key], h5py.Group):
                        # Handle nested groups
                        subdict = {}
                        subgroup = group[key]
                        for subkey in subgroup.keys():
                            subdict[subkey] = subgroup[subkey][()]
                        result_dict[key] = subdict
                
                results.append(result_dict)
    
    return results

def load_hdf5_data(data_path: str, p_fixed_name: str = 'p_ctrl', p_fixed_value: float = 0.4, 
                   n: int = 1, thresholds: Optional[List[float]] = None) -> List[Dict]:
    """
    Load data from all HDF5 files in directory and compute von Neumann entropy.
    
    Parameters:
    - data_path: path to directory containing HDF5 files
    - p_fixed_name: name of the fixed parameter ('p_ctrl' or 'p_proj')
    - p_fixed_value: value of the fixed parameter
    - n: Renyi entropy parameter (0 for Hartley, 1 for von Neumann)
    - thresholds: list of threshold values for entropy computation (if None, uses single default)
    
    Returns:
    - List of dictionaries containing computed entropies and metadata
    """
    if thresholds is None:
        thresholds = [1e-16]  # Default single threshold for backward compatibility
    
    # Find all HDF5 files
    pattern = os.path.join(data_path, f"{p_fixed_name}{p_fixed_value}", "*.h5")
    h5_files = glob.glob(pattern)
    
    print(f"Found {len(h5_files)} HDF5 files")
    if len(thresholds) > 1:
        print(f"Computing entropy (n={n}) for {len(thresholds)} threshold values")
    
    all_data = []
    
    for h5_file in tqdm(h5_files, desc="Processing HDF5 files"):
        
        try:
            # Read all results from this file
            results = read_hdf5_file(h5_file, load_sv_arrays=True)
            
            for result in results:
                # Compute entropy if sv_arr exists
                if 'sv_arr' in result:
                    # Create base data entry
                    data_entry = {
                        'file': os.path.basename(h5_file),
                        'sv_array_length': len(result['sv_arr']),
                        **result  # Include all metadata
                    }
                    
                    # Compute entropy for each threshold
                    for threshold in thresholds:
                        entropy = von_neumann_entropy_sv(result['sv_arr'], n=n, positivedefinite=False, threshold=threshold)
                        if len(thresholds) == 1:
                            # Backward compatibility: single threshold uses 'EE' key
                            data_entry['EE'] = entropy
                        else:
                            # Multiple thresholds: use descriptive keys
                            data_entry[f'EE_n{n}_threshold_{threshold:.0e}'] = entropy
                    
                    all_data.append(data_entry)
        
        except Exception as e:
            print(f"Error reading file {h5_file}: {e}")
            continue
    
    print(f"Successfully processed {len(all_data)} data points")
    return all_data


# %%
def group_hdf5_data(data_list: List[Dict], groupname: str, p_fixed_name: str) -> Tuple[Dict, str]:
    """
    Group HDF5 data by specified parameter, similar to the JSON grouping function.
    
    Parameters:
    - data_list: list of data dictionaries from HDF5 files
    - groupname: parameter to group by (e.g., 'L')
    - p_fixed_name: name of the fixed parameter
    
    Returns:
    - Tuple of (grouped_data, varying_p_name)
    """
    grouped = {}
    varying_p_name = 'p_proj' if p_fixed_name == 'p_ctrl' else 'p_ctrl'
    
    for data in data_list:
        # Extract grouping value (e.g., L value)
        if groupname in data:
            group_value = data[groupname]
        elif 'args' in data and groupname in data['args']:
            group_value = data['args'][groupname]
        else:
            continue
            
        # Extract varying parameter value
        if varying_p_name in data:
            varying_p_value = data[varying_p_name]
        elif 'args' in data and varying_p_name in data['args']:
            varying_p_value = data['args'][varying_p_name]
        else:
            continue
            
        # Group the data
        key = (group_value, varying_p_value)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(data['EE'])
    
    return grouped, varying_p_name


# %%
def calculate_mean_and_error(ee_values: List[float]) -> Tuple[float, float]:
    """Calculate mean and standard error of the mean."""
    ee_array = np.array(ee_values)
    mean = np.mean(ee_array)
    sem = np.std(ee_array, ddof=1) / np.sqrt(len(ee_array))
    return mean, sem

def calculate_variance_and_error(ee_values: List[float]) -> Tuple[float, float]:
    """Calculate variance and standard error of variance."""
    ee_array = np.array(ee_values)
    variance = np.var(ee_array, ddof=1)
    # Standard error of variance approximation
    n = len(ee_array)
    se_var = variance * np.sqrt(2.0 / (n - 1))
    return variance, se_var


# %%
def plot_hdf5_EE_vs_p(data_list: List[Dict], groupname: str, p_fixed_name: str, p_fixed_value: float, 
                     save_plot: bool = True, show_plot: bool = False, save_data: bool = True, 
                     plot_type: str = 'variance') -> Dict:
    """
    Plot characteristic of EE (variance or mean) vs the varying p parameter for HDF5 data.
    
    Parameters:
    - data_list: list of data dictionaries from HDF5 files
    - groupname: the name of the parameter to group by
    - p_fixed_name: which parameter is fixed ('p_proj' or 'p_ctrl')
    - p_fixed_value: the value of the fixed parameter
    - save_plot: whether to save the plot
    - show_plot: whether to display the plot
    - save_data: whether to save the data
    - plot_type: str, either 'variance' or 'mean' to specify what to plot
    
    Returns:
    - Dictionary containing plot data
    """
    
    # Validate plot_type argument
    if plot_type not in ['variance', 'mean']:
        raise ValueError("plot_type must be either 'variance' or 'mean'")
    
    # Select the calculation function based on plot_type
    if plot_type == 'variance':
        calculate_function = calculate_variance_and_error
        y_label = 'Variance of EE'
        data_key = 'variance'
        error_key = 'sem_variance'
    else:  # plot_type == 'mean'
        calculate_function = calculate_mean_and_error
        y_label = 'Mean EE'
        data_key = 'mean_EE'
        error_key = 'sem_EE'
    
    # Group data by the specified parameter and varying p parameter
    ee_grouped, varying_p_name = group_hdf5_data(data_list, groupname, p_fixed_name)
    
    # Organize data for plotting
    plot_data = {}
    for (group_value, varying_p_value), ee_values in ee_grouped.items():
        if group_value not in plot_data:
            plot_data[group_value] = {
                varying_p_name: [],
                data_key: [],
                error_key: []
            }
        
        # Calculate statistic and error
        stat_value, error_value = calculate_function(ee_values)
        
        plot_data[group_value][varying_p_name].append(varying_p_value)
        plot_data[group_value][data_key].append(stat_value)
        plot_data[group_value][error_key].append(error_value)
    
    # Convert lists to numpy arrays and sort by varying parameter
    for group_value in plot_data:
        # Sort by varying parameter value
        sort_indices = np.argsort(plot_data[group_value][varying_p_name])
        for key in plot_data[group_value]:
            plot_data[group_value][key] = np.array(plot_data[group_value][key])[sort_indices]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(plot_data)))
    
    for i, (group_value, color) in enumerate(zip(sorted(plot_data.keys()), colors)):
        x = plot_data[group_value][varying_p_name]
        y = plot_data[group_value][data_key]
        yerr = plot_data[group_value][error_key]
        
        plt.errorbar(x, y, yerr=yerr, 
                    marker='o', linestyle='-', linewidth=2, markersize=6,
                    color=color, label=f'{groupname} = {group_value}')
    
    plt.xlabel(f'{varying_p_name}')
    plt.ylabel(y_label)
    plt.title(f'{y_label} vs {varying_p_name} (fixed {p_fixed_name} = {p_fixed_value}) - HDF5 Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot if requested
    if save_plot:
        os.makedirs('/scratch/ty296/plots', exist_ok=True)
        plot_filename = f'/scratch/ty296/plots/{plot_type}_of_EE_vs_{varying_p_name}_fixed_{p_fixed_name}{p_fixed_value}_hdf5_threshold{threshold:.0e}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_filename}")
    
    # Save data if requested    
    if save_data:
        os.makedirs('/scratch/ty296/plots', exist_ok=True)
        data_filename = f'/scratch/ty296/plots/{plot_type}_of_EE_vs_{varying_p_name}_fixed_{p_fixed_name}{p_fixed_value}_hdf5_threshold{threshold:.0e}.npz'
        np.savez(data_filename, **{f'{groupname}_{gv}_{varying_p_name}': plot_data[gv][varying_p_name] 
                                   for gv in plot_data},
                 **{f'{groupname}_{gv}_{data_key}': plot_data[gv][data_key] 
                    for gv in plot_data},
                 **{f'{groupname}_{gv}_{error_key}': plot_data[gv][error_key] 
                    for gv in plot_data})
        print(f"Data saved to: {data_filename}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    # Print some statistics
    print(f"\\nData summary (varying {varying_p_name}, fixed {p_fixed_name}) - HDF5 Data:")
    for group_value in sorted(plot_data.keys()):
        n_points = len(plot_data[group_value][varying_p_name])
        print(f"{groupname} = {group_value}: {n_points} data points")
        print(f"  {varying_p_name} range: {plot_data[group_value][varying_p_name].min():.3f} to {plot_data[group_value][varying_p_name].max():.3f}")
        print(f"  {plot_type} range: {plot_data[group_value][data_key].min():.3f} ± {plot_data[group_value][error_key][np.argmin(plot_data[group_value][data_key])]:.3f} to {plot_data[group_value][data_key].max():.3f} ± {plot_data[group_value][error_key][np.argmax(plot_data[group_value][data_key])]:.3f}")
    
    return plot_data


# %%
# Test the HDF5 data loading and plotting functions

if __name__ == "__main__":
    hdf5_data_path = '/scratch/ty296/hdf5_data/'
    groupname = 'L'
    p_fixed_name = 'p_ctrl'
    p_fixed_value = 0.4

    # Load HDF5 data
    print("Loading HDF5 data...")
    hdf5_data = load_hdf5_data(hdf5_data_path, p_fixed_name, p_fixed_value)

    if len(hdf5_data) > 0:
        print(f"\\nLoaded {len(hdf5_data)} data points")
        print("Sample data point structure:")
        print(list(hdf5_data[0].keys()))
        
        # Plot mean EE vs varying parameter
        print("\\nGenerating mean EE plot...")
        plot_data_mean = plot_hdf5_EE_vs_p(hdf5_data, groupname, p_fixed_name, p_fixed_value, 
                                        save_plot=True, show_plot=True, plot_type='mean')
        
        # Plot variance of EE vs varying parameter
        print("\\nGenerating variance plot...")
        plot_data_var = plot_hdf5_EE_vs_p(hdf5_data, groupname, p_fixed_name, p_fixed_value, 
                                        save_plot=True, show_plot=True, plot_type='variance')
    else:
        print("No data found. Please check the file paths and structure.")

    # Threshold analysis for Hartley entropy (n=0)
    print("\\n" + "="*60)
    print("THRESHOLD ANALYSIS FOR HARTLEY ENTROPY (n=0)")
    print("="*60)
    
    # Define threshold range from 1e-15 to 1e-5 on log scale
    thresholds = np.logspace(-15, -5, 11)
    print(f"Analyzing {len(thresholds)} thresholds from {thresholds[0]:.0e} to {thresholds[-1]:.0e}")
    
    # Load data with threshold analysis using the updated function
    print("\\nLoading HDF5 data with threshold analysis...")
    hdf5_data_thresholds = load_hdf5_data(hdf5_data_path, p_fixed_name, p_fixed_value, 
                                          n=0, thresholds=thresholds)
    
    if len(hdf5_data_thresholds) > 0:
        print(f"\\nLoaded {len(hdf5_data_thresholds)} data points with threshold analysis")
        print("Sample entropy keys:")
        sample_keys = [k for k in hdf5_data_thresholds[0].keys() if 'EE_n0_threshold' in k]
        print(f"{sample_keys[:3]}...")  # Show first 3
        
        # Generate threshold analysis plots using existing function
        print("\\nGenerating threshold analysis plots...")
        varying_p_name = 'p_proj' if p_fixed_name == 'p_ctrl' else 'p_ctrl'
        
        # Create plots for each threshold by modifying the data temporarily
        for i, threshold in enumerate(tqdm(thresholds, desc="Generating plots for thresholds")):
            threshold_data = []
            ee_key = f'EE_n0_threshold_{threshold:.0e}'
            
            for data_point in hdf5_data_thresholds:
                if ee_key in data_point:
                    # Create a temporary data point with 'EE' key for compatibility
                    temp_point = data_point.copy()
                    temp_point['EE'] = data_point[ee_key]
                    threshold_data.append(temp_point)
            
            if threshold_data:
                # Plot mean
                plot_hdf5_EE_vs_p(threshold_data, groupname, p_fixed_name, p_fixed_value, 
                                 save_plot=True, show_plot=False, plot_type='mean')
                # Plot variance  
                plot_hdf5_EE_vs_p(threshold_data, groupname, p_fixed_name, p_fixed_value, 
                                 save_plot=True, show_plot=False, plot_type='variance')
        
        print("\\nThreshold analysis completed!")
        print(f"Generated plots for mean and variance of Hartley entropy vs {varying_p_name}")
        print("Plots saved to /scratch/ty296/plots/")
    else:
        print("No data found for threshold analysis.")


# %%
# # Load data from json_data folder
# # benchmark_data_path = '/scratch/ty296/precision_benchmark_results/'

# # Add the CT_MPS_mini directory to Python path to import local modules
# import sys
# import os
# sys.path.append('/scratch/ty296/CT_MPS_mini')
# from read_json_func import load_data_flexible, group_by, plot_char_EE_vs_p

# data_path = '/scratch/ty296/json_data/'
# groupname = 'L'
# p_fixed_name = 'p_ctrl'
# p_fixed_value = 0.4
# # json_data_benchmark = load_data_flexible(benchmark_data_path, p_fixed_name, p_fixed_value)
# json_data = load_data_flexible(data_path, p_fixed_name, p_fixed_value)
# # print(f"Loaded {len(json_data_benchmark)} total data points")
# grouped, varying_p_name = group_by(json_data, groupname, p_fixed_name)

# plot_data_proj = plot_char_EE_vs_p(json_data, groupname, p_fixed_name, p_fixed_value, save_plot=True, show_plot=True, plot_type='mean')
# # plot_data_proj = plot_variance_of_EE_vs_p(json_data, groupname='L', p_fixed_name=p_fixed_name, p_fixed_value=p_fixed_value, save_plot=True, show_plot=True)

# # plot_benchmark = plot_average_EE_vs_p(json_data_benchmark, groupname='cutoff', p_fixed_name=p_fixed_name, p_fixed_value=p_fixed_value, save_plot=False, show_plot=True)



