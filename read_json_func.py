# %%
import numpy as np
import json
import glob
import os
import matplotlib.pyplot as plt

def load_data_flexible(data_path, p_fixed_name='p_proj', p_fixed_value=0.5):
    """
    Load data flexibly - handles individual JSON files, JSON arrays, and JSON Lines format
    """
    json_data = []
    json_files = []
    for root, dirs, files in os.walk(os.path.join(data_path, f'{p_fixed_name}{p_fixed_value}')):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    
    print(f"Found {len(json_files)} JSON files")
    
    for file in json_files:
        with open(file, 'r') as f:
            try:
                # Try to load as regular JSON first (single object or array)
                data = json.load(f)
                if isinstance(data, list):
                    # If it's an array of results, extend the main list
                    json_data.extend(data)
                else:
                    # Single result
                    json_data.append(data)
            except json.JSONDecodeError:
                # If that fails, try JSON Lines format (one JSON object per line)
                f.seek(0)  # Reset file pointer
                line_count = 0
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            data = json.loads(line)
                            json_data.append(data)
                            line_count += 1
                        except json.JSONDecodeError as e:
                            print(f"Error parsing line in {file}: {e}")
                # print(f"Loaded {line_count} results from JSON Lines file: {os.path.basename(file)}")
    
    return json_data

# %%
def group_data_by_params(json_data):
    """Group data by all parameter combinations for histogram analysis"""
    grouped_data = {}
    
    for data in json_data:
        # Create a key tuple with the parameters we want to group by
        key = (
            data['args']['L'],
            data['args']['ancilla'],
            data['p_ctrl'],
            data['p_proj']
        )
        
        # If this parameter combination hasn't been seen before, initialize lists
        if key not in grouped_data:
            grouped_data[key] = {
                'EE': [],
                'O': [],
                'max_bond': []
            }
        
        # Append the values to their respective lists
        grouped_data[key]['EE'].append(data['EE'])
        grouped_data[key]['O'].append(data['O'])
        grouped_data[key]['max_bond'].append(data['max_bond'])
    
    return grouped_data

def plot_histograms(json_data, save_plots=True, max_plots=5):
    """Plot histograms for each parameter combination (limited to max_plots)"""
    grouped_results = group_data_by_params(json_data)
    
    print(f"Found {len(grouped_results)} parameter combinations")
    if len(grouped_results) > max_plots:
        print(f"Limiting to first {max_plots} combinations for demonstration")
    
    count = 0
    for key, values in grouped_results.items():
        if count >= max_plots:
            break
            
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot EE histogram
        axes[0].hist(values['EE'], bins=10)
        axes[0].set_title('EE Distribution')
        axes[0].set_xlabel('EE')
        axes[0].set_ylabel('Frequency')

        # Plot O histogram
        axes[1].hist(values['O'], bins=10)
        axes[1].set_title('O Distribution')
        axes[1].set_xlabel('O')
        axes[1].set_ylabel('Frequency')

        # Plot max_bond histogram
        axes[2].hist(values['max_bond'], bins=10)
        axes[2].set_title('Max Bond Distribution')
        axes[2].set_xlabel('Max Bond')
        axes[2].set_ylabel('Frequency')

        fig.tight_layout()
        
        if save_plots:
            filename = f'/scratch/ty296/plots/histogram_{key[0]:03d}_a{key[1]:03d}_p_ctrl{key[2]:.3f}_p_proj{key[3]:.3f}.png'
            fig.savefig(filename)
            
        plt.close()

        # Print the parameters for the plotted data
        L, ancilla, p_ctrl, p_proj = key
        print(f"Histograms for: L = {L}, ancilla = {ancilla}, p_ctrl = {p_ctrl}, p_proj = {p_proj}")
        count += 1

# Optional: Run histogram analysis (commented out by default)
# plot_histograms(json_data, save_plots=True, max_plots=20)




# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def group_by(json_data, groupname, p_fixed_name):
    """
    Group data by anything and the varying p parameter to calculate statistics
    
    Parameters:
    - json_data: list of data dictionaries
    - groupname: the name of the parameter to group by
    - p_fixed_name: the name of the parameter that is fixed
    """
    grouped = {}
    varying_p_name = 'p_ctrl' if p_fixed_name == 'p_proj' else 'p_proj'
    for data in json_data:
        anything_value = data['args'][groupname]
        varying_p_value = data[varying_p_name]
        key = (anything_value, varying_p_value)
        
        if key not in grouped:
            grouped[key] = []
        
        grouped[key].append(data['EE'])
    
    return grouped, varying_p_name

def calculate_mean(ee_values):
    """Calculate mean and standard error of the mean"""
    ee_array = np.array(ee_values)
    mean = np.mean(ee_array)
    std_err = stats.sem(ee_array)  # Standard error of the mean

    ee_array_resampled = np.random.choice(ee_array, size=len(ee_array), replace=True)
    std_err_resampled = stats.sem(ee_array_resampled)

    # Return the larger standard error between original and resampled
    return mean, np.maximum(std_err, std_err_resampled)

def se_variance(data):
    """Calculate standard error of variance for normally distributed data"""
    n = len(data)
    variance = np.var(data, ddof=1)  # sample variance
    se_var = variance * np.sqrt(2 / (n - 1))
    return se_var

def calculate_variance(ee_values):
    """Calculate mean and standard error of the variance"""
    ee_array = np.array(ee_values)
    variance = np.var(ee_array)
    se_var = se_variance(ee_array)
    return variance, se_var

def plot_char_EE_vs_p(json_data, groupname, p_fixed_name, p_fixed_value, save_plot=True, show_plot=False, save_data=True, plot_type='variance'):
    """
    Plot characteristic of EE (variance or mean) vs the varying p parameter
    
    Parameters:
    - json_data: list of data dictionaries
    - groupname: the name of the parameter to group by
    - p_fixed_name: which parameter is fixed ('p_proj' or 'p_ctrl')
    - p_fixed_value: the value of the fixed parameter
    - save_plot: whether to save the plot
    - show_plot: whether to display the plot
    - save_data: whether to save the data
    - plot_type: str, either 'variance' or 'mean' to specify what to plot
    """
    
    # Validate plot_type argument
    if plot_type not in ['variance', 'mean']:
        raise ValueError("plot_type must be either 'variance' or 'mean'")
    
    # Select the calculation function based on plot_type
    if plot_type == 'variance':
        calculate_function = calculate_variance
        y_label = 'Variance'
        data_key = 'variance'
        error_key = 'sem_variance'
    else:  # plot_type == 'mean'
        calculate_function = calculate_mean
        y_label = 'Mean EE'
        data_key = 'mean_EE'
        error_key = 'sem_EE'
    
    # Group data by L and varying p parameter
    ee_grouped, varying_p_name = group_by(json_data, groupname, p_fixed_name)
    
    # Organize data for plotting
    plot_data = {}
    for (group_value, varying_p_value), ee_values in ee_grouped.items():
        # Get L values for this group from original data
        L_values = []
        rescaled_ee_values = []
        ee_values = []
    
        # Find the original data points for this group
        for data in json_data:
            if (data['args'][groupname] == group_value and data[varying_p_name] == varying_p_value):
                L = data['args']['L']
                ee = data['EE']
                rescaled_ee_values.append(ee)
    
        ee_values = rescaled_ee_values
        if group_value not in plot_data:
            plot_data[group_value] = {varying_p_name: [], 'mean_EE': [], 'sem_EE': [], 'variance': [], 'sem_variance': []}
        
        print(f'{groupname} = {group_value}, {varying_p_name} = {varying_p_value}, ensemble size: {len(ee_values)}')
        
        # Use the selected calculation function
        calculated_value, calculated_error = calculate_function(ee_values)
        plot_data[group_value][varying_p_name].append(varying_p_value)
        plot_data[group_value][data_key].append(calculated_value)
        plot_data[group_value][error_key].append(calculated_error)
    
    # Sort data by varying p parameter for each L
    for group_value in plot_data:
        sorted_indices = np.argsort(plot_data[group_value][varying_p_name])
        plot_data[group_value][varying_p_name] = np.array(plot_data[group_value][varying_p_name])[sorted_indices]
        plot_data[group_value][data_key] = np.array(plot_data[group_value][data_key])[sorted_indices]
        plot_data[group_value][error_key] = np.array(plot_data[group_value][error_key])[sorted_indices]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    markers = ['o', 's', '^', 'D', 'v', '<']
    
    for i, group_value in enumerate(sorted(plot_data.keys())):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        print(len(plot_data[group_value][varying_p_name]))
        if save_data:
            np.savez(f'/scratch/ty296/CT_MPS_mini/entropy_vs_p_values_{groupname}{group_value}_{plot_type}.npz', 
                     p_values=plot_data[group_value][varying_p_name], 
                     values=plot_data[group_value][data_key], 
                     errors=plot_data[group_value][error_key])
            print(f'Saved data for {groupname} = {group_value}')
        
        ax.errorbar(plot_data[group_value][varying_p_name], 
                    plot_data[group_value][data_key], 
                    yerr=plot_data[group_value][error_key],
                    label=f'{groupname} = {group_value}',
                    color=color,
                    marker=marker,
                    markersize=6,
                    linewidth=2,
                    capsize=3,
                    capthick=1)
    
    # Set labels and title based on which parameter is varying
    ax.set_xlabel(varying_p_name, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    
    title = f'{y_label} of Entanglement Entropy vs {varying_p_name} (fixed {p_fixed_name} = {p_fixed_value})'
    filename = f'/scratch/ty296/plots/{plot_type}_of_EE_vs_{varying_p_name}_fixed_{p_fixed_name}{p_fixed_value}.png' 
    
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save or show the plot
    fig.tight_layout()
    if save_plot:
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {filename}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    # Print some statistics
    print(f"\nData summary (varying {varying_p_name}, fixed {p_fixed_name}):")
    for group_value in sorted(plot_data.keys()):
        n_points = len(plot_data[group_value][varying_p_name])
        print(f"{groupname} = {group_value}: {n_points} data points")
        print(f"  {varying_p_name} range: {plot_data[group_value][varying_p_name].min():.3f} to {plot_data[group_value][varying_p_name].max():.3f}")
        print(f"  {plot_type} range: {plot_data[group_value][data_key].min():.3f} ± {plot_data[group_value][error_key][np.argmin(plot_data[group_value][data_key])]:.3f} to {plot_data[group_value][data_key].max():.3f} ± {plot_data[group_value][error_key][np.argmax(plot_data[group_value][data_key])]:.3f}")
    
    return plot_data


# # %%
# # Load data from json_data folder
# # benchmark_data_path = '/scratch/ty296/precision_benchmark_results/'
# data_path = '/scratch/ty296/json_data/'
# groupname = 'L'
# p_fixed_name = 'p_ctrl'
# p_fixed_value = 0.4
# # json_data_benchmark = load_data_flexible(benchmark_data_path, p_fixed_name, p_fixed_value)
# json_data = load_data_flexible(data_path, p_fixed_name, p_fixed_value)
# # print(f"Loaded {len(json_data_benchmark)} total data points")
# grouped, varying_p_name = group_by(json_data, groupname, p_fixed_name)

# plot_data_proj = plot_char_EE_vs_p(json_data, groupname, p_fixed_name, p_fixed_value, save_plot=True, show_plot=True, plot_type='mean')
# plot_data_proj = plot_variance_of_EE_vs_p(json_data, groupname='L', p_fixed_name=p_fixed_name, p_fixed_value=p_fixed_value, save_plot=True, show_plot=True)

# plot_benchmark = plot_average_EE_vs_p(json_data_benchmark, groupname='cutoff', p_fixed_name=p_fixed_name, p_fixed_value=p_fixed_value, save_plot=False, show_plot=True)


# # %%
# epsilon = -1e-14
# for data in json_data_benchmark:
#     if data['EE'] <= epsilon:
#         print(data['EE'])

# %%
# import glob
# import os

# data_path = '/scratch/ty296/json_data/'
# p_fixed_name = 'p_proj'
# p_fixed_value = 0.5
# json_data_benchmark_files = glob.glob(os.path.join(data_path, f'{p_fixed_name}{p_fixed_value}/*.json'))
# print(json_data_benchmark_files[0], len(json_data_benchmark_files))
# count = 0
# for data, json_file in zip(json_data_benchmark, json_data_benchmark_files):
#     print(data['args']['cutoff'])
#     if data['args']['cutoff'] == 1e-7:
#         count += 1
#         os.rename(json_file, os.path.join(data_path, f'{p_fixed_name}{p_fixed_value}/cutoff1e-8/{os.path.basename(json_file)}'))
#     else:
#         print(f'{json_file} is not 1e-8')
# 
# print(count)

# %%
# # Get all data files recursively from json_data directory
# data_path = '/scratch/ty296/precision_benchmark_results/'
# all_json_files = []

# # Walk through all subdirectories
# for root, dirs, files in os.walk(data_path):
#     for file in files:
#         if file.endswith('.json'):
#             # Get full path to json file
#             json_path = os.path.join(root, file)
#             all_json_files.append(json_path)

# print(f"Found {len(all_json_files)} total JSON files")
# print("\nExample files:")
# # [32331:32332]
# counter = 0
# for file in all_json_files:  # Print first 5 files as examples
#     with open(file, 'r') as f:
#         # Look for cutoff parameter in each line of the file
#         data = f.readlines()
#         try:
#             first_line = data[0]
#             json_data = json.loads(first_line)
#             if json_data['args']['p_fixed_name'] == 'p_ctrl' and json_data['args']['p_fixed_value'] == 0.0 and json_data['args']['L'] == 16:
#                 print(json_data['args']['cutoff'])
#                 counter += 1
#                 target_dir = f'/scratch/ty296/json_data/{json_data["args"]["p_fixed_name"]}{json_data["args"]["p_fixed_value"]}/cutoff{json_data["args"]["cutoff"]}'
#                 os.makedirs(target_dir, exist_ok=True)
#                 target_path = os.path.join(target_dir, os.path.basename(file))
#                 os.rename(file, target_path)
#             # if 'args' not in json_data or 'cutoff' not in json_data['args']:
#             #     continue
#             # if json_data['args']['cutoff'] == 1e-6:
#             #     target_dir = f'/scratch/ty296/precision_benchmark_results/{json_data["args"]["p_fixed_name"]}{json_data["args"]["p_fixed_value"]}/cutoff{json_data["args"]["cutoff"]}'
#             #     os.makedirs(target_dir, exist_ok=True)
                
#             # # Move the file
#             # target_path = os.path.join(target_dir, os.path.basename(file))
#             # os.rename(file, target_path)
#             # counter += 1
#         except:
#             continue
# print(counter)


