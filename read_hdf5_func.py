# %%
import sys
sys.path.append('/scratch/ty296/CT_MPS_mini')
# %%
import h5py
import glob
import os
import tqdm
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def parse_name(name):
    L = int(name.split('_')[2].split('L')[1])
    p_ctrl = float(name.split('_')[3].split('pctrl')[1].split('_')[0])
    p_proj = float(name.split('_')[4].split('pproj')[1].split('_')[0])
    seed = int(name.split('_')[5].split('seed')[1])
    return L, p_ctrl, p_proj, seed

def plot_sv_arr(filename):
    # the input filename must be full file path to a single trajectory h5 file
    with h5py.File(filename, 'r') as f:
        sv_arr = f['singular_values'][:]
        # print(sv_arr)
        plt.yscale('log')
        plt.plot(sv_arr)
        # plt.show()
        plt.savefig(f'/scratch/ty296/plots/sv_arr_{filename}.png')
        # print(sv_arr.shape)

def combine(combined_sv_filename, p_fixed_name, p_fixed_value, eps_value=None, hdf5_data_path="/scratch/ty296/hdf5_data"):
    if eps_value is not None:
        all_files = glob.glob(os.path.join(f"{hdf5_data_path}/{p_fixed_name}{p_fixed_value}", f"*eps{eps_value}*", "**"), recursive=True)
    else:
        all_files = glob.glob(os.path.join(f"{hdf5_data_path}/{p_fixed_name}{p_fixed_value}", 
                                        "*", "**"), 
                            recursive=True)
    
    # Filter out directories, keep only files
    all_files = [f for f in all_files if os.path.isfile(f)]
    with h5py.File(combined_sv_filename, 'w') as f_target:
        duplicate_count = 0
        duplicate_list = []
        for filename in tqdm.tqdm(all_files):
            try:
                with h5py.File(filename, 'r+') as f_source:
                    dataset = f_source['singular_values']
                    attrs = dataset.attrs
                    # Create dataset name
                    dataset_name = f'sv_arr_L{int(attrs["L"])}_pctrl{float(attrs["p_ctrl"])}_pproj{float(attrs["p_proj"])}_seed{int(attrs["seed"])}'
                    
                    # Check if name already exists, skip if duplicate
                    if dataset_name in f_target:
                        duplicate_count += 1
                        duplicate_list.append(filename)
                        continue
                    
                    dataset_target = f_target.create_dataset(dataset_name, data=np.transpose(dataset[:]))
                    dataset_target.attrs.update(attrs)
            except OSError as e:
                print(f"CORRUPTED FILE (removing): {filename}")
                print(f"  Error: {e}")
                os.remove(filename)
                continue
    print(f"Duplicate count: {duplicate_count}")
    print(f"Duplicate list: {duplicate_list}")

def total_s0_dict(combined_sv_filename, threshold_val):
    """
    Groups entropy values by (L, p_ctrl, p_proj) from combined HDF5 file.
    
    Returns:
        total_dict: {(L, p_ctrl, p_proj): array of entropy values}
    """
    total_dict = {}
    with h5py.File(combined_sv_filename, 'r') as f:
        for key in list(f.keys()):
            L, p_ctrl, p_proj, seed = parse_name(key)
            s = []
            for sv_arr in f[key][()].T:
                # print(sv_arr.shape)
                s0 = von_neumann_entropy_sv(sv_arr, n=0, positivedefinite=False, threshold=threshold_val)
                s.append(s0)
            s = np.array(s)
            # print(np.shape(s))
        
            # Check if s contains any infinity values
            if np.any(np.isinf(s)):
                print(f"INFINITY VALUE (skipping): seed={seed}")
                print(f"  Parameters: L={L}, p_ctrl={p_ctrl}, p_proj={p_proj}")
                continue

            s = np.array(s).flatten()

            if (L, p_ctrl, p_proj) not in total_dict:
                total_dict[(L, p_ctrl, p_proj)] = s
            else:
                total_dict[(L, p_ctrl, p_proj)] = np.concatenate((total_dict[(L, p_ctrl, p_proj)], s))
    return total_dict

# read into the combined data file
def distribution_dict(combined_sv_filename, L_target, threshold_val, temporal_average=True):
    min_sv_dict = {}
    maxbond_dict = {}
    entropy_dict = {}
    with h5py.File(combined_sv_filename, 'r') as f:
        for key in list(f.keys()):
            L, p_ctrl, p_proj, seed = parse_name(key)
            if L == L_target:

                if 'max_bond' in f[key].attrs.keys():
                    maxbond = f[key].attrs['max_bond']
                    if (L, p_ctrl, p_proj) not in maxbond_dict:
                        maxbond_dict[(L, p_ctrl, p_proj)] = [maxbond]
                    else:
                        maxbond_dict[(L, p_ctrl, p_proj)].append(maxbond)

                # Calculate entropy first to check for infinity before adding anything
                if temporal_average:
                    entropy = [von_neumann_entropy_sv(sv_arr, n=0, positivedefinite=False, threshold=threshold_val) for sv_arr in f[key][()]]
                    entropy = np.mean(entropy)
                else:
                    entropy = von_neumann_entropy_sv(f[key][()], n=0, positivedefinite=False, threshold=threshold_val)
                    
                if np.any(np.isinf(entropy)):
                    print(f"INFINITY VALUE (skipping): seed={seed}")
                    print(f"  Parameters: L={L}, p_ctrl={p_ctrl}, p_proj={p_proj}")
                    continue
                
                if f[key][()].ndim == 1:
                    min_sv = np.min(f[key][()])
                else:
                    min_sv = []
                    for sv_arr in f[key][()]:
                        min_sv.append(np.min(sv_arr))
                    min_sv = np.array(min_sv)
                
                # Check if min_sv is too small (< 1e-10) or contains inf
                if np.any(np.isinf(min_sv)) or np.any(min_sv == 0.0):
                    print(f"INVALID MIN_SV (skipping): seed={seed}")
                    print(f"  Parameters: L={L}, p_ctrl={p_ctrl}, p_proj={p_proj}")
                    print(f"  min_sv value: {min_sv if np.isscalar(min_sv) else np.min(min_sv)}")
                    continue
                
                if (L, p_ctrl, p_proj) not in min_sv_dict:
                    min_sv_dict[(L, p_ctrl, p_proj)] = [min_sv]
                else:
                    min_sv_dict[(L, p_ctrl, p_proj)].append(min_sv)
            
                if (L, p_ctrl, p_proj) not in entropy_dict:
                    entropy_dict[(L, p_ctrl, p_proj)] = [entropy]
                else:
                    entropy_dict[(L, p_ctrl, p_proj)].append(entropy)
    return min_sv_dict, maxbond_dict, entropy_dict

def plot_distribution(data_dict, n_plots=None, x_label='s0', log_scale=False, n_bins=20):
    if n_plots is None:
        n_plots = len(data_dict)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()

    # Plot histogram for each p_proj value

    # for idx, (p_proj, sv_seed_list) in enumerate(sorted(min_sv_dict.items())[0:n_plots]):
    for idx, (key, data_array) in enumerate(sorted(data_dict.items())[0:n_plots]):
        print(key, len(data_array))
        L, p_ctrl, p_proj = key
        ax = axes[idx]
        if log_scale:
            ax.hist(np.log10(data_array), edgecolor='black', bins=n_bins)
        else:
            ax.hist(data_array, edgecolor='black', bins=n_bins)
        ax.set_title(f'p_proj = {p_proj:.2f}')
        ax.set_xlabel(x_label) # chosen from min_sv, maxbond or entropy
        ax.set_ylabel('Count')
        ax.legend()

    # Remove any empty subplots
    for idx in range(len(data_dict), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig(f'/scratch/ty296/plots/distribution_{x_label}.png')
    # plt.show()

import matplotlib.pyplot as plt

def plot_from_dict(total_dict, threshold_val, p_fixed_name='p_ctrl', p_fixed_value=0.4, save_folder='/scratch/ty296/plots'):
    """
    Plot p_proj vs mean±SEM and p_proj vs variance±SEVAR from dictionary
    data_dict format: {(L, p_ctrl, p_proj): (mean, sem, var, semvar)}
    """
    import os
    import numpy as np
    
    # Create save folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)

    plot_dict = {}
    for key in total_dict.keys():
        if len(total_dict[key]) > 1:
            mean, sem = calculate_mean_and_error(total_dict[key])
            var, semvar = calculate_variance_and_error(total_dict[key])
            plot_dict[key] = (mean, sem, var, semvar)
        else:
            print(key, total_dict[key])

    
    # Filter data for the fixed parameter and organize by L values
    plot_data = {}
    for (L, p_ctrl, p_proj), (mean, sem, var, semvar) in plot_dict.items():
        # Filter based on fixed parameter
        if p_fixed_name == 'p_ctrl' and p_ctrl == p_fixed_value:
            if L not in plot_data:
                plot_data[L] = {'p_proj': [], 'mean': [], 'sem': [], 'variance': [], 'se_var': []}
            
            plot_data[L]['p_proj'].append(p_proj)
            plot_data[L]['mean'].append(mean)
            plot_data[L]['sem'].append(sem)
            plot_data[L]['variance'].append(var)
            plot_data[L]['se_var'].append(semvar)
        elif p_fixed_name == 'p_proj' and p_proj == p_fixed_value:
            if L not in plot_data:
                plot_data[L] = {'p_ctrl': [], 'mean': [], 'sem': [], 'variance': [], 'se_var': []}
            
            plot_data[L]['p_ctrl'].append(p_ctrl)
            plot_data[L]['mean'].append(mean)
            plot_data[L]['sem'].append(sem)
            plot_data[L]['variance'].append(var)
            plot_data[L]['se_var'].append(semvar)

    if not plot_data:
        print(f"No data found for {p_fixed_name}={p_fixed_value}")
        return

    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Get sorted L values and create color map
    L_values = sorted(plot_data.keys())
    n_L = len(L_values)
    
    # Create increasingly deeper shades of blue proportional to L value
    colors = []
    if n_L > 1:
        min_L = min(L_values)
        max_L = max(L_values)
        
        for L in L_values:
            # Normalize L to range [0, 1]
            norm_L = (L - min_L) / (max_L - min_L)
            
            # Create light blue to dark blue gradient
            red = 0.7 * (1 - norm_L)      # From 0.7 to 0.0
            green = 0.7 * (1 - norm_L)    # From 0.7 to 0.0  
            blue = 1.0 - 0.2 * norm_L     # From 1.0 to 0.8
            
            blue_color = (red, green, blue)
            colors.append(blue_color)
    else:
        colors = [(0.0, 0.0, 0.8)]  # Single dark blue color

    # Determine x-axis variable and label
    x_var = 'p_proj' if p_fixed_name == 'p_ctrl' else 'p_ctrl'
    x_label = 'p_proj' if p_fixed_name == 'p_ctrl' else 'p_ctrl'

    # Plot 1: x_var vs mean ± sem
    for i, L in enumerate(L_values):
        data = plot_data[L]
        # Sort by x variable for cleaner lines
        sorted_indices = np.argsort(data[x_var])
        x_sorted = np.array(data[x_var])[sorted_indices]
        mean_sorted = np.array(data['mean'])[sorted_indices]
        sem_sorted = np.array(data['sem'])[sorted_indices]
        
        ax1.errorbar(x_sorted, mean_sorted, yerr=sem_sorted, 
                    label=f'L={L}', marker='o', capsize=5, capthick=2, color=colors[i])

    ax1.set_xlabel(x_label)
    ax1.set_ylabel('Mean Entropy ± SEM')
    ax1.set_title(f'Mean Entropy vs {x_label} for Different L ({p_fixed_name}={p_fixed_value})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: x_var vs variance ± se_var
    for i, L in enumerate(L_values):
        data = plot_data[L]
        # Sort by x variable for cleaner lines
        sorted_indices = np.argsort(data[x_var])
        x_sorted = np.array(data[x_var])[sorted_indices]
        variance_sorted = np.array(data['variance'])[sorted_indices]
        se_var_sorted = np.array(data['se_var'])[sorted_indices]
        
        ax2.errorbar(x_sorted, variance_sorted, yerr=se_var_sorted, 
                    label=f'L={L}', marker='s', capsize=5, capthick=2, color=colors[i])

    ax2.set_xlabel(x_label)
    ax2.set_ylabel('Variance ± SEVar')
    ax2.set_title(f'Variance vs {x_label} for Different L ({p_fixed_name}={p_fixed_value})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join(save_folder, f's0_threshold{threshold_val}_{p_fixed_name}{p_fixed_value}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Plot saved to {save_path}')
    plt.show()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

import pandas as pd

def h5_to_csv(sv_combined, n, threshold: float, p_fixed_name: str, p_fixed_value: float, save_folder="/scratch/ty296/plots"):
    groups = total_s0_dict(sv_combined, threshold)
    # with h5py.File(sv_combined, 'r') as f:
    #     from collections import defaultdict
    #     groups = defaultdict(list)
    #     for real_key in tqdm.tqdm(f.keys()):
    #         s0 = von_neumann_entropy_sv(f[real_key][()], n=n, positivedefinite=False, threshold=threshold)
    #         # print(f[real_key].attrs['p_proj'],f[real_key].attrs['p_ctrl'],f[real_key].attrs['L'],f[real_key].attrs['maxbond'],s0)
    #         key_val = (f[real_key].attrs['L'],f[real_key].attrs['p_ctrl'],f[real_key].attrs['p_proj'])
    #         groups[key_val].append(s0)
    
    data = []
    for key_val, s0_list in groups.items():
        ensemble_size = len(s0_list)
        # print(f'key_val {key_val} ensemble_size {ensemble_size}')
        mean, sem = calculate_mean_and_error(s0_list)
        if len(s0_list) > 1:
            variance, se_var = calculate_variance_and_error(s0_list)
        else:
            variance, se_var = 0, 0
        # print(key_val, "mean", mean, "sem", sem, "variance", variance, "se_var", se_var)
        data.append(list(key_val) + [mean, sem, variance, se_var, ensemble_size])
    # print(data)

    df = pd.DataFrame(data, columns=['L', 'p_ctrl', 'p_proj', 'mean', 'sem', 'variance', 'se_var', 'ensemble_size'])
    # save the data to a csv file
    csv_path = os.path.join(save_folder, f's{n}_threshold{threshold:.1e}_{p_fixed_name}{p_fixed_value}.csv')
    df.to_csv(csv_path, index=False)
    print(f'threadhold {threshold} saved to {csv_path}')

    return df

def von_neumann_entropy_sv(sv_arr: np.ndarray, n: int, positivedefinite: bool, threshold: float) -> float:
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
    mask = sv_arr > threshold
    sv_arr = sv_arr[mask]
    if positivedefinite:
        p = np.maximum(sv_arr, threshold)
        p = sv_arr
    else:
        p = np.maximum(sv_arr, threshold) ** 2
        p = sv_arr ** 2

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

def calculate_mean_and_error(sv_values: List[float]) -> Tuple[float, float]:
    """Calculate mean and standard error of the mean."""
    sv_array = np.array(sv_values)
    mean = np.mean(sv_array)
    sem = np.std(sv_array, ddof=1) / np.sqrt(len(sv_array))
    return mean, sem

def calculate_variance_and_error(sv_values: List[float]) -> Tuple[float, float]:
    """Calculate variance and standard error of variance."""
    sv_array = np.array(sv_values)
    mean = np.mean(sv_array)
    var = np.var(sv_array, ddof=1)

    deviations = sv_array - mean
    fourth_moment = np.mean(deviations**4)
    se_var = (1/len(sv_array)) * (fourth_moment - (len(sv_array)-3)/(len(sv_array)-1) * var**2)
    return var, se_var


class Postprocessing:
    def __init__(self, p_fixed_name: str, p_fixed_value: float, n: int, pwd: str):
        self.p_fixed_name = p_fixed_name
        self.p_fixed_value = p_fixed_value
        self.n = n
        self.pwd = pwd
        print("p_fixed_name", self.p_fixed_name, "p_fixed_value", self.p_fixed_value, "n", self.n)
        self.sv_combined = os.path.join(self.pwd, "hdf5_data_combined/sv_combined_{}{}.h5".format(self.p_fixed_name, self.p_fixed_value))
        print("sv_combined", self.sv_combined)
        self.dir_name = os.path.join(self.pwd, "hdf5_data/{}{}".format(self.p_fixed_name, self.p_fixed_value))
        print("dir_name", self.dir_name)
        self.save_folder = os.path.join(self.pwd, "plots")
        print("save_folder", self.save_folder)
        self.counter = 0

    def postprocessing(self):
        h5_files = glob.glob(os.path.join(self.dir_name, '*.h5'))
        with h5py.File(self.sv_combined, 'w') as f_combined:
            for file in tqdm.tqdm(h5_files):
                with h5py.File(file, 'r') as f:
                    metadata = f['metadata']
                    singular_values = f['singular_values']
                    for result_group in metadata.keys():
                        self.counter += 1
                        p_proj = metadata[result_group]['p_proj'][()]
                        p_ctrl = metadata[result_group]['p_ctrl'][()]
                        L = metadata[result_group]['args']['L'][()]
                        seed = metadata[result_group]['seed'][()]
                        # Print all attributes in args group
                        args_group = metadata[result_group]['args']
                        maxdim = args_group['maxdim'][()]
                        n_chunk_realizations = args_group['n_chunk_realizations'][()]
                        
                        # # View all keys/attributes in args_group
                        # print(f"Args group keys: {list(args_group.keys())}")

                        # # Access specific values (example with L)
                        # L = args_group['L'][()]
                        # print(f"L value: {L}")
                        
                        # # You can also iterate through all items
                        # for key in args_group.keys():
                        #     value = args_group[key][()]
                        #     print(f"  {key}: {value}")
                        
                        maxbond = metadata[result_group]['max_bond'][()]
                        sv_arr = singular_values[result_group][()]
                        group_name = f'real{self.counter}'
                        grp = f_combined.create_dataset(group_name, data=sv_arr)
                        grp.attrs['p_proj'] = p_proj
                        grp.attrs['p_ctrl'] = p_ctrl
                        grp.attrs['L'] = L
                        grp.attrs['maxbond'] = maxbond
                        grp.attrs['maxdim'] = maxdim
                        grp.attrs['n_chunk_realizations'] = n_chunk_realizations
                        grp.attrs['seed'] = seed
                        
    def h5_to_csv(self, threshold: float):
        with h5py.File(self.sv_combined, 'r') as f:
            from collections import defaultdict
            groups = defaultdict(list)
            for real_key in tqdm.tqdm(f.keys()):
                s0 = von_neumann_entropy_sv(f[real_key][()], n=self.n, positivedefinite=False, threshold=threshold)
                # print(f[real_key].attrs['p_proj'],f[real_key].attrs['p_ctrl'],f[real_key].attrs['L'],f[real_key].attrs['maxbond'],s0)
                key_val = (f[real_key].attrs['L'],f[real_key].attrs['p_ctrl'],f[real_key].attrs['p_proj'])
                groups[key_val].append(s0)
            
            data = []
            for key_val, s0_list in groups.items():
                ensemble_size = len(s0_list)
                # print(f'key_val {key_val} ensemble_size {ensemble_size}')
                mean, sem = calculate_mean_and_error(s0_list)
                if len(s0_list) > 1:
                    variance, se_var = calculate_variance_and_error(s0_list)
                else:
                    variance, se_var = 0, 0
                # print(key_val, "mean", mean, "sem", sem, "variance", variance, "se_var", se_var)
                data.append(list(key_val) + [mean, sem, variance, se_var])
            # print(data)

            df = pd.DataFrame(data, columns=['L', 'p_ctrl', 'p_proj', 'mean', 'sem', 'variance', 'se_var'])
            # save the data to a csv file
            csv_path = os.path.join(self.save_folder, f's{self.n}_threshold{threshold:.1e}_{self.p_fixed_name}{self.p_fixed_value}.csv')
            df.to_csv(csv_path, index=False)
            print(f'threadhold {threshold} saved to {csv_path}')

            return df

    def plot_from_csv(self, threshold: float):
        """
        Plot p_proj vs mean±SEM and p_proj vs variance±SEVAR from CSV file
        CSV should have columns: L, p_ctrl, p_proj, mean, sem, variance, se_var
        """
        
        # Read CSV data
        csv_path = os.path.join(self.save_folder, f's{self.n}_threshold{threshold:.1e}_{self.p_fixed_name}{self.p_fixed_value}.csv')
        df = pd.read_csv(csv_path)
        
        # Organize data by L values
        plot_data = {}
        for _, row in df.iterrows():
            L = row['L']
            if L not in plot_data:
                plot_data[L] = {'p_proj': [], 'mean': [], 'sem': [], 'variance': [], 'se_var': []}
            
            plot_data[L]['p_proj'].append(row['p_proj'])
            plot_data[L]['mean'].append(row['mean'])
            plot_data[L]['sem'].append(row['sem'])
            plot_data[L]['variance'].append(row['variance'])
            plot_data[L]['se_var'].append(row['se_var'])

        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Get sorted L values and create color map
        L_values = sorted(plot_data.keys())[0:4]
        n_L = len(L_values)
        
        # Create increasingly deeper shades of blue proportional to L value
        # Smaller L = very light blue, larger L = darker blue
        colors = []
        min_L = min(L_values)
        max_L = max(L_values)
        
        for L in L_values:
            # Normalize L to range [0, 1]
            norm_L = (L - min_L) / (max_L - min_L)
            
            # Create light blue to dark blue gradient
            # Light blue: (0.7, 0.7, 1.0), Dark blue: (0.0, 0.0, 0.8)
            red = 0.7 * (1 - norm_L)      # From 0.7 to 0.0
            green = 0.7 * (1 - norm_L)    # From 0.7 to 0.0  
            blue = 1.0 - 0.2 * norm_L     # From 1.0 to 0.8
            
            blue_color = (red, green, blue)
            
            colors.append(blue_color)

        # Plot 1: p_proj vs mean ± sem
        for i, L in enumerate(L_values):
            data = plot_data[L]
            # Sort by p_proj for cleaner lines
            sorted_indices = np.argsort(data['p_proj'])
            p_proj_sorted = np.array(data['p_proj'])[sorted_indices]
            mean_sorted = np.array(data['mean'])[sorted_indices]
            sem_sorted = np.array(data['sem'])[sorted_indices]
            
            ax1.errorbar(p_proj_sorted, mean_sorted, yerr=sem_sorted, 
                        label=f'L={L}', marker='o', capsize=5, capthick=2, color=colors[i])

        ax1.set_xlabel('p_proj')
        ax1.set_ylabel('Mean Entropy ± SEM')
        ax1.set_title('Mean Entropy vs p_proj for Different L')
        ax1.legend()
        ax1.set_xlim(0.2, 1.0)
        ax1.grid(True, alpha=0.3)

        # Plot 2: p_proj vs variance ± se_var
        for i, L in enumerate(L_values):
            data = plot_data[L]
            # Sort by p_proj for cleaner lines
            sorted_indices = np.argsort(data['p_proj'])
            p_proj_sorted = np.array(data['p_proj'])[sorted_indices]
            variance_sorted = np.array(data['variance'])[sorted_indices]
            se_var_sorted = np.array(data['se_var'])[sorted_indices]
            
            ax2.errorbar(p_proj_sorted, variance_sorted, yerr=se_var_sorted, 
                        label=f'L={L}', marker='s', capsize=5, capthick=2, color=colors[i])

        ax2.set_xlabel('p_proj')
        ax2.set_ylabel('Variance ± SEVar')
        ax2.set_title('Variance vs p_proj for Different L')
        ax2.legend()
        ax2.set_xlim(0.2, 1.0)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.save_folder}/s{self.n}_threshold{threshold:.1e}_{self.p_fixed_name}{self.p_fixed_value}.png')
        plt.close()
        print(f'threshold {threshold} saved to {self.save_folder}/s{self.n}_threshold{threshold:.1e}_{self.p_fixed_name}{self.p_fixed_value}.png')
        # plt.show()

    def fixed_L_threshold_comparison_plot(self, L: int):
        """
        Plot the comparison of the entropy for different thresholds
        """
        
        # Read from csv files
        csv_paths = glob.glob(os.path.join(self.save_folder, f's{self.n}_threshold*.csv'))
        
        # Organize data by threshold values
        plot_data = {}
        import re
        
        for csv_path in csv_paths:
            df = pd.read_csv(csv_path)
            # Find data specific to L
            df_L = df[df['L'] == L]
            
            if len(df_L) == 0:
                continue
                
            # Extract threshold from csv_path filename
            threshold_match = re.search(r'threshold([\d\.e\-\+]+)', csv_path)
            if threshold_match:
                threshold_str = threshold_match.group(1)
                threshold_val = float(threshold_str)
            else:
                continue
                
            # Sort by p_proj and get corresponding values
            sorted_indices = np.argsort(df_L['p_proj'])
            plot_data[threshold_val] = {
                'p_proj': df_L['p_proj'].iloc[sorted_indices].values,
                'mean': df_L['mean'].iloc[sorted_indices].values,
                'sem': df_L['sem'].iloc[sorted_indices].values,
                'variance': df_L['variance'].iloc[sorted_indices].values,
                'se_var': df_L['se_var'].iloc[sorted_indices].values
            }

        # Create plots with same formatting as plot_from_csv
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Get sorted threshold values and create color map
        threshold_values = sorted(plot_data.keys())
        n_thresholds = len(threshold_values)
        
        # Create increasingly deeper shades of blue proportional to threshold value
        colors = []
        if n_thresholds > 1:
            min_thresh = min(threshold_values)
            max_thresh = max(threshold_values)
            
            for thresh in threshold_values:
                # Normalize threshold to range [0, 1]
                norm_thresh = (thresh - min_thresh) / (max_thresh - min_thresh)
                
                # Create dark blue to light blue gradient
                # Lowest threshold (norm_thresh=0) -> dark blue (0.0, 0.0, 0.8)
                # Highest threshold (norm_thresh=1) -> light blue (0.7, 0.7, 1.0)
                red = 0.7 * norm_thresh            # From 0.0 to 0.7
                green = 0.7 * norm_thresh          # From 0.0 to 0.7  
                blue = 0.8 + 0.2 * norm_thresh     # From 0.8 to 1.0
                
                blue_color = (red, green, blue)
                colors.append(blue_color)
        else:
            colors = [(0.0, 0.0, 0.8)]  # Single dark blue color

        # Plot 1: p_proj vs mean ± sem
        for i, threshold in enumerate(threshold_values):
            data = plot_data[threshold]
            min_ = abs(np.log(sorted(threshold_values, reverse=True)[0]))
            max_ = abs(np.log(min(threshold_values)))
            alpha = (abs(np.log(threshold))-min_)/(max_-min_)
            ax1.errorbar(data['p_proj'], data['mean'], yerr=data['sem'], 
                        label=f'threshold={threshold:.1e}', marker='o', capsize=5, capthick=2, color='blue', alpha=alpha)

        ax1.set_xlabel('p_proj')
        ax1.set_ylabel('Mean Entropy ± SEM')
        ax1.set_title(f'Mean Entropy vs p_proj for L={L}')
        ax1.legend()
        ax1.set_xlim(0.2, 1.0)
        ax1.grid(True, alpha=0.3)

        # Plot 2: p_proj vs variance ± se_var
        for i, threshold in enumerate(threshold_values):
            data = plot_data[threshold]
            min_ = abs(np.log(sorted(threshold_values, reverse=True)[0]))
            max_ = abs(np.log(min(threshold_values)))
            alpha = (abs(np.log(threshold))-min_)/(max_-min_)
            ax2.errorbar(data['p_proj'], data['variance'], yerr=data['se_var'], 
                        label=f'threshold={threshold:.1e}', marker='s', capsize=5, capthick=2, color='blue', alpha=alpha)

        ax2.set_xlabel('p_proj')
        ax2.set_ylabel('Variance ± SEVar')
        ax2.set_title(f'Variance vs p_proj for L={L}')
        ax2.legend()
        ax2.set_xlim(0.2, 1.0)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.save_folder}/s{self.n}_threshold_comparison_{L}.png')
        plt.close()
        print(f'Threshold comparison for L={L} saved to {self.save_folder}/s{self.n}_threshold_comparison_{L}.png')
        #     # find all 
        #     csv_path = os.path.join(self.save_folder, f's{self.n}_threshold{threshold:.1e}_{self.p_fixed_name}{self.p_fixed_value}.csv')
        #     df = pd.read_csv(csv_path)
        #     # find in each row the L value is equal to L
        #     df_L = df[df['L'] == L]
        #     p_proj = df_L['p_proj']
        #     mean = df_L['mean']
        #     sem = df_L['sem']


# %%
# Example usage:
# First generate CSV using h5_to_csv function:

if __name__ == "__main__":

    p_fixed_name = 'p_ctrl'
    p_fixed_value = 0.0
    n = 0
    postprocessing = Postprocessing(p_fixed_name, p_fixed_value, n, pwd="/scratch/ty296") 
    postprocessing.postprocessing()
    # print(postprocessing.counter, 'realizations * num_p_values')

    # for threshold in np.logspace(-15, -5, 10):
    #     postprocessing.h5_to_csv(threshold)
    #     postprocessing.plot_from_csv(threshold)

    # postprocessing.fixed_L_threshold_comparison_plot(L=24)