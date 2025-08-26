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
    def __init__(self, p_fixed_name: str, p_fixed_value: float, n: int):
        self.p_fixed_name = p_fixed_name
        self.p_fixed_value = p_fixed_value
        self.n = n
        print("p_fixed_name", self.p_fixed_name, "p_fixed_value", self.p_fixed_value, "n", self.n)
        self.sv_combined = "/scratch/ty296/hdf5_data_combined/sv_combined_{}{}.h5".format(self.p_fixed_name, self.p_fixed_value)
        self.dir_name = "/scratch/ty296/hdf5_data/{}{}".format(self.p_fixed_name, self.p_fixed_value)
        self.save_folder = '/scratch/ty296/plots'

    def postprocessing(self):
        h5_files = glob.glob(os.path.join(self.dir_name, '*.h5'))
        with h5py.File(self.sv_combined, 'w') as f_combined:
            counter = 0
            for file in tqdm.tqdm(h5_files):
                with h5py.File(file, 'r') as f:
                    metadata = f['metadata']
                    singular_values = f['singular_values']
                    for result_group in metadata.keys():
                        counter += 1
                        p_proj = metadata[result_group]['p_proj'][()]
                        p_ctrl = metadata[result_group]['p_ctrl'][()]
                        L = metadata[result_group]['args']['L'][()]
                        maxbond = metadata[result_group]['max_bond'][()]
                        sv_arr = singular_values[result_group][()]
                        group_name = f'real{counter}'
                        grp = f_combined.create_dataset(group_name, data=sv_arr)
                        grp.attrs['p_proj'] = p_proj
                        grp.attrs['p_ctrl'] = p_ctrl
                        grp.attrs['L'] = L
                        grp.attrs['maxbond'] = maxbond

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
                mean, sem = calculate_mean_and_error(s0_list)
                variance, se_var = calculate_variance_and_error(s0_list)
                # print(key_val, "mean", mean, "sem", sem, "variance", variance, "se_var", se_var)
                data.append(list(key_val) + [mean, sem, variance, se_var])
            # print(data)

            df = pd.DataFrame(data, columns=['L', 'p_ctrl', 'p_proj', 'mean', 'sem', 'variance', 'se_var'])
            # save the data to a csv file
            csv_path = os.path.join(self.save_folder, f's{self.n}_threshold{threshold:.1e}_{self.p_fixed_name}{self.p_fixed_value}.csv')
            df.to_csv(csv_path, index=False)

            return df

    def plot_from_csv(self, threshold: float):
        """
        Plot p_proj vs mean±SEM and p_proj vs variance±SEVAR from CSV file
        CSV should have columns: L, p_ctrl, p_proj, mean, sem, variance, se_var
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
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

        # Plot 1: p_proj vs mean ± sem
        for L in sorted(plot_data.keys()):
            data = plot_data[L]
            # Sort by p_proj for cleaner lines
            sorted_indices = np.argsort(data['p_proj'])
            p_proj_sorted = np.array(data['p_proj'])[sorted_indices]
            mean_sorted = np.array(data['mean'])[sorted_indices]
            sem_sorted = np.array(data['sem'])[sorted_indices]
            
            ax1.errorbar(p_proj_sorted, mean_sorted, yerr=sem_sorted, 
                        label=f'L={L}', marker='o', capsize=5, capthick=2)

        ax1.set_xlabel('p_proj')
        ax1.set_ylabel('Mean Entropy ± SEM')
        ax1.set_title('Mean Entropy vs p_proj for Different L')
        ax1.legend()
        ax1.set_xlim(0.0, 1.0)
        ax1.grid(True, alpha=0.3)

        # Plot 2: p_proj vs variance ± se_var
        for L in sorted(plot_data.keys()):
            data = plot_data[L]
            # Sort by p_proj for cleaner lines
            sorted_indices = np.argsort(data['p_proj'])
            p_proj_sorted = np.array(data['p_proj'])[sorted_indices]
            variance_sorted = np.array(data['variance'])[sorted_indices]
            se_var_sorted = np.array(data['se_var'])[sorted_indices]
            
            ax2.errorbar(p_proj_sorted, variance_sorted, yerr=se_var_sorted, 
                        label=f'L={L}', marker='s', capsize=5, capthick=2)

        ax2.set_xlabel('p_proj')
        ax2.set_ylabel('Variance ± SEVar')
        ax2.set_title('Variance vs p_proj for Different L')
        ax2.legend()
        ax2.set_xlim(0.0, 1.0)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'/scratch/ty296/plots/s{self.n}_threshold{threshold:.1e}_{self.p_fixed_name}{self.p_fixed_value}.png')
        # plt.show()


# %%
# Example usage:
# First generate CSV using h5_to_csv function:

if __name__ == "__main__":

    p_fixed_name = 'p_ctrl'
    p_fixed_value = 0.0
    n = 0
    # postprocessing = Postprocessing(p_fixed_name, p_fixed_value, n, threshold=1e-16) 
    # postprocessing.postprocessing()

    for threshold in np.logspace(-15, -5, 10):
        postprocessing = Postprocessing(p_fixed_name, p_fixed_value, n) 
        print(threshold)

        # postprocessing.postprocessing() # once run this once to combine all the hdf5 files
        postprocessing.h5_to_csv(threshold)
        postprocessing.plot_from_csv(threshold)
