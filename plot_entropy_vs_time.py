#!/usr/bin/env python3
"""
Plot entropy evolution from benchmark_adder.jl output
"""
import json
import numpy as np
import matplotlib.pyplot as plt

# Load data
with open('entropy_evolution_data.json', 'r') as f:
    data = json.load(f)

time_steps = np.array(data['time_steps'])
entropy_mean = np.array(data['entropy_mean'])
entropy_std = np.array(data['entropy_std'])
params = data['parameters']

# Load individual trajectories if available
if 'entropy_trajectories' in data:
    trajectories = [np.array(traj) for traj in data['entropy_trajectories']]
else:
    trajectories = None

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot individual trajectories (light lines)
if trajectories:
    for traj in trajectories:
        ax.plot(time_steps, traj, '-', linewidth=0.5, alpha=0.3, color='gray')

# Plot ensemble mean with shaded error region
ax.plot(time_steps, entropy_mean, '-', linewidth=2, color='blue', label=f'Ensemble Mean (N={params.get("ensemble_size", 1)})')
ax.fill_between(time_steps, entropy_mean - entropy_std, entropy_mean + entropy_std, 
                 alpha=0.3, color='blue', label='±1 std')

# Labels and title
ax.set_xlabel('Time Step', fontsize=12)
ax.set_ylabel('von Neumann Entropy', fontsize=12)
ax.set_title(f"Entropy Evolution (L={params['L']}, p_ctrl={params['p_ctrl']}, p_proj={params['p_proj']})", 
             fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)

# Add statistics text box
stats_text = f"Final: {entropy_mean[-1]:.4f}±{entropy_std[-1]:.4f}\nMax: {np.max(entropy_mean):.4f}\nMin: {np.min(entropy_mean):.4f}"
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Save figure
plt.tight_layout()
plt.savefig('entropy_evolution.png', dpi=300, bbox_inches='tight')
print("Plot saved as entropy_evolution.png")

plt.show()

