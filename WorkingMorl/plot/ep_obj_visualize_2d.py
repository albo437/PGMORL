'''
Visualize the computed Pareto policies in the performance space for WorkingMorl.
Simple visualization without interactive policy playing (since we don't use mujoco).
'''
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

np.set_printoptions(precision=1)

def get_ep_indices(obj_batch):
    # return sorted indices of undominated objs
    if len(obj_batch) == 0: return np.array([])
    if len(obj_batch[0]) == 2:
        # 2D case
        sort_indices = np.lexsort((obj_batch.T[1], obj_batch.T[0]))
        ep_indices = []
        max_val = -np.inf
        for idx in sort_indices[::-1]:
            if obj_batch[idx][1] > max_val:
                max_val = obj_batch[idx][1]
                ep_indices.append(idx)
        return ep_indices[::-1]
    else:
        # 3D+ case - use proper Pareto dominance
        obj_batch = np.array(obj_batch)
        n = len(obj_batch)
        is_efficient = np.ones(n, dtype=bool)
        for i in range(n):
            if is_efficient[i]:
                # Check if obj_batch[i] is dominated by any other point
                is_efficient[i] = not np.any(np.all(obj_batch > obj_batch[i], axis=1))
                # Remove dominated points
                is_efficient[np.all(obj_batch <= obj_batch[i], axis=1) & 
                           np.any(obj_batch < obj_batch[i], axis=1)] = False
        return np.where(is_efficient)[0]

def compute_hypervolume_sparsity(obj_batch, ref_point):
    if len(obj_batch) == 0:
        return 0.0, 0.0
        
    objs = obj_batch[get_ep_indices(obj_batch)]
    
    if len(obj_batch[0]) == 2:
        # 2D case
        ref_x, ref_y = ref_point[:2]
        x, hypervolume = ref_x, 0.0
        sqdist = 0.0
        for i in range(len(objs)):
            hypervolume += (max(ref_x, objs[i][0]) - x) * (max(ref_y, objs[i][1]) - ref_y)
            x = max(ref_x, objs[i][0])
            if i > 0:
                sqdist += np.sum(np.square(objs[i] - objs[i - 1]))

        if len(objs) == 1:
            sparsity = 0.0
        else:
            sparsity = sqdist / (len(objs) - 1)
    else:
        # 3D+ case
        if len(objs) == 0:
            return 0.0, 0.0
        
        # Simple hypervolume approximation
        ranges = np.max(objs, axis=0) - np.array(ref_point[:len(objs[0])])
        hypervolume = np.prod(np.maximum(ranges, 0))
        
        # Sparsity calculation
        if len(objs) < 2:
            sparsity = 0.0
        else:
            distances = []
            for i in range(len(objs)):
                for j in range(i+1, len(objs)):
                    distances.append(np.linalg.norm(objs[i] - objs[j]))
            sparsity = np.mean(distances) if distances else 0.0

    return hypervolume, sparsity

parser = argparse.ArgumentParser()
parser.add_argument('--log-dir', type=str, required=True, help='Path to results directory')
parser.add_argument('--env', type=str, default='MO-Dummy-v0', help='Environment name')
parser.add_argument('--save-fig', default=False, action='store_true', help='Save figure')
parser.add_argument('--title', type=str, default=None, help='Plot title')
parser.add_argument('--obj', type=str, nargs='+', default=None, help='Objective names')
parser.add_argument('--ref-point', type=float, nargs='+', default=[0., 0., 0.], help='Reference point')

args = parser.parse_args()

# Load objectives
objs_file = os.path.join(args.log_dir, 'final', 'objs.txt')
if not os.path.exists(objs_file):
    objs_file = os.path.join(args.log_dir, 'objs.txt')
    
if not os.path.exists(objs_file):
    print(f"Error: Could not find objs.txt in {args.log_dir}")
    sys.exit(1)

objs = []
with open(objs_file, 'r') as fp:
    data = fp.readlines()
    for line_data in data:
        line_data = line_data.split(',')
        line_data = [float(x) for x in line_data if x.strip()]
        if line_data:
            objs.append(line_data)

if len(objs) == 0:
    print("No objectives found in file")
    sys.exit(1)

objs = np.array(objs)
num_objectives = objs.shape[1]

print(f"Loaded {len(objs)} solutions with {num_objectives} objectives")

# Auto-detect objective names
if args.obj is None:
    if 'Dummy' in args.env:
        if num_objectives == 2:
            args.obj = ['Distance', 'Efficiency']
        elif num_objectives == 3:
            args.obj = ['Distance', 'Efficiency', 'Stability']
        else:
            args.obj = [f'Objective {i+1}' for i in range(num_objectives)]
    elif args.env in ['MO-HalfCheetah-v2', 'MO-Walker2d-v2', 'MO-Swimmer-v2', 'MO-Humanoid-v2']:
        args.obj = ['Forward Speed', 'Energy Efficiency']
    elif args.env == 'MO-Ant-v2':
        args.obj = ['X-Axis Speed', 'Y-Axis Speed']
    elif args.env == 'MO-Hopper-v2':
        args.obj = ['Running Speed', 'Jumping Height']
    else:
        args.obj = [f'Objective {i+1}' for i in range(num_objectives)]

# Get Pareto-efficient points
ep_indices = get_ep_indices(objs)
pareto_objs = objs[ep_indices]

print(f"Pareto-efficient solutions: {len(pareto_objs)}")
print(f"Objective ranges:")
for i, obj_name in enumerate(args.obj):
    print(f"  {obj_name}: [{objs[:, i].min():.3f}, {objs[:, i].max():.3f}]")

# Compute metrics
hypervolume, sparsity = compute_hypervolume_sparsity(objs, args.ref_point)
print(f"Hypervolume: {hypervolume:.2f}")
print(f"Sparsity: {sparsity:.2f}")

# Create visualization
if num_objectives == 2:
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot all solutions
    ax.scatter(objs[:, 0], objs[:, 1], alpha=0.6, s=50, color='lightblue', label=f'All solutions ({len(objs)})')
    
    # Highlight Pareto front
    ax.scatter(pareto_objs[:, 0], pareto_objs[:, 1], s=80, color='red', edgecolors='darkred', linewidth=1, label=f'Pareto front ({len(pareto_objs)})')
    
    # Connect Pareto points
    if len(pareto_objs) > 1:
        ax.plot(pareto_objs[:, 0], pareto_objs[:, 1], 'r--', alpha=0.5, linewidth=1)
    
    ax.set_xlabel(args.obj[0])
    ax.set_ylabel(args.obj[1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
elif num_objectives == 3:
    fig = plt.figure(figsize=(18, 6))
    
    # 3D plot
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(objs[:, 0], objs[:, 1], objs[:, 2], alpha=0.6, s=50, color='lightblue', label=f'All solutions ({len(objs)})')
    ax1.scatter(pareto_objs[:, 0], pareto_objs[:, 1], pareto_objs[:, 2], s=80, color='red', edgecolors='darkred', linewidth=1, label=f'Pareto front ({len(pareto_objs)})')
    ax1.set_xlabel(args.obj[0])
    ax1.set_ylabel(args.obj[1])
    ax1.set_zlabel(args.obj[2])
    ax1.legend()
    ax1.set_title('3D View')
    
    # 2D projections
    ax2 = fig.add_subplot(132)
    ax2.scatter(objs[:, 0], objs[:, 1], alpha=0.6, s=50, color='lightblue')
    ax2.scatter(pareto_objs[:, 0], pareto_objs[:, 1], s=80, color='red', edgecolors='darkred', linewidth=1)
    ax2.set_xlabel(args.obj[0])
    ax2.set_ylabel(args.obj[1])
    ax2.grid(True, alpha=0.3)
    ax2.set_title(f'{args.obj[0]} vs {args.obj[1]}')
    
    ax3 = fig.add_subplot(133)
    ax3.scatter(objs[:, 0], objs[:, 2], alpha=0.6, s=50, color='lightblue')
    ax3.scatter(pareto_objs[:, 0], pareto_objs[:, 2], s=80, color='red', edgecolors='darkred', linewidth=1)
    ax3.set_xlabel(args.obj[0])
    ax3.set_ylabel(args.obj[2])
    ax3.grid(True, alpha=0.3)
    ax3.set_title(f'{args.obj[0]} vs {args.obj[2]}')
    
else:
    # For higher dimensions, show pairwise projections
    n_plots = min(6, num_objectives * (num_objectives - 1) // 2)
    cols = 3
    rows = (n_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    
    if rows == 1:
        axes = [axes] if n_plots == 1 else axes
    else:
        axes = axes.flatten()
    
    plot_idx = 0
    for i in range(num_objectives):
        for j in range(i+1, num_objectives):
            if plot_idx < len(axes):
                ax = axes[plot_idx]
                ax.scatter(objs[:, i], objs[:, j], alpha=0.6, s=50, color='lightblue')
                ax.scatter(pareto_objs[:, i], pareto_objs[:, j], s=80, color='red', edgecolors='darkred', linewidth=1)
                ax.set_xlabel(args.obj[i])
                ax.set_ylabel(args.obj[j])
                ax.grid(True, alpha=0.3)
                ax.set_title(f'{args.obj[i]} vs {args.obj[j]}')
                plot_idx += 1
    
    # Hide unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].set_visible(False)

# Add title
if args.title:
    plt.suptitle(args.title)
else:
    plt.suptitle(f'{args.env} - Pareto Front Analysis\\nHV: {hypervolume:.2f}, Sparsity: {sparsity:.2f}')

plt.tight_layout()

# Add click handler for 2D case to show point info
if num_objectives == 2:
    def on_click(event):
        if event.inaxes == ax and event.dblclick:
            # Find nearest point
            distances = np.sqrt((objs[:, 0] - event.xdata)**2 + (objs[:, 1] - event.ydata)**2)
            nearest_idx = np.argmin(distances)
            nearest_obj = objs[nearest_idx]
            
            print(f"\\nClicked point (index {nearest_idx}):")
            for i, obj_name in enumerate(args.obj):
                print(f"  {obj_name}: {nearest_obj[i]:.3f}")
            
            # Check if it's on Pareto front
            if nearest_idx in ep_indices:
                print("  Status: Pareto-efficient")
            else:
                print("  Status: Dominated")
    
    fig.canvas.mpl_connect('button_press_event', on_click)
    print("\\nDouble-click on points to see their values")

if args.save_fig:
    fig_path = os.path.join(args.log_dir, f'pareto_analysis_{num_objectives}d.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {fig_path}")

plt.show()
