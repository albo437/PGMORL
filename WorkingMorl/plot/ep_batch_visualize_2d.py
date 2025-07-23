'''
Visualize the pareto fronts of all runs (different seeds) of one algorithm for one problem.
Adapted for WorkingMorl - supports both 2D and 3D results visualization.
'''
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add parent directory to path to import utils
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

# compute the hypervolume given the pareto points
def compute_hypervolume_sparsity(obj_batch, ref_point):
    objs = obj_batch[get_ep_indices(obj_batch)]
    objs = np.array(objs)
    print('input size : {}, pareto size : {}'.format(len(obj_batch), len(objs)))

    if len(objs[0]) == 2:
        # 2D hypervolume calculation
        ref_x, ref_y = ref_point[:2]
        x, hypervolume = ref_x, 0.0
        sparsity = 0.0
        for i in range(len(objs)):
            hypervolume += (max(ref_x, objs[i][0]) - x) * (max(ref_y, objs[i][1]) - ref_y)
            x = max(ref_x, objs[i][0])
            if i > 0:
                sparsity += np.sum(np.square(objs[i] - objs[i - 1]))
        
        if len(objs) == 1:
            sparsity = 0.0
        else:
            sparsity = sparsity / (len(objs) - 1)
    else:
        # 3D+ hypervolume - use simple volume estimation
        if len(objs) == 0:
            return 0.0, 0.0
        
        # Simple hypervolume approximation for 3D+
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
parser.add_argument('--log-dir', type=str, required=True, help='Directory containing result folders')
parser.add_argument('--save-fig', default=False, action='store_true', help='Save figure instead of showing')
parser.add_argument('--title', type=str, default=None, help='Plot title')
parser.add_argument('--obj', type=str, nargs='+', default=None, help='Objective names')
parser.add_argument('--num-seeds', type=int, default=1, help='Number of different seed runs')
parser.add_argument('--ref-point', type=float, nargs='+', default=[0., 0., 0.], help='Reference point for hypervolume')
args = parser.parse_args()

# Auto-detect if we have 2D or 3D results
sample_file = None
for seed in range(args.num_seeds):
    for subdir in ['final', 'test']:
        potential_path = os.path.join(args.log_dir, str(seed) if args.num_seeds > 1 else '', subdir, 'objs.txt')
        if os.path.exists(potential_path):
            sample_file = potential_path
            break
    if sample_file:
        break

if not sample_file:
    # Try direct path
    sample_file = os.path.join(args.log_dir, 'final', 'objs.txt')

if not os.path.exists(sample_file):
    print(f"Error: Could not find objs.txt in {args.log_dir}")
    sys.exit(1)

# Read sample to detect dimensionality
with open(sample_file, 'r') as fp:
    first_line = fp.readline().strip()
    if first_line:
        num_objectives = len(first_line.split(','))
        print(f"Detected {num_objectives}D objectives")

# Set default objective names if not provided
if args.obj is None:
    if num_objectives == 2:
        args.obj = ['Distance', 'Efficiency']
    elif num_objectives == 3:
        args.obj = ['Distance', 'Efficiency', 'Stability']
    else:
        args.obj = [f'Obj {i+1}' for i in range(num_objectives)]

# Create visualization
if num_objectives == 2:
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
elif num_objectives == 3:
    fig = plt.figure(figsize=(15, 5))
    ax = [fig.add_subplot(131), fig.add_subplot(132), fig.add_subplot(133, projection='3d')]
else:
    # For higher dimensions, show 2D projections
    n_plots = min(6, num_objectives * (num_objectives - 1) // 2)
    cols = 3
    rows = (n_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if rows == 1:
        axes = [axes] if n_plots == 1 else axes
    else:
        axes = axes.flatten()

legends = []
hv_mean, sp_mean = 0, 0

for ii in range(args.num_seeds):
    objs_file = 'objs.txt'
    if args.num_seeds > 1:
        log_path = os.path.join(args.log_dir, str(ii), 'final', objs_file)
        if not os.path.exists(log_path): 
            log_path = os.path.join(args.log_dir, str(ii), 'test', objs_file)
    else:
        log_path = os.path.join(args.log_dir, 'final', objs_file)
        if not os.path.exists(log_path):
            log_path = os.path.join(args.log_dir, 'test', objs_file)
    
    if not os.path.exists(log_path):
        print(f"Warning: {log_path} does not exist, skipping seed {ii}")
        continue
        
    with open(log_path, 'r') as fp:
        data = fp.readlines()
        rew_data = []
        for j, line_data in enumerate(data):
            line_data = line_data.split(',')
            line_data = list(map(lambda x: float(x), line_data))
            rew_data.append(line_data)
        
        if len(rew_data) == 0:
            print(f"Warning: No data in {log_path}, skipping seed {ii}")
            continue
            
        objs = rew_data.copy()
        rew_data = np.array(rew_data)

        # Plot based on dimensionality
        if num_objectives == 2:
            ax.scatter(rew_data[:, 0], rew_data[:, 1], alpha=0.7, s=50)
        elif num_objectives == 3:
            # 2D projections
            ax[0].scatter(rew_data[:, 0], rew_data[:, 1], alpha=0.7, s=50)
            ax[1].scatter(rew_data[:, 0], rew_data[:, 2], alpha=0.7, s=50)
            # 3D plot
            ax[2].scatter(rew_data[:, 0], rew_data[:, 1], rew_data[:, 2], alpha=0.7, s=50)
        else:
            # Multiple 2D projections
            plot_idx = 0
            for i in range(num_objectives):
                for j in range(i+1, num_objectives):
                    if plot_idx < len(axes):
                        axes[plot_idx].scatter(rew_data[:, i], rew_data[:, j], alpha=0.7, s=50)
                        axes[plot_idx].set_xlabel(args.obj[i])
                        axes[plot_idx].set_ylabel(args.obj[j])
                        plot_idx += 1

    hypervolume, sparsity = compute_hypervolume_sparsity(np.array(objs), args.ref_point)

    print(f'Seed {ii}: hypervolume = {hypervolume:.0f}, sparsity = {sparsity:.0f}')
    
    if args.num_seeds > 1:
        legends.append(f'Seed {ii}, H = {hypervolume:.0f}, S = {sparsity:.0f}')
    else:
        legends.append(f'H = {hypervolume:.0f}, S = {sparsity:.0f}')

    hv_mean += hypervolume / args.num_seeds
    sp_mean += sparsity / args.num_seeds

print(f'Mean hypervolume = {hv_mean:.0f}, mean sparsity = {sp_mean:.0f}')

# Set labels and title
if num_objectives == 2:
    ax.set_xlabel(args.obj[0])
    ax.set_ylabel(args.obj[1])
    ax.legend(legends, loc='best')
elif num_objectives == 3:
    ax[0].set_xlabel(args.obj[0])
    ax[0].set_ylabel(args.obj[1])
    ax[1].set_xlabel(args.obj[0])
    ax[1].set_ylabel(args.obj[2])
    ax[2].set_xlabel(args.obj[0])
    ax[2].set_ylabel(args.obj[1])
    ax[2].set_zlabel(args.obj[2])
    ax[0].legend(legends, loc='best')

if args.title is not None:
    plt.suptitle(args.title)
elif num_objectives == 2:
    plt.title('2D Pareto Front')
elif num_objectives == 3:
    plt.suptitle('3D Pareto Front Analysis')

plt.tight_layout()

if args.save_fig:
    fig_path = os.path.join(args.log_dir, f'pareto_comparison_{num_objectives}d.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {fig_path}")

plt.show()
