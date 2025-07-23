'''
Visualize the training process for WorkingMorl.
Supports both 2D and 3D objectives with interactive controls.
'''
import argparse
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import os
from copy import deepcopy
from multiprocessing import Process
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

np.set_printoptions(precision=1)

iterations_str = []
iterations = []
ep_objs = []
population_objs = []
elites_objs = []
elites_weights = []
predictions = []
offsprings = []

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

# compute the hypervolume and sparsity given the pareto points
def compute_metrics(obj_batch):
    if len(obj_batch) == 0:
        return 0.0, 0.0
        
    objs = obj_batch[get_ep_indices(obj_batch)]
    
    if len(obj_batch[0]) == 2:
        # 2D case
        ref_x, ref_y = 0.0, 0.0  # reference point
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
        ranges = np.max(objs, axis=0)
        hypervolume = np.prod(ranges) if len(ranges) > 0 else 0.0
        
        # Sparsity calculation
        if len(objs) < 2:
            sparsity = 0.0
        else:
            distances = []
            for i in range(len(objs)):
                for j in range(i+1, len(objs)):
                    distances.append(np.linalg.norm(objs[i] - objs[j]))
            sparsity = np.mean(distances) if distances else 0.0

    print('Pareto size : {}, Hypervolume : {:.0f}, Sparsity : {:.2f}'.format(len(objs), hypervolume, sparsity))
    return hypervolume, sparsity

def get_objs(objs_path):
    objs = []
    if os.path.exists(objs_path):
        with open(objs_path, 'r') as fp:
            data = fp.readlines()
            for j, line_data in enumerate(data):
                line_data = line_data.split(',')
                obj_values = [float(x) for x in line_data if x.strip()]
                if obj_values:  # Only add if we have valid data
                    objs.append(obj_values)
    return objs

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='MO-Dummy-v0')
parser.add_argument('--log-dir', type=str, required=True)
parser.add_argument('--save-fig', default=False, action='store_true')
parser.add_argument('--title', type=str, default=None)
parser.add_argument('--obj', type=str, nargs='+', default=None)

args = parser.parse_args()

# Auto-detect objective names based on environment
if args.obj is None:
    if 'Dummy' in args.env:
        # Check sample file to detect dimensionality
        sample_file = os.path.join(args.log_dir, 'final', 'objs.txt')
        if os.path.exists(sample_file):
            with open(sample_file, 'r') as fp:
                first_line = fp.readline().strip()
                if first_line:
                    num_objectives = len(first_line.split(','))
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
        args.obj = ['Objective 1', 'Objective 2']

# Load training data
all_iteration_folders = os.listdir(args.log_dir)
for folder in all_iteration_folders:
    if os.path.isdir(os.path.join(args.log_dir, folder)) and folder != 'final':
        try:
            iterations.append(int(folder))
            population_log_dir = os.path.join(args.log_dir, folder, 'population')
            # load population objs
            population_objs.append(get_objs(os.path.join(population_log_dir, 'objs.txt')))
            # load ep
            ep_log_dir = os.path.join(args.log_dir, folder, 'ep')
            ep_objs.append(get_objs(os.path.join(ep_log_dir, 'objs.txt')))
            # load elites
            elites_log_dir = os.path.join(args.log_dir, folder, 'elites')
            elites_objs.append(get_objs(os.path.join(elites_log_dir, 'elites.txt')))
            elites_weights.append(get_objs(os.path.join(elites_log_dir, 'weights.txt')))
            predictions.append(get_objs(os.path.join(elites_log_dir, 'predictions.txt')))
            offsprings.append(get_objs(os.path.join(elites_log_dir, 'offsprings.txt')))
        except ValueError:
            # Skip non-numeric folder names
            continue

if len(iterations) == 0:
    print(f"No training iterations found in {args.log_dir}")
    sys.exit(1)

# Normalize weights for 2D case
for weights in elites_weights:
    for weight in weights:
        if len(weight) >= 2:
            norm = np.sqrt(weight[0] ** 2 + weight[1] ** 2)
            if norm > 0:
                weight[0] /= norm
                weight[1] /= norm

iterations = np.array(iterations)
ep_objs = np.array(ep_objs, dtype=object)
population_objs = np.array(population_objs, dtype=object)
elites_objs = np.array(elites_objs, dtype=object)
elites_weights = np.array(elites_weights, dtype=object)
predictions = np.array(predictions, dtype=object)
offsprings = np.array(offsprings, dtype=object)

have_pred = len(predictions) > 0 and len(predictions[0]) > 0
have_offspring = len(offsprings) > 0 and len(offsprings[0]) > 0

# Sort by iteration
sorted_index = np.argsort(iterations)
sorted_ep_objs = []
sorted_population_objs = []
sorted_elites_objs = []
sorted_elites_weights = []
sorted_predictions = []
sorted_offsprings = []
utopians = []

for i in range(len(sorted_index)):
    index = sorted_index[i]
    sorted_ep_objs.append(deepcopy(ep_objs[index]))
    sorted_population_objs.append(deepcopy(population_objs[index]))
    sorted_elites_objs.append(deepcopy(elites_objs[index]))
    sorted_elites_weights.append(deepcopy(elites_weights[index]))
    if have_pred and len(predictions[index]) > 0:
        sorted_predictions.append(deepcopy(predictions[index]))
    else:
        sorted_predictions.append([])
    if have_offspring and len(offsprings) > index and len(offsprings[index]) > 0:
        if i < len(sorted_index) - 1:
            next_idx = sorted_index[i + 1] if sorted_index[i + 1] < len(offsprings) else index
            sorted_offsprings.append(deepcopy(offsprings[next_idx]))
        else:
            sorted_offsprings.append([])
    else:
        sorted_offsprings.append([])
    
    # Calculate utopian point
    if len(sorted_ep_objs[i]) > 0:
        utopian = np.max(sorted_ep_objs[i], axis=0)
        utopians.append(utopian)
    else:
        utopians.append([0, 0] if len(args.obj) == 2 else [0] * len(args.obj))

# Calculate metrics for each iteration
hypervolumes, sparsities = [], []
for i in range(len(sorted_ep_objs)):
    if len(sorted_ep_objs[i]) > 0:
        hypervolume, sparsity = compute_metrics(np.array(sorted_ep_objs[i]))
        hypervolumes.append(hypervolume)
        sparsities.append(sparsity)
    else:
        hypervolumes.append(0.0)
        sparsities.append(0.0)

if len(sorted_ep_objs) > 0 and len(sorted_ep_objs[-1]) > 0:
    print('Final Pareto size : {}, Hypervolume : {:.0f}, Sparsity : {:.2f}'.format(
        len(sorted_ep_objs[-1]), hypervolumes[-1], sparsities[-1]))

# Detect dimensionality
if len(sorted_ep_objs) > 0 and len(sorted_ep_objs[0]) > 0:
    num_objectives = len(sorted_ep_objs[0][0])
else:
    num_objectives = len(args.obj)

# Create appropriate visualization
if num_objectives == 2:
    fig, ax = plt.subplots(1, 3, figsize=(24, 8))
    main_ax = ax[0]
    main_ax.set_aspect('equal')
elif num_objectives == 3:
    fig = plt.figure(figsize=(24, 8))
    ax = [fig.add_subplot(131, projection='3d'), fig.add_subplot(132), fig.add_subplot(133)]
    main_ax = ax[0]
else:
    fig, ax = plt.subplots(1, 3, figsize=(24, 8))
    main_ax = ax[0]

# Set up interactive visualization
print('\\n-----------------------------------------------')
print('> Interactive Training Visualization')
print(f'> Found {len(iterations)} training iterations')
print(f'> Objectives: {num_objectives}D - {args.obj}')
print(f'> Final Pareto front: {len(sorted_ep_objs[-1]) if sorted_ep_objs else 0} solutions')
print('-----------------------------------------------')
print('> Controls:')
print("- change epoch: 'left', 'right'")
print("- turn on/off visualization of population: 'p'")
print("- turn on/off visualization of pareto: 'P'")
print("- turn on/off visualization of elites: 'e'")
print('-----------------------------------------------')

# Initialize visualization elements
epoch_drawings = {
    'pareto': [],
    'population': [],
    'elites': [],
    'utopians': [],
    'hypervolume': [],
    'sparsity': []
}

# Create initial plots
for i in range(len(sorted_ep_objs)):
    # Pareto front
    if len(sorted_ep_objs[i]) > 0:
        ep_data = np.array(sorted_ep_objs[i])
        if num_objectives == 2:
            epoch_drawings['pareto'].append(main_ax.scatter(ep_data[:, 0], ep_data[:, 1], s=15, color='red', label='Pareto'))
        elif num_objectives == 3:
            epoch_drawings['pareto'].append(main_ax.scatter(ep_data[:, 0], ep_data[:, 1], ep_data[:, 2], s=15, color='red', label='Pareto'))
        else:
            # Use first two objectives for visualization
            epoch_drawings['pareto'].append(main_ax.scatter(ep_data[:, 0], ep_data[:, 1], s=15, color='red', label='Pareto'))
    else:
        epoch_drawings['pareto'].append(None)
    
    # Population
    if len(sorted_population_objs[i]) > 0:
        pop_data = np.array(sorted_population_objs[i])
        if num_objectives == 2:
            epoch_drawings['population'].append(main_ax.scatter(pop_data[:, 0], pop_data[:, 1], s=10, color='grey', alpha=0.5, label='Population'))
        elif num_objectives == 3:
            epoch_drawings['population'].append(main_ax.scatter(pop_data[:, 0], pop_data[:, 1], pop_data[:, 2], s=10, color='grey', alpha=0.5, label='Population'))
        else:
            epoch_drawings['population'].append(main_ax.scatter(pop_data[:, 0], pop_data[:, 1], s=10, color='grey', alpha=0.5, label='Population'))
    else:
        epoch_drawings['population'].append(None)
    
    # Elites
    if len(sorted_elites_objs[i]) > 0:
        elite_data = np.array(sorted_elites_objs[i])
        if num_objectives == 2:
            epoch_drawings['elites'].append(main_ax.scatter(elite_data[:, 0], elite_data[:, 1], s=60, facecolors='none', edgecolors='green', linewidth=2, label='Elites'))
        elif num_objectives == 3:
            epoch_drawings['elites'].append(main_ax.scatter(elite_data[:, 0], elite_data[:, 1], elite_data[:, 2], s=60, facecolors='none', edgecolors='green', linewidth=2, label='Elites'))
        else:
            epoch_drawings['elites'].append(main_ax.scatter(elite_data[:, 0], elite_data[:, 1], s=60, facecolors='none', edgecolors='green', linewidth=2, label='Elites'))
    else:
        epoch_drawings['elites'].append(None)
    
    # Utopian points
    if len(utopians[i]) > 0:
        if num_objectives == 2:
            epoch_drawings['utopians'].append(main_ax.scatter(utopians[i][0], utopians[i][1], s=80, marker='*', color='blue', label='Utopian'))
        elif num_objectives == 3:
            epoch_drawings['utopians'].append(main_ax.scatter(utopians[i][0], utopians[i][1], utopians[i][2], s=80, marker='*', color='blue', label='Utopian'))
        else:
            epoch_drawings['utopians'].append(main_ax.scatter(utopians[i][0], utopians[i][1], s=80, marker='*', color='blue', label='Utopian'))
    else:
        epoch_drawings['utopians'].append(None)
    
    # Metrics plots
    if num_objectives == 2:
        epoch_drawings['hypervolume'].append(ax[1].scatter(i, hypervolumes[i], color='red'))
        epoch_drawings['sparsity'].append(ax[2].scatter(i, sparsities[i], color='blue'))

# Plot hypervolume and sparsity lines
if num_objectives == 2:
    ax[1].plot(hypervolumes, color='red', alpha=0.5)
    ax[2].plot(sparsities, color='blue', alpha=0.5)
    ax[1].set_title('Hypervolume')
    ax[1].set_xlabel('Generation')
    ax[2].set_title('Sparsity')
    ax[2].set_xlabel('Generation')

# Set labels
main_ax.set_xlabel(args.obj[0])
main_ax.set_ylabel(args.obj[1])
if num_objectives == 3:
    main_ax.set_zlabel(args.obj[2])

main_ax.set_title('Training Progress')
if args.title:
    fig.suptitle(args.title)

# Hide all but first iteration initially
current_iter = 0
visibility_flags = {
    'pareto': True,
    'population': True,
    'elites': True
}

def update_visibility():
    for i in range(len(epoch_drawings['pareto'])):
        visible = (i == current_iter)
        
        if epoch_drawings['pareto'][i] is not None:
            epoch_drawings['pareto'][i].set_visible(visible and visibility_flags['pareto'])
        if epoch_drawings['population'][i] is not None:
            epoch_drawings['population'][i].set_visible(visible and visibility_flags['population'])
        if epoch_drawings['elites'][i] is not None:
            epoch_drawings['elites'][i].set_visible(visible and visibility_flags['elites'])
        if epoch_drawings['utopians'][i] is not None:
            epoch_drawings['utopians'][i].set_visible(visible)
        
        if num_objectives == 2:
            if epoch_drawings['hypervolume'][i] is not None:
                epoch_drawings['hypervolume'][i].set_visible(visible)
            if epoch_drawings['sparsity'][i] is not None:
                epoch_drawings['sparsity'][i].set_visible(visible)

update_visibility()

def on_key_press(event):
    global current_iter
    
    if event.key == 'right':
        current_iter = min(current_iter + 1, len(epoch_drawings['pareto']) - 1)
    elif event.key == 'left':
        current_iter = max(current_iter - 1, 0)
    elif event.key == 'p':
        visibility_flags['population'] = not visibility_flags['population']
    elif event.key == 'P':
        visibility_flags['pareto'] = not visibility_flags['pareto']
    elif event.key == 'e':
        visibility_flags['elites'] = not visibility_flags['elites']
    
    update_visibility()
    
    # Update title with current info
    if len(sorted_ep_objs[current_iter]) > 0:
        main_ax.set_title(f'Generation {iterations[sorted_index[current_iter]]}: {len(sorted_ep_objs[current_iter])} Pareto solutions, HV={hypervolumes[current_iter]:.0f}')
    else:
        main_ax.set_title(f'Generation {iterations[sorted_index[current_iter]]}: No solutions')
    
    fig.canvas.draw()

fig.canvas.mpl_connect('key_press_event', on_key_press)

plt.tight_layout()

if args.save_fig:
    fig_path = os.path.join(args.log_dir, 'training_visualization.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {fig_path}")

plt.show()
