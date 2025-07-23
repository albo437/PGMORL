"""
PGMORL Training on Deep Sea Treasure Environment
==================================================

Train the PGMORL algorithm on the classic Deep Sea Treasure benchmark.
This environment has exactly 3 Pareto-optimal points representing different
trade-offs between treasure value and fuel consumption.

Expected Pareto Front:
- Point 1: (Treasure=1.0, Fuel=-1)   - Quick & Small
- Point 2: (Treasure=8.0, Fuel=-6)   - Medium Effort & Reward  
- Point 3: (Treasure=50.0, Fuel=-14) - High Effort & High Reward
"""

import os
import sys
import torch
import gym
import numpy as np

# Add the current directory and necessary paths to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'morl'))
sys.path.insert(0, os.path.join(current_dir, 'externals/baselines'))
sys.path.insert(0, os.path.join(current_dir, 'externals/pytorch-a2c-ppo-acktr-gail'))

# Import and register Deep Sea Treasure environment
import environments.deep_sea_treasure
print("Deep Sea Treasure environment imported")

# Import configuration
from deep_sea_treasure_config import (
    get_all_args, print_config_summary, apply_fast_test_config, 
    apply_full_training_config, SAVE_DIR, EXPECTED_PARETO_FRONT
)

def convert_to_classic_format(results_dir):
    """
    Convert the saved objectives from PGMORL format to classic Deep Sea Treasure format.
    
    PGMORL format: (treasure_rate, fuel_efficiency) - both positive
    Classic format: (treasure_value, -steps) - negative steps for fuel cost
    
    We need to:
    1. Use actual treasure values (1, 3, 8, 20) instead of efficiency rates
    2. Convert fuel efficiency back to negative step count
    """
    objs_file = os.path.join(results_dir, 'final', 'objs.txt')
    
    if not os.path.exists(objs_file):
        print(f"Warning: {objs_file} not found, skipping conversion")
        return
    
    # Read current objectives
    objectives = []
    with open(objs_file, 'r') as f:
        for line in f:
            if line.strip():
                vals = [float(x) for x in line.strip().split(',')]
                objectives.append(vals)
    
    if not objectives:
        print("No objectives found to convert")
        return
    
    print(f"Converting {len(objectives)} objectives to classic format...")
    
    # With --raw flag, we should get the actual step-based efficiency values
    obj2_values = [obj2 for obj1, obj2 in objectives]
    min_obj2, max_obj2 = min(obj2_values), max(obj2_values)
    print(f"Efficiency range: {min_obj2:.3f} to {max_obj2:.3f}")
    
    # Convert to classic format
    classic_objectives = []
    for obj1, obj2 in objectives:
        # obj1 = treasure value
        # obj2 = fuel efficiency = 10.0/(steps + 1)
        # Convert efficiency back to steps: steps = (10.0/obj2) - 1
        
        if obj2 > 0:
            estimated_steps = max(1, int(round((10.0 / obj2) - 1)))
        else:
            estimated_steps = 50  # fallback for edge cases
        
        # Determine actual treasure value based on obj1
        if obj1 >= 20.0:
            treasure_value = 20.0
        elif obj1 >= 8.0:
            treasure_value = 8.0
        elif obj1 >= 3.0:
            treasure_value = 3.0
        elif obj1 >= 1.0:
            treasure_value = 1.0
        else:
            # No treasure found, use minimal value for comparison
            treasure_value = 0.0
        
        # Classic format: (treasure_value, -steps)
        classic_objectives.append([treasure_value, -estimated_steps])
    
    # Save converted objectives
    with open(objs_file, 'w') as f:
        for treasure, neg_steps in classic_objectives:
            f.write(f'{treasure:.6f},{neg_steps:.6f}\n')
    
    print(f"Converted objectives to classic format and saved to {objs_file}")
    print("Classic format: (treasure_value, -steps)")
    
    # Show some examples
    unique_objectives = []
    for obj in classic_objectives:
        if obj not in unique_objectives:
            unique_objectives.append(obj)
    
    print(f"Found {len(unique_objectives)} unique solutions:")
    for treasure, neg_steps in sorted(unique_objectives):
        print(f"  Treasure: {treasure:.1f}, Steps: {int(-neg_steps)}")

# Verify registration
try:
    # Try new Gym API first
    if hasattr(gym.envs.registry, 'all'):
        available_envs = [env.id for env in gym.envs.registry.all() if 'DeepSeaTreasure' in env.id]
    else:
        # Fallback to older API
        available_envs = [env_id for env_id in gym.envs.registry.env_specs.keys() if 'DeepSeaTreasure' in env_id]
except AttributeError:
    # Alternative approach for newer Gym versions
    available_envs = [env_id for env_id in gym.envs.registry.keys() if 'DeepSeaTreasure' in env_id]

print(f"Available Deep Sea Treasure environments: {available_envs}")

# Import MORL components
from arguments import get_args
from morl import run as morl_train

def main():
    print("=" * 60)
    print("PGMORL TRAINING - DEEP SEA TREASURE")
    print("=" * 60)
    
    # Print current configuration
    print_config_summary()
    
    # Uncomment one of these to apply preset configurations:
    # apply_fast_test_config()        # For quick testing (5,000 steps)
    # apply_full_training_config()    # For full training (100,000 steps)
    # apply_large_network_config()    # For larger network (128 hidden units)
    # apply_small_network_config()    # For smaller network (32 hidden units)
    
    print("Starting MORL training...")
    print()
    
    # Get arguments from configuration file
    sys.argv = get_all_args()
    
    # Parse arguments and start training
    args = get_args()
    
    # Set device
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    
    print(f"Device: {args.device}")
    print(f"Using {args.num_processes} parallel environments")
    print()
    
    # Start training
    try:
        morl_train(args)
        
        # POST-PROCESS: Convert objectives to classic Deep Sea Treasure format
        convert_to_classic_format(SAVE_DIR)
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Results saved to: {SAVE_DIR}")
        print("Check the logs for training progress and final Pareto front.")
        print()
        print("Expected Pareto-optimal solutions:")
        for point in EXPECTED_PARETO_FRONT:
            print(f"  Treasure: {point['treasure']}, Steps: {point['steps']} ({point['description']})")
        print()
        print("Note: objs.txt now shows classic format (treasure_value, -steps)")
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
