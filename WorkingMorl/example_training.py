#!/usr/bin/env python3
"""
Example script showing how to use the MORL system with a custom environment.
This demonstrates how to integrate your own multi-objective environments.
"""

import os
import sys
import torch
import gym
import numpy as np

# Add the current directory and morl directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'morl'))
sys.path.insert(0, os.path.join(current_dir, 'externals/baselines'))
sys.path.insert(0, os.path.join(current_dir, 'externals/pytorch-a2c-ppo-acktr-gail'))

# Import and register our dummy environment
import environments.dummy_mo_env
print("Dummy multi-objective environment registered successfully")

# Verify registration
available_envs = [env_id for env_id in gym.envs.registry.env_specs.keys() if 'MO-Dummy' in env_id]
print(f"Available environments: {available_envs}")

# Import MORL components
from arguments import get_args
from morl import run as morl_train

def main():
    print("=" * 60)
    print("PGMORL 2D TRAINING EXAMPLE")
    print("=" * 60)
    print(f"Environment: MO-Dummy-v0")
    print(f"Objectives: 2")
    print(f"Selection Method: Prediction-guided (PGMORL)")
    print(f"Total steps: 5000 (fast test)")
    print(f"Processes: 1")
    print(f"Results will be saved to: ./example_results")
    print()
    
    # Set up arguments for MORL training
    sys.argv = ['example_training.py',
                '--env-name', 'MO-Dummy-v0',
                '--obj-num', '2',
                '--num-steps', '64',   # Reduced from 128
                '--ppo-epoch', '2',    # Reduced from 4
                '--num-mini-batch', '2',  # Reduced from 4
                '--lr', '3e-4',
                '--gamma', '0.99',
                '--gae-lambda', '0.95',
                '--num-env-steps', '5000',  # Much shorter - reduced from 10000
                '--num-processes', '1',     # Single process to reduce overhead
                '--warmup-iter', '2',       # Quick warmup - same as 3D
                '--update-iter', '4',       # Match 3D version
                '--min-weight', '0.0',      # Weight parameters
                '--max-weight', '1.0',      
                '--delta-weight', '0.2',    # Keep smaller delta for 2D
                '--eval-num', '1',          
                '--num-tasks', '3',         # Reduced from 6 to match 3D
                '--pbuffer-num', '20',      # Much smaller - reduced from 100
                '--pbuffer-size', '1',      # Reduced from 2
                '--selection-method', 'prediction-guided',  # Use PGMORL's main method
                '--num-weight-candidates', '3',  # Reduced from 7 to match 3D
                '--obj-rms',                # Missing parameter - objective normalization
                '--ob-rms',                 # Missing parameter - observation normalization
                '--raw',                    # Missing parameter
                '--save-dir', './example_results']
    
    print("Starting MORL training...")
    print()
    
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
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Results saved to: ./example_results")
        print("Check the logs for training progress and final Pareto front.")
        
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
