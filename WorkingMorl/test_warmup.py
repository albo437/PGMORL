#!/usr/bin/env python3
"""
Test script to debug MORL warm-up stage
"""

import sys
import os
import torch
import gym
import numpy as np

# Add the current directory and morl directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'morl'))
sys.path.insert(0, os.path.join(current_dir, 'externals/baselines'))
sys.path.insert(0, os.path.join(current_dir, 'externals/pytorch-a2c-ppo-acktr-gail'))

# Import environment
import environments.dummy_mo_env

# Import MORL components
from arguments import get_args
from warm_up import initialize_warm_up_batch
from utils import generate_weights_batch_dfs
from scalarization_methods import WeightedSumScalarization

def test_warmup_initialization():
    print("Testing MORL warm-up initialization...")
    
    # Set up minimal arguments
    sys.argv = ['test_warmup.py',
                '--env-name', 'MO-Dummy-v0',
                '--obj-num', '2',
                '--num-steps', '128',
                '--ppo-epoch', '4',
                '--num-mini-batch', '4',
                '--lr', '3e-4',
                '--gamma', '0.99',
                '--gae-lambda', '0.95',
                '--num-env-steps', '1000',  # Very short for testing
                '--num-processes', '1',     # Single process for debugging
                '--warmup-iter', '1',       # Just one warmup iteration
                '--update-iter', '1',
                '--min-weight', '0.0',
                '--max-weight', '1.0',
                '--delta-weight', '0.2',
                '--eval-num', '1',
                '--num-tasks', '6',
                '--selection-method', 'random',
                '--num-weight-candidates', '7',
                '--obj-rms',
                '--ob-rms',
                '--raw',
                '--save-dir', './test_warmup_results']
    
    # Parse arguments
    args = get_args()
    
    # Set device
    device = torch.device("cpu")
    torch.set_default_dtype(torch.float64)
    
    print(f"Args parsed successfully")
    print(f"Environment: {args.env_name}")
    print(f"Objectives: {args.obj_num}")
    print(f"Device: {device}")
    
    # Test weight generation
    print(f"\nTesting weight generation...")
    weights_batch = []
    generate_weights_batch_dfs(0, args.obj_num, args.min_weight, args.max_weight, args.delta_weight, [], weights_batch)
    print(f"Generated {len(weights_batch)} weights: {weights_batch}")
    
    # Test basic environment creation
    print(f"\nTesting environment creation...")
    temp_env = gym.make(args.env_name)
    print(f"Environment created: {temp_env}")
    print(f"Action space: {temp_env.action_space}")
    print(f"Observation space: {temp_env.observation_space}")
    temp_env.close()
    
    # Test warm-up initialization
    print(f"\nTesting warm-up initialization...")
    try:
        elite_batch, scalarization_batch = initialize_warm_up_batch(args, device)
        
        print(f"Warm-up completed successfully!")
        print(f"Elite batch size: {len(elite_batch)}")
        print(f"Scalarization batch size: {len(scalarization_batch)}")
        
        # Check each sample
        for i, (sample, scalarization) in enumerate(zip(elite_batch, scalarization_batch)):
            print(f"Sample {i}:")
            print(f"  Weights: {scalarization.weights}")
            print(f"  Objectives: {sample.objs}")
            print(f"  OptGraph ID: {sample.optgraph_id}")
            
        return True, elite_batch, scalarization_batch
        
    except Exception as e:
        print(f"ERROR during warm-up initialization: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

if __name__ == "__main__":
    success, elite_batch, scalarization_batch = test_warmup_initialization()
    if success:
        print("\n✅ Warm-up initialization successful!")
    else:
        print("\n❌ Warm-up initialization failed!")
        sys.exit(1)
