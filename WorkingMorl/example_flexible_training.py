"""
PGMORL Training with Flexible Neural Network Architectures
==========================================================

This example shows how to use the flexible neural network configuration system
with PGMORL, supporting both MLP (for 1D observations) and CNN (for image observations).

Example Usage:
    # For 1D environments (like Deep Sea Treasure)
    python example_flexible_training.py --env MO-DeepSeaTreasure-v0
    
    # For image environments (like Atari)
    python example_flexible_training.py --env BreakoutNoFrameskip-v4 --cnn-preset atari
    
    # With custom CNN
    python example_flexible_training.py --env MyImageEnv-v0 --custom-cnn
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

# Import configuration system
from deep_sea_treasure_config import *

# Import and register environments
try:
    import environments.deep_sea_treasure
    print("Deep Sea Treasure environment imported")
except ImportError:
    print("Deep Sea Treasure environment not available")

# Import MORL components
from arguments import get_args
from morl import run as morl_train

def setup_environment_specific_config(env_name):
    """
    Setup network configuration based on environment type.
    
    Args:
        env_name: Name of the environment
    """
    global ENV_NAME, EXPECTED_OBS_SHAPE, NETWORK_CONFIG
    
    ENV_NAME = env_name
    
    # Create temporary environment to inspect observation space
    try:
        temp_env = gym.make(env_name)
        obs_shape = temp_env.observation_space.shape
        temp_env.close()
        
        print(f"Environment: {env_name}")
        print(f"Observation shape: {obs_shape}")
        
        EXPECTED_OBS_SHAPE = obs_shape
        
        # Auto-configure network based on observation shape
        if len(obs_shape) == 3:
            print("Detected image observations - using CNN architecture")
            
            # Auto-select appropriate CNN preset based on image size
            height, width = obs_shape[1], obs_shape[2]
            
            if height == 84 and width == 84:
                print("Using Atari CNN preset (84x84 images)")
                apply_atari_cnn_config()
            elif height >= 96 and width >= 96:
                print("Using Deep Nature CNN preset (large images)")
                apply_deep_nature_cnn_config()
            else:
                print("Using Lightweight CNN preset (small/medium images)")
                apply_lightweight_cnn_config()
                
        elif len(obs_shape) == 1:
            print("Detected vector observations - using MLP architecture")
            NETWORK_CONFIG['use_custom_cnn'] = False
            
            # Adjust MLP size based on observation size
            if obs_shape[0] > 100:
                print("Large observation space - using larger MLP")
                NETWORK_CONFIG['hidden_size'] = 128
            elif obs_shape[0] > 20:
                print("Medium observation space - using standard MLP")
                NETWORK_CONFIG['hidden_size'] = 64
            else:
                print("Small observation space - using compact MLP")
                NETWORK_CONFIG['hidden_size'] = 32
        else:
            print(f"Unknown observation shape {obs_shape} - using default MLP")
            NETWORK_CONFIG['use_custom_cnn'] = False
            
    except Exception as e:
        print(f"Could not inspect environment {env_name}: {e}")
        print("Using default configuration")

def convert_to_classic_format(results_dir):
    """Convert objectives to classic format (for Deep Sea Treasure)."""
    objs_file = os.path.join(results_dir, 'final', 'objs.txt')
    
    if not os.path.exists(objs_file):
        print(f"Warning: {objs_file} not found, skipping conversion")
        return
    
    # Only convert for Deep Sea Treasure environment
    if 'DeepSeaTreasure' not in ENV_NAME:
        print("Not Deep Sea Treasure environment - skipping format conversion")
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
    
    print(f"Converting {len(objectives)} objectives to classic Deep Sea Treasure format...")
    
    # Convert to classic format: (treasure_value, -steps)
    classic_objectives = []
    for obj1, obj2 in objectives:
        # Convert efficiency back to steps
        if obj2 > 0:
            estimated_steps = max(1, int(round((10.0 / obj2) - 1)))
        else:
            estimated_steps = 10
        
        # Determine treasure value based on obj1
        if obj1 >= 20.0:
            treasure_value = 20.0
        elif obj1 >= 8.0:
            treasure_value = 8.0
        elif obj1 >= 3.0:
            treasure_value = 3.0
        elif obj1 >= 1.0:
            treasure_value = 1.0
        else:
            treasure_value = 0.0
        
        classic_objectives.append([treasure_value, -estimated_steps])
    
    # Save converted objectives
    with open(objs_file, 'w') as f:
        for treasure, neg_steps in classic_objectives:
            f.write(f'{treasure:.6f},{neg_steps:.6f}\n')
    
    print(f"Converted to classic format: (treasure_value, -steps)")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='PGMORL Training with Flexible Architectures')
    parser.add_argument('--env', default='MO-DeepSeaTreasure-v0', 
                       help='Environment name')
    parser.add_argument('--cnn-preset', choices=['atari', 'meltingpot', 'deep_nature', 'lightweight'],
                       help='Use CNN preset')
    parser.add_argument('--custom-cnn', action='store_true',
                       help='Use custom CNN configuration')
    parser.add_argument('--hidden-size', type=int, default=None,
                       help='MLP hidden size override')
    parser.add_argument('--fast-test', action='store_true',
                       help='Use fast test configuration')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PGMORL TRAINING - FLEXIBLE ARCHITECTURES")
    print("=" * 60)
    
    # Setup environment-specific configuration
    setup_environment_specific_config(args.env)
    
    # Apply user-specified configurations
    if args.cnn_preset:
        print(f"Applying CNN preset: {args.cnn_preset}")
        apply_cnn_preset(args.cnn_preset)
    
    if args.custom_cnn:
        print("Using custom CNN configuration")
        create_custom_cnn_config([
            [32, 8, 4, 0],
            [64, 4, 2, 0], 
            [128, 3, 1, 1]
        ], cnn_hidden_size=512, cnn_final_layers=[512, 256])
    
    if args.hidden_size:
        print(f"Overriding hidden size: {args.hidden_size}")
        NETWORK_CONFIG['hidden_size'] = args.hidden_size
    
    if args.fast_test:
        print("Applying fast test configuration")
        apply_fast_test_config()
    
    # Print final configuration
    print_config_summary()
    
    print("Starting MORL training...")
    print()
    
    # Get arguments from configuration
    sys.argv = get_all_args()
    
    # Parse arguments and start training
    morl_args = get_args()
    
    # Set device
    morl_args.cuda = torch.cuda.is_available()
    morl_args.device = torch.device("cuda" if morl_args.cuda else "cpu")
    
    print(f"Device: {morl_args.device}")
    print(f"Using {morl_args.num_processes} parallel environments")
    
    # Add network configuration to args (this would need modification in warm_up.py)
    if hasattr(morl_args, 'base_kwargs'):
        morl_args.base_kwargs = get_base_kwargs()
    else:
        print("Warning: base_kwargs not supported - network config may not be applied")
    
    print()
    
    # Start training
    try:
        morl_train(morl_args)
        
        # Post-process results if needed
        convert_to_classic_format(SAVE_DIR)
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Results saved to: {SAVE_DIR}")
        print("Check the logs for training progress and final results.")
        
        if 'DeepSeaTreasure' in ENV_NAME:
            print("\nExpected Pareto-optimal solutions:")
            for point in EXPECTED_PARETO_FRONT:
                print(f"  Treasure: {point['treasure']}, Steps: {point['steps']} ({point['description']})")
        
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
