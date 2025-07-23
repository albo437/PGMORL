#!/usr/bin/env python3
"""
Example training script for PGMORL with MeltingPot substrates.

This script demonstrates how to train multi-objective reinforcement learning agents
on MeltingPot multi-agent environments using the PGMORL framework.

Features:
- Integration between PGMORL and MeltingPot
- Multi-objective reward transformation
- CNN architecture optimization for visual observations
- Support for competitive and collaborative scenarios

Usage:
    python example_meltingpot_training.py --substrate collaborative_cooking__asymmetric
    python example_meltingpot_training.py --substrate bach_or_stravinsky_in_the_matrix__repeated --mode multi_agent_decentralized
"""

import sys
import os
import argparse
import numpy as np
import torch

# Add PGMORL paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'morl'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'externals', 'pytorch-a2c-ppo-acktr-gail'))

from deep_sea_treasure_config import (
    apply_meltingpot_preset,
    get_meltingpot_base_kwargs,
    create_meltingpot_environment,
    list_meltingpot_substrates,
    print_meltingpot_info,
    MELTINGPOT_CONFIG
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train PGMORL agents on MeltingPot substrates',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Environment selection
    parser.add_argument('--substrate', type=str, 
                      choices=list(MELTINGPOT_CONFIG['substrates'].keys()),
                      default='collaborative_cooking__asymmetric',
                      help='MeltingPot substrate to use')
    
    parser.add_argument('--mode', type=str,
                      choices=list(MELTINGPOT_CONFIG['training_modes'].keys()),
                      default='single_agent',
                      help='Training mode')
    
    # Training parameters
    parser.add_argument('--num-env-steps', type=int, default=100000,
                      help='Total environment steps')
    
    parser.add_argument('--num-processes', type=int, default=4,
                      help='Number of parallel environments')
    
    parser.add_argument('--lr', type=float, default=1e-4,
                      help='Learning rate')
    
    parser.add_argument('--save-dir', type=str, default='./meltingpot_results',
                      help='Directory to save results')
    
    # Multi-objective parameters
    parser.add_argument('--num-weight-candidates', type=int, default=6,
                      help='Number of weight candidates for scalarization')
    
    parser.add_argument('--delta-weight', type=float, default=0.1,
                      help='Weight step size for Pareto front approximation')
    
    # Experimental flags
    parser.add_argument('--test-env', action='store_true',
                      help='Test environment creation and exit')
    
    parser.add_argument('--list-substrates', action='store_true',
                      help='List available substrates and exit')
    
    parser.add_argument('--info', action='store_true',
                      help='Print MeltingPot configuration info and exit')
    
    parser.add_argument('--dry-run', action='store_true',
                      help='Set up training but do not run')
    
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug output')
    
    return parser.parse_args()


def test_environment(substrate_name, mode, debug=False):
    """Test environment creation and basic functionality."""
    print(f"Testing environment: {substrate_name} with mode: {mode}")
    
    try:
        # Create environment
        env = create_meltingpot_environment(substrate_name, mode)
        if env is None:
            print("❌ Environment creation failed - MeltingPot not available")
            return False
        
        print(f"✓ Environment created successfully")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        
        # Test reset
        obs = env.reset()
        print(f"✓ Reset successful")
        if isinstance(obs, dict):
            print(f"  Observation keys: {list(obs.keys())}")
            for key, value in obs.items():
                if hasattr(value, 'shape'):
                    print(f"    {key}: {value.shape}")
        else:
            print(f"  Observation shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
        
        # Test step
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"✓ Step successful")
        print(f"  Reward shape: {reward.shape if hasattr(reward, 'shape') else type(reward)}")
        print(f"  Reward: {reward}")
        print(f"  Done: {done}")
        print(f"  Info keys: {list(info.keys()) if isinstance(info, dict) else type(info)}")
        
        # Test a few more steps
        for i in range(3):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            print(f"  Step {i+2}: reward={reward}, done={done}")
            if done:
                print("  Episode ended, resetting...")
                obs = env.reset()
                break
        
        env.close()
        print("✓ Environment test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        return False


def setup_training(substrate_name, mode, **overrides):
    """Set up training configuration and environment."""
    print(f"Setting up training for {substrate_name} in {mode} mode...")
    
    # Get configuration
    config = apply_meltingpot_preset(substrate_name, mode, **overrides)
    base_kwargs = get_meltingpot_base_kwargs(substrate_name, mode, **overrides)
    
    print("Configuration:")
    print(f"  Substrate: {config['substrate_name']}")
    print(f"  Training mode: {config['training_mode']}")
    print(f"  Objectives: {config['obj_num']} ({config['objectives']})")
    print(f"  CNN preset: {config['cnn_preset']}")
    print(f"  Max episode steps: {config['max_episode_steps']}")
    print(f"  Environment steps: {config['num_env_steps']}")
    print(f"  Parallel processes: {config['num_processes']}")
    
    # Create environment for testing
    env = create_meltingpot_environment(substrate_name, mode)
    if env is None:
        raise RuntimeError("Failed to create environment - MeltingPot not available")
    
    print("Environment info:")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    
    # Test environment
    obs = env.reset()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(f"  Sample reward shape: {reward.shape}")
    print(f"  Sample reward: {reward}")
    
    env.close()
    
    return config, base_kwargs


def run_training(config, base_kwargs, args):
    """Run the actual PGMORL training."""
    print("Starting PGMORL training...")
    
    try:
        # Import PGMORL training components
        from morl.warm_up import warm_up
        from morl.arguments import get_args
        
        # Convert config to PGMORL args format
        pgmorl_args = []
        
        # Basic environment args
        pgmorl_args.extend(['--env-name', f"MeltingPot-{config['substrate_name']}-{config['training_mode']}"])
        pgmorl_args.extend(['--obj-num', str(config['obj_num'])])
        pgmorl_args.extend(['--save-dir', args.save_dir])
        
        # Training parameters
        pgmorl_args.extend(['--num-env-steps', str(config['num_env_steps'])])
        pgmorl_args.extend(['--num-processes', str(config['num_processes'])])
        pgmorl_args.extend(['--lr', str(config['lr'])])
        
        # Multi-objective parameters
        pgmorl_args.extend(['--min-weight', str(config['min_weight'])])
        pgmorl_args.extend(['--max-weight', str(config['max_weight'])])
        pgmorl_args.extend(['--delta-weight', str(config['delta_weight'])])
        pgmorl_args.extend(['--num-weight-candidates', str(config['num_weight_candidates'])])
        pgmorl_args.extend(['--pbuffer-num', str(config['pbuffer_num'])])
        
        # Flags
        if config.get('obj_rms', True):
            pgmorl_args.append('--obj-rms')
        if config.get('ob_rms', True):
            pgmorl_args.append('--ob-rms')
        if config.get('raw', True):
            pgmorl_args.append('--raw')
        
        print(f"PGMORL arguments: {' '.join(pgmorl_args)}")
        
        # Parse arguments
        old_argv = sys.argv
        sys.argv = ['meltingpot_training.py'] + pgmorl_args
        
        try:
            pgmorl_args_obj = get_args()
            
            # Add base_kwargs for CNN architecture
            pgmorl_args_obj.base_kwargs = base_kwargs
            
            print("Starting warm_up training...")
            warm_up(pgmorl_args_obj)
            
        finally:
            sys.argv = old_argv
        
        print("✓ Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return False
    
    return True


def main():
    """Main execution function."""
    args = parse_args()
    
    # Handle info/listing commands
    if args.info:
        print_meltingpot_info()
        return
    
    if args.list_substrates:
        print("Available MeltingPot substrates:")
        for substrate in list_meltingpot_substrates():
            print(f"  • {substrate['name']} ({substrate['type']}, {substrate['num_agents']} agents)")
            print(f"    Objectives: {substrate['objectives']}")
            print(f"    {substrate['description']}")
            print()
        return
    
    # Test environment creation
    if args.test_env:
        success = test_environment(args.substrate, args.mode, args.debug)
        sys.exit(0 if success else 1)
    
    # Setup training
    try:
        config, base_kwargs = setup_training(
            args.substrate, 
            args.mode,
            num_env_steps=args.num_env_steps,
            num_processes=args.num_processes,
            lr=args.lr,
            num_weight_candidates=args.num_weight_candidates,
            delta_weight=args.delta_weight
        )
        
        if args.dry_run:
            print("✓ Dry run completed - configuration is valid")
            return
        
        # Run training
        success = run_training(config, base_kwargs, args)
        
        if success:
            print(f"\n🎉 Training completed successfully!")
            print(f"Results saved to: {args.save_dir}")
            print(f"Substrate: {args.substrate}")
            print(f"Mode: {args.mode}")
        else:
            print(f"\n❌ Training failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
