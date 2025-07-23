#!/usr/bin/env python3
"""
Example Usage of Deep Sea Treasure Configuration
===============================================

This script demonstrates how to use the configuration system for different
training scenarios. Run this script to see examples of different configurations.
"""

import sys
import os

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from deep_sea_treasure_config import *

def demo_configurations():
    """Demonstrate different configuration presets."""
    
    print("=" * 80)
    print("DEEP SEA TREASURE CONFIGURATION DEMO")
    print("=" * 80)
    
    # Default configuration
    print("\n1. DEFAULT CONFIGURATION:")
    print("-" * 40)
    print_config_summary()
    
    # Fast test configuration
    print("\n2. FAST TEST CONFIGURATION:")
    print("-" * 40)
    apply_fast_test_config()
    print_config_summary()
    
    # Reset to defaults and show full training
    print("\n3. FULL TRAINING CONFIGURATION:")
    print("-" * 40)
    # Reset
    TRAINING_CONFIG['num_env_steps'] = 50000
    TRAINING_CONFIG['warmup_iter'] = 20
    TRAINING_CONFIG['update_iter'] = 10
    
    apply_full_training_config()
    print_config_summary()
    
    # Large network configuration
    print("\n4. LARGE NETWORK CONFIGURATION:")
    print("-" * 40)
    apply_large_network_config()
    print_config_summary()
    
    print("\n" + "=" * 80)
    print("CONFIGURATION EXAMPLES COMPLETED")
    print("=" * 80)
    print("\nTo use these configurations in your training:")
    print("1. Edit deep_sea_treasure_config.py directly")
    print("2. Or uncomment the preset functions in main()")
    print("3. Or call the apply_*_config() functions before training")

def show_command_line_args():
    """Show what command line arguments would be generated."""
    print("\n" + "=" * 80)
    print("GENERATED COMMAND LINE ARGUMENTS")
    print("=" * 80)
    
    args = get_all_args()
    print("sys.argv would be set to:")
    print("[")
    for i, arg in enumerate(args):
        if i == 0:
            print(f"    '{arg}',")
        elif arg.startswith('--'):
            print(f"    '{arg}',")
        else:
            print(f"    '{arg}',")
    print("]")
    
    print(f"\nTotal arguments: {len(args)}")

def custom_configuration_example():
    """Show how to create custom configurations."""
    print("\n" + "=" * 80)
    print("CUSTOM CONFIGURATION EXAMPLE")
    print("=" * 80)
    
    # Example: High-performance configuration
    print("Creating a high-performance configuration...")
    
    # Modify global configurations
    global TRAINING_CONFIG, NETWORK_CONFIG, PPO_CONFIG
    
    # More training steps
    TRAINING_CONFIG['num_env_steps'] = 200000
    TRAINING_CONFIG['num_processes'] = 8
    
    # Larger network
    NETWORK_CONFIG['hidden_size'] = 256
    NETWORK_CONFIG['layernorm'] = True
    
    # More aggressive PPO settings
    PPO_CONFIG['ppo_epoch'] = 15
    PPO_CONFIG['num_mini_batch'] = 32
    PPO_CONFIG['lr'] = 1e-4
    
    print("High-performance configuration applied!")
    print_config_summary()

if __name__ == "__main__":
    demo_configurations()
    show_command_line_args()
    custom_configuration_example()
    
    print("\n" + "=" * 80)
    print("To run training with current configuration:")
    print("python example_deep_sea_treasure.py")
    print("=" * 80)
