#!/usr/bin/env python3
"""
MeltingPot Integration Demo for PGMORL

This script demonstrates the integration between PGMORL (multi-objective reinforcement learning)
and MeltingPot (multi-agent environments). It shows:

1. How to set up MeltingPot environments for PGMORL
2. CNN architecture configuration for visual observations
3. Multi-objective reward transformation
4. Training setup and execution

Run this script to test the integration without MeltingPot installed.
"""

import sys
import os

# Add PGMORL paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'morl'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'externals', 'pytorch-a2c-ppo-acktr-gail'))

from deep_sea_treasure_config import (
    apply_meltingpot_preset,
    get_meltingpot_base_kwargs,
    list_meltingpot_substrates,
    print_meltingpot_info,
    MELTINGPOT_CONFIG,
    NETWORK_CONFIG
)

def demo_configuration_system():
    """Demonstrate the configuration system for MeltingPot integration."""
    print("=" * 70)
    print("PGMORL-MeltingPot Integration Demo")
    print("=" * 70)
    print()
    
    # Show available substrates
    print("📋 Available Substrates:")
    substrates = list_meltingpot_substrates()
    for substrate in substrates:
        print(f"  🎮 {substrate['name']}")
        print(f"     Type: {substrate['type']}")
        print(f"     Agents: {substrate['num_agents']}")
        print(f"     Objectives: {substrate['objectives']}")
        print(f"     Description: {substrate['description']}")
        print()
    
    print("\n" + "=" * 70)
    print("Configuration Examples")
    print("=" * 70)
    
    # Demonstrate configuration for different substrates
    example_substrates = [
        'collaborative_cooking__asymmetric',
        'bach_or_stravinsky_in_the_matrix__repeated'
    ]
    
    for substrate_name in example_substrates:
        print(f"\n🔧 Configuration for: {substrate_name}")
        print("-" * 50)
        
        # Single agent configuration
        config = apply_meltingpot_preset(substrate_name, 'single_agent')
        print(f"Single Agent Mode:")
        print(f"  • Objectives: {config['obj_num']} ({config['objectives']})")
        print(f"  • CNN Preset: {config['cnn_preset']}")
        print(f"  • Training Steps: {config['num_env_steps']}")
        print(f"  • Episode Length: {config['max_episode_steps']}")
        
        # Multi-agent configuration
        config_ma = apply_meltingpot_preset(substrate_name, 'multi_agent_decentralized')
        print(f"Multi-Agent Decentralized Mode:")
        print(f"  • Agents: {config_ma['num_agents']}")
        print(f"  • Objectives per agent: {config_ma['obj_num']}")
        print(f"  • Substrate type: {config_ma['substrate_type']}")
        
        # Base kwargs for neural network
        base_kwargs = get_meltingpot_base_kwargs(substrate_name, 'single_agent')
        print(f"Neural Network Configuration:")
        print(f"  • CNN Layers: {len(base_kwargs['cnn_layers'])}")
        print(f"  • Hidden Size: {base_kwargs['hidden_size']}")
        print(f"  • Multi-objective: {base_kwargs['multi_objective']}")
        print(f"  • Recurrent: {base_kwargs['recurrent']}")


def demo_cnn_architectures():
    """Demonstrate CNN architecture configurations for MeltingPot."""
    print("\n" + "=" * 70)
    print("CNN Architecture Configurations")
    print("=" * 70)
    
    # Show MeltingPot-specific CNN presets
    meltingpot_presets = [
        'meltingpot_competitive',
        'meltingpot_collaborative', 
        'meltingpot_general'
    ]
    
    for preset in meltingpot_presets:
        if preset in NETWORK_CONFIG['cnn_presets']:
            config = NETWORK_CONFIG['cnn_presets'][preset]
            print(f"\n🧠 {preset.replace('_', ' ').title()}:")
            print(f"  • Layers: {len(config['cnn_layers'])}")
            print(f"  • Layer configs: {config['cnn_layers']}")
            print(f"  • Hidden size: {config['cnn_hidden_size']}")
            print(f"  • Final layers: {config['cnn_final_layers']}")
            
            # Explain the architecture
            if 'competitive' in preset:
                print(f"  • Optimized for: Competitive scenarios (Bach/Stravinsky, Prisoner's Dilemma)")
                print(f"  • Features: Sprite-aligned convolutions, efficient processing")
            elif 'collaborative' in preset:
                print(f"  • Optimized for: Collaborative scenarios (Cooking, Clean Up)")
                print(f"  • Features: Deep feature extraction, multi-scale processing")
            elif 'general' in preset:
                print(f"  • Optimized for: General MeltingPot environments")
                print(f"  • Features: Balanced architecture, good default choice")


def demo_reward_transformations():
    """Demonstrate multi-objective reward transformations."""
    print("\n" + "=" * 70)
    print("Multi-Objective Reward Transformations")
    print("=" * 70)
    
    reward_configs = MELTINGPOT_CONFIG['reward_transformations']
    
    for scenario_type, transformations in reward_configs.items():
        print(f"\n🎯 {scenario_type.title()} Scenarios:")
        for objective, description in transformations.items():
            print(f"  • {objective.replace('_', ' ').title()}: {description}")
    
    print(f"\n💡 Example Transformation for Collaborative Scenario:")
    print(f"  Single reward: 5.0 (agent collected item)")
    print(f"  All agent rewards: [5.0, 2.0, 1.0, 0.0]")
    print(f"  Transformed to multi-objective:")
    print(f"    • Individual performance: 5.0")
    print(f"    • Team cooperation: 2.0 (average)")
    print(f"    • Efficiency: 0.05 (reward/step at step 100)")


def demo_training_setup():
    """Demonstrate training setup for different scenarios."""
    print("\n" + "=" * 70)
    print("Training Setup Examples")
    print("=" * 70)
    
    scenarios = [
        {
            'substrate': 'collaborative_cooking__asymmetric',
            'mode': 'single_agent',
            'description': 'Single PGMORL agent learning cooperation'
        },
        {
            'substrate': 'bach_or_stravinsky_in_the_matrix__repeated',
            'mode': 'multi_agent_decentralized',
            'description': 'Two PGMORL agents in competitive scenario'
        }
    ]
    
    for scenario in scenarios:
        print(f"\n🚀 Training Setup: {scenario['description']}")
        print(f"   Substrate: {scenario['substrate']}")
        print(f"   Mode: {scenario['mode']}")
        print("-" * 50)
        
        # Get configuration
        config = apply_meltingpot_preset(scenario['substrate'], scenario['mode'])
        base_kwargs = get_meltingpot_base_kwargs(scenario['substrate'], scenario['mode'])
        
        # Show key training parameters
        print(f"Training Parameters:")
        print(f"  • Environment steps: {config['num_env_steps']:,}")
        print(f"  • Parallel processes: {config['num_processes']}")
        print(f"  • Learning rate: {config['lr']}")
        print(f"  • Episode steps: {config['max_episode_steps']}")
        
        print(f"Multi-Objective Setup:")
        print(f"  • Objectives: {config['obj_num']}")
        print(f"  • Weight candidates: {config['num_weight_candidates']}")
        print(f"  • Weight step: {config['delta_weight']}")
        print(f"  • Pareto buffer: {config['pbuffer_num']}")
        
        print(f"Network Architecture:")
        print(f"  • CNN preset: {config['cnn_preset']}")
        print(f"  • Hidden size: {base_kwargs['hidden_size']}")
        print(f"  • Recurrent: {base_kwargs['recurrent']}")
        print(f"  • Layer norm: {base_kwargs['layer_norm']}")


def demo_command_examples():
    """Show example commands for running training."""
    print("\n" + "=" * 70)
    print("Example Training Commands")
    print("=" * 70)
    
    examples = [
        {
            'command': 'python example_meltingpot_training.py --info',
            'description': 'Show all available substrates and configurations'
        },
        {
            'command': 'python example_meltingpot_training.py --list-substrates',
            'description': 'List available MeltingPot substrates'
        },
        {
            'command': 'python example_meltingpot_training.py --substrate collaborative_cooking__asymmetric --test-env',
            'description': 'Test environment creation for collaborative cooking'
        },
        {
            'command': 'python example_meltingpot_training.py --substrate collaborative_cooking__asymmetric --mode single_agent --dry-run',
            'description': 'Dry run setup for single agent training'
        },
        {
            'command': 'python example_meltingpot_training.py --substrate bach_or_stravinsky_in_the_matrix__repeated --mode multi_agent_decentralized --num-env-steps 50000',
            'description': 'Train two competing PGMORL agents'
        }
    ]
    
    for example in examples:
        print(f"\n📝 {example['description']}:")
        print(f"   {example['command']}")


def demo_integration_benefits():
    """Explain the benefits of PGMORL-MeltingPot integration."""
    print("\n" + "=" * 70)
    print("Integration Benefits")
    print("=" * 70)
    
    benefits = [
        {
            'title': 'Multi-Objective Multi-Agent Learning',
            'description': 'Combine multiple objectives (cooperation, competition, efficiency) in multi-agent settings'
        },
        {
            'title': 'Rich Visual Environments',
            'description': 'CNN architectures optimized for MeltingPot\'s visual observations'
        },
        {
            'title': 'Flexible Training Modes',
            'description': 'Single-agent, decentralized multi-agent, or centralized multi-agent training'
        },
        {
            'title': 'Diverse Scenarios',
            'description': 'From collaborative cooking to competitive matrix games'
        },
        {
            'title': 'Pareto-Optimal Solutions',
            'description': 'Find trade-offs between competing objectives in social dilemmas'
        },
        {
            'title': 'Configurable Rewards',
            'description': 'Transform single rewards into meaningful multi-objective signals'
        }
    ]
    
    for benefit in benefits:
        print(f"\n✨ {benefit['title']}:")
        print(f"   {benefit['description']}")


def main():
    """Run the complete demonstration."""
    try:
        demo_configuration_system()
        demo_cnn_architectures()
        demo_reward_transformations()
        demo_training_setup()
        demo_command_examples()
        demo_integration_benefits()
        
        print("\n" + "=" * 70)
        print("Integration Status")
        print("=" * 70)
        
        # Test if MeltingPot is available
        try:
            from environments.meltingpot_wrapper import MELTINGPOT_AVAILABLE
            if MELTINGPOT_AVAILABLE:
                print("✅ MeltingPot is available - ready to train!")
                print("   You can now run the example training scripts.")
            else:
                print("⚠️  MeltingPot not installed")
                print("   Install MeltingPot to enable full functionality:")
                print("   pip install dm-meltingpot")
        except ImportError:
            print("⚠️  MeltingPot wrapper not found")
            print("   The wrapper should be available at environments/meltingpot_wrapper.py")
        
        print("\n🎉 Demo completed! The integration framework is ready.")
        print("   Next steps:")
        print("   1. Install MeltingPot if not already installed")
        print("   2. Test environment creation with --test-env")
        print("   3. Run training with example_meltingpot_training.py")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
