"""
Deep Sea Treasure Training Configuration
========================================

This file contains all the configurable parameters for PGMORL training on the Deep Sea Treasure environment.
Modify these parameters to experiment with different training configurations without changing the main code.
"""

# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================
ENV_NAME = 'MO-DeepSeaTreasure-v0'
OBJECTIVES_NUM = 2
SAVE_DIR = './deep_sea_treasure_results'

# =============================================================================
# NEURAL NETWORK ARCHITECTURE
# =============================================================================
NETWORK_CONFIG = {
    # MLP Configuration (for 1D observations)
    'hidden_size': 64,           # Size of hidden layers (default: 64)
    'num_layers': 2,             # Number of hidden layers (fixed in current implementation)
    'layernorm': False,          # Use layer normalization
    'activation': 'tanh',        # Activation function (tanh is hardcoded currently)
    
    # CNN Configuration (for 2D/3D observations like images)
    'use_custom_cnn': False,     # Whether to use custom CNN architecture
    'cnn_layers': [              # Custom CNN layers: [channels, kernel_size, stride, padding]
        [32, 8, 4, 0],           # First conv layer: 32 filters, 8x8 kernel, stride 4
        [64, 4, 2, 0],           # Second conv layer: 64 filters, 4x4 kernel, stride 2  
        [64, 3, 1, 0],           # Third conv layer: 64 filters, 3x3 kernel, stride 1
    ],
    'cnn_activation': 'relu',    # CNN activation function
    'cnn_hidden_size': 512,      # Size of fully connected layer after CNN
    'cnn_final_layers': [512],   # Additional FC layers after CNN extraction
    
    # Advanced CNN Configurations (inspired by MeltingPot/Atari)
    'cnn_presets': {
        'atari': {
            'cnn_layers': [[32, 8, 4, 0], [64, 4, 2, 0], [64, 3, 1, 0]],
            'cnn_hidden_size': 512,
            'cnn_final_layers': [512]
        },
        'meltingpot': {
            'cnn_layers': [[16, 8, 8, 0], [128, [1, 1], 1, 0]],  # Sprite-aligned
            'cnn_hidden_size': 256,
            'cnn_final_layers': [256, 256]
        },
        'deep_nature': {
            'cnn_layers': [[32, 8, 4, 0], [64, 4, 2, 0], [64, 3, 1, 0], [128, 3, 1, 1]],
            'cnn_hidden_size': 1024,
            'cnn_final_layers': [1024, 512]
        },
        'lightweight': {
            'cnn_layers': [[16, 8, 4, 0], [32, 4, 2, 0]],
            'cnn_hidden_size': 256,
            'cnn_final_layers': [256]
        },
        'meltingpot_competitive': {
            # CNN optimized for competitive MeltingPot scenarios (Bach/Stravinsky, Prisoner's Dilemma)
            'cnn_layers': [[16, 8, 8], [128, 11, 1]],  # Matches RLLib example for sprite alignment
            'cnn_hidden_size': 512,
            'cnn_final_layers': [512, 256]
        },
        'meltingpot_collaborative': {
            # CNN optimized for collaborative MeltingPot scenarios (Collaborative Cooking, Clean Up)
            'cnn_layers': [[32, 8, 4, 0], [64, 4, 2, 0], [128, 3, 1, 0]],
            'cnn_hidden_size': 1024,
            'cnn_final_layers': [1024, 512, 256]
        },
        'meltingpot_general': {
            # General purpose CNN for any MeltingPot substrate
            'cnn_layers': [[16, 8, 8], [128, 11, 1]],  # Standard MeltingPot configuration
            'cnn_hidden_size': 256,
            'cnn_final_layers': [256]
        }
    }
}

# =============================================================================
# MELTINGPOT INTEGRATION CONFIGURATION
# =============================================================================
MELTINGPOT_CONFIG = {
    # Popular substrates for multi-objective learning
    'substrates': {
        'collaborative_cooking__asymmetric': {
            'type': 'collaborative',
            'num_agents': 4,
            'objectives': ['individual_reward', 'team_cooperation', 'efficiency'],
            'num_objectives': 3,
            'cnn_preset': 'meltingpot_collaborative',
            'episode_steps': 1000,
            'description': 'Asymmetric collaborative cooking scenario'
        },
        'bach_or_stravinsky_in_the_matrix__repeated': {
            'type': 'competitive', 
            'num_agents': 2,
            'objectives': ['individual_performance', 'relative_performance'],
            'num_objectives': 2,
            'cnn_preset': 'meltingpot_competitive',
            'episode_steps': 1000,
            'description': 'Bach or Stravinsky repeated game matrix'
        },
        'prisoners_dilemma_in_the_matrix__repeated': {
            'type': 'competitive',
            'num_agents': 2, 
            'objectives': ['individual_reward', 'cooperation_level'],
            'num_objectives': 2,
            'cnn_preset': 'meltingpot_competitive',
            'episode_steps': 1000,
            'description': 'Repeated Prisoner\'s Dilemma'
        },
        'clean_up': {
            'type': 'mixed',
            'num_agents': 4,
            'objectives': ['individual_reward', 'collective_cleanup', 'sustainability'],
            'num_objectives': 3,
            'cnn_preset': 'meltingpot_collaborative',
            'episode_steps': 1000,
            'description': 'Clean up the commons dilemma'
        }
    },
    
    # Training modes
    'training_modes': {
        'single_agent': {
            'description': 'Train one PGMORL agent against random/baseline agents',
            'wrapper': 'MeltingPotSingleAgentWrapper',
            'recommended_for': ['initial_testing', 'baseline_comparison']
        },
        'multi_agent_decentralized': {
            'description': 'Train multiple PGMORL agents independently',
            'wrapper': 'MeltingPotMultiAgentWrapper',
            'training_mode': 'decentralized',
            'recommended_for': ['competitive_scenarios', 'independent_learning']
        },
        'multi_agent_centralized': {
            'description': 'Train multiple PGMORL agents with shared observations',
            'wrapper': 'MeltingPotMultiAgentWrapper', 
            'training_mode': 'centralized',
            'recommended_for': ['collaborative_scenarios', 'coordination_learning']
        }
    },
    
    # Multi-objective reward transformations
    'reward_transformations': {
        'collaborative': {
            'individual_task_performance': 'Direct agent reward',
            'team_cooperation': 'Average of all agent rewards',
            'efficiency': 'Reward per timestep ratio'
        },
        'competitive': {
            'individual_performance': 'Direct agent reward',
            'relative_performance': 'Agent reward minus average of others',
            'consistency': 'Negative variance of recent rewards'
        },
        'mixed': {
            'individual_reward': 'Direct agent reward',
            'collective_performance': 'Sum of all agent rewards',
            'social_welfare': 'Min-max fairness metric'
        }
    }
}

# =============================================================================
# PPO ALGORITHM PARAMETERS
# =============================================================================
PPO_CONFIG = {
    'num_steps': 80,             # Number of steps per rollout
    'ppo_epoch': 10,             # Number of PPO epochs per update
    'num_mini_batch': 20,        # Number of mini-batches for PPO
    'lr': 3e-4,                  # Learning rate
    'gamma': 0.99,               # Discount factor
    'gae_lambda': 0.95,          # GAE lambda parameter
    'clip_param': 0.2,           # PPO clipping parameter (default)
    'value_loss_coef': 0.5,      # Value loss coefficient (default)
    'entropy_coef': 0.01,        # Entropy loss coefficient (default)
    'max_grad_norm': 0.5,        # Gradient clipping (default)
}

# =============================================================================
# TRAINING SCHEDULE
# =============================================================================
TRAINING_CONFIG = {
    'num_env_steps': 25000,      # Total environment steps
    'num_processes': 4,          # Number of parallel environments
    'warmup_iter': 20,           # Warmup iterations
    'update_iter': 10,           # Update iterations
    'eval_num': 1,               # Number of evaluations
}

# =============================================================================
# MULTI-OBJECTIVE CONFIGURATION
# =============================================================================
MO_CONFIG = {
    'min_weight': 0.0,           # Minimum weight for scalarization
    'max_weight': 1.0,           # Maximum weight for scalarization
    'delta_weight': 0.02,        # Weight step size
    'selection_method': 'prediction-guided',  # PGMORL selection method
    'num_weight_candidates': 8,  # Number of weight candidates
    'num_tasks': 8,              # Number of tasks
    'pbuffer_num': 20,           # Pareto buffer number
    'pbuffer_size': 1,           # Pareto buffer size
}

# =============================================================================
# NORMALIZATION SETTINGS
# =============================================================================
NORMALIZATION_CONFIG = {
    'obj_rms': False,            # Use objective normalization - DISABLED for debugging
    'ob_rms': True,              # Use observation normalization
    'ret_rms': False,            # Use return normalization (default)
}

# =============================================================================
# EXPERIMENTAL SETTINGS
# =============================================================================
EXPERIMENTAL_CONFIG = {
    'raw': True,                 # Use raw objectives (no discounting) - CRITICAL for Deep Sea Treasure
    'no_save_models': True,      # Skip saving .pt and .pkl files to save disk space
    'cuda': True,                # Use CUDA if available (auto-detected)
    'seed': None,                # Random seed (None for random)
}

# =============================================================================
# LOGGING AND OUTPUT
# =============================================================================
OUTPUT_CONFIG = {
    'log_interval': 10,          # Logging interval (default)
    'save_interval': 0,          # Model saving interval (0 = disabled)
    'verbose': True,             # Verbose output
}

# =============================================================================
# EXPECTED RESULTS (for reference)
# =============================================================================
EXPECTED_PARETO_FRONT = [
    {'treasure': 1.0, 'steps': 1, 'description': 'Quick & Small'},
    {'treasure': 8.0, 'steps': 6, 'description': 'Medium Effort & Reward'},
    {'treasure': 20.0, 'steps': 14, 'description': 'High Effort & High Reward'},  # Updated to match actual DST
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def apply_meltingpot_preset(substrate_name, training_mode='single_agent', **overrides):
    """
    Apply MeltingPot-specific configuration preset.
    
    Args:
        substrate_name: Name of the MeltingPot substrate
        training_mode: Training mode ('single_agent', 'multi_agent_decentralized', 'multi_agent_centralized')
        **overrides: Override specific configuration values
        
    Returns:
        dict: Complete configuration for MeltingPot training
    """
    if substrate_name not in MELTINGPOT_CONFIG['substrates']:
        available = list(MELTINGPOT_CONFIG['substrates'].keys())
        raise ValueError(f"Unknown substrate: {substrate_name}. Available: {available}")
    
    substrate_config = MELTINGPOT_CONFIG['substrates'][substrate_name]
    training_config = MELTINGPOT_CONFIG['training_modes'][training_mode]
    
    # Build configuration
    config = {
        # Environment configuration
        'env_name': f'MeltingPot-{substrate_name}-{training_mode}',
        'substrate_name': substrate_name,
        'training_mode': training_mode,
        'wrapper_class': training_config['wrapper'],
        
        # Objectives
        'obj_num': substrate_config['num_objectives'],
        'objectives': substrate_config['objectives'],
        
        # CNN Architecture
        'cnn_preset': substrate_config['cnn_preset'],
        'cnn_config': NETWORK_CONFIG['cnn_presets'][substrate_config['cnn_preset']],
        
        # Episode configuration
        'max_episode_steps': substrate_config['episode_steps'],
        
        # Multi-agent specific
        'num_agents': substrate_config['num_agents'],
        'substrate_type': substrate_config['type'],
        
        # Training parameters (can be overridden)
        'num_env_steps': 100000,  # More steps for complex multi-agent scenarios
        'num_processes': 4,       # Parallel environments
        'num_steps': 128,         # Longer rollouts for multi-agent
        'ppo_epoch': 4,           # Fewer epochs to avoid overfitting
        'lr': 1e-4,               # Lower learning rate for stability
        
        # Multi-objective parameters
        'min_weight': 0.0,
        'max_weight': 1.0,
        'delta_weight': 0.1,      # Coarser weight grid for complex objectives
        'num_weight_candidates': 6,
        'pbuffer_num': 15,
        'pbuffer_size': 1,
        
        # Normalization
        'obj_rms': True,
        'ob_rms': True,
        'ret_rms': False,
        
        # Experimental
        'raw': True,              # Keep raw objectives for interpretability
        'cuda': True,
    }
    
    # Apply overrides
    config.update(overrides)
    
    return config


def get_meltingpot_base_kwargs(substrate_name, training_mode='single_agent', **overrides):
    """
    Get base_kwargs for MeltingPot environment with CNN architecture.
    
    Args:
        substrate_name: Name of the MeltingPot substrate  
        training_mode: Training mode
        **overrides: Override specific values
        
    Returns:
        dict: base_kwargs compatible with PGMORL's Policy class
    """
    config = apply_meltingpot_preset(substrate_name, training_mode, **overrides)
    
    # Get CNN configuration
    cnn_config = config['cnn_config']
    
    base_kwargs = {
        'recurrent': True,  # MeltingPot benefits from memory
        'hidden_size': cnn_config['cnn_hidden_size'],
        
        # CNN architecture
        'use_cnn': True,
        'cnn_layers': cnn_config['cnn_layers'],
        'cnn_final_layers': cnn_config['cnn_final_layers'],
        
        # Multi-objective support
        'multi_objective': True,
        'obj_num': config['obj_num'],
        
        # Activation functions
        'cnn_activation': 'relu',
        'final_activation': 'relu',
        
        # Normalization
        'layer_norm': True,  # Helps with multi-agent training stability
        
        # Environment specific
        'substrate_name': substrate_name,
        'training_mode': training_mode,
    }
    
    return base_kwargs


def create_meltingpot_environment(substrate_name, training_mode='single_agent', **env_kwargs):
    """
    Create a MeltingPot environment wrapper for PGMORL.
    
    Args:
        substrate_name: Name of the MeltingPot substrate
        training_mode: Training mode
        **env_kwargs: Additional environment arguments
        
    Returns:
        Gym environment or None if MeltingPot not available
    """
    try:
        from environments.meltingpot_wrapper import (
            MeltingPotSingleAgentWrapper, 
            MeltingPotMultiAgentWrapper
        )
        
        config = apply_meltingpot_preset(substrate_name, training_mode)
        
        if training_mode == 'single_agent':
            env = MeltingPotSingleAgentWrapper(
                substrate_name=substrate_name,
                num_objectives=config['obj_num'],
                **env_kwargs
            )
        else:
            training_mode_key = training_mode.replace('multi_agent_', '')
            env = MeltingPotMultiAgentWrapper(
                substrate_name=substrate_name,
                num_objectives=config['obj_num'],
                training_mode=training_mode_key,
                **env_kwargs
            )
        
        return env
        
    except ImportError as e:
        print(f"MeltingPot not available: {e}")
        print("Please install MeltingPot to use these environments.")
        return None


def list_meltingpot_substrates():
    """List all configured MeltingPot substrates."""
    substrates = []
    for name, config in MELTINGPOT_CONFIG['substrates'].items():
        substrates.append({
            'name': name,
            'type': config['type'],
            'num_agents': config['num_agents'],
            'objectives': config['objectives'],
            'description': config['description']
        })
    return substrates


def print_meltingpot_info():
    """Print information about available MeltingPot configurations."""
    print("=== MeltingPot Integration for PGMORL ===")
    print()
    
    print("Available Substrates:")
    for substrate in list_meltingpot_substrates():
        print(f"  • {substrate['name']}")
        print(f"    Type: {substrate['type']}")
        print(f"    Agents: {substrate['num_agents']}")
        print(f"    Objectives: {substrate['objectives']}")
        print(f"    Description: {substrate['description']}")
        print()
    
    print("Available Training Modes:")
    for mode, config in MELTINGPOT_CONFIG['training_modes'].items():
        print(f"  • {mode}: {config['description']}")
    print()
    
    print("CNN Presets for MeltingPot:")
    for preset in ['meltingpot_competitive', 'meltingpot_collaborative', 'meltingpot_general']:
        if preset in NETWORK_CONFIG['cnn_presets']:
            cnn_config = NETWORK_CONFIG['cnn_presets'][preset]
            print(f"  • {preset}: {len(cnn_config['cnn_layers'])} layers, {cnn_config['cnn_hidden_size']} hidden")


def get_all_args():
    """
    Convert all configuration parameters into a list suitable for sys.argv.
    
    Returns:
        list: Command line arguments for PGMORL training
    """
    args = ['example_deep_sea_treasure.py']
    
    # Environment
    args.extend(['--env-name', ENV_NAME])
    args.extend(['--obj-num', str(OBJECTIVES_NUM)])
    args.extend(['--save-dir', SAVE_DIR])
    
    # PPO Configuration
    args.extend(['--num-steps', str(PPO_CONFIG['num_steps'])])
    args.extend(['--ppo-epoch', str(PPO_CONFIG['ppo_epoch'])])
    args.extend(['--num-mini-batch', str(PPO_CONFIG['num_mini_batch'])])
    args.extend(['--lr', str(PPO_CONFIG['lr'])])
    args.extend(['--gamma', str(PPO_CONFIG['gamma'])])
    args.extend(['--gae-lambda', str(PPO_CONFIG['gae_lambda'])])
    
    # Training Schedule
    args.extend(['--num-env-steps', str(TRAINING_CONFIG['num_env_steps'])])
    args.extend(['--num-processes', str(TRAINING_CONFIG['num_processes'])])
    args.extend(['--warmup-iter', str(TRAINING_CONFIG['warmup_iter'])])
    args.extend(['--update-iter', str(TRAINING_CONFIG['update_iter'])])
    args.extend(['--eval-num', str(TRAINING_CONFIG['eval_num'])])
    
    # Multi-objective Configuration
    args.extend(['--min-weight', str(MO_CONFIG['min_weight'])])
    args.extend(['--max-weight', str(MO_CONFIG['max_weight'])])
    args.extend(['--delta-weight', str(MO_CONFIG['delta_weight'])])
    args.extend(['--selection-method', MO_CONFIG['selection_method']])
    args.extend(['--num-weight-candidates', str(MO_CONFIG['num_weight_candidates'])])
    args.extend(['--num-tasks', str(MO_CONFIG['num_tasks'])])
    args.extend(['--pbuffer-num', str(MO_CONFIG['pbuffer_num'])])
    args.extend(['--pbuffer-size', str(MO_CONFIG['pbuffer_size'])])
    
    # Normalization (flags)
    if NORMALIZATION_CONFIG['obj_rms']:
        args.append('--obj-rms')
    if NORMALIZATION_CONFIG['ob_rms']:
        args.append('--ob-rms')
    if NORMALIZATION_CONFIG['ret_rms']:
        args.append('--ret-rms')
    
    # Network Configuration (flags)
    if NETWORK_CONFIG['layernorm']:
        args.append('--layernorm')
    
    # Experimental Settings (flags)
    if EXPERIMENTAL_CONFIG['raw']:
        args.append('--raw')
    if EXPERIMENTAL_CONFIG['no_save_models']:
        args.append('--no-save-models')
    if EXPERIMENTAL_CONFIG['seed'] is not None:
        args.extend(['--seed', str(EXPERIMENTAL_CONFIG['seed'])])
    
    return args

def get_base_kwargs():
    """
    Get base_kwargs for neural network configuration.
    
    Returns:
        dict: Base kwargs for Policy initialization
    """
    base_kwargs = {
        'layernorm': NETWORK_CONFIG['layernorm'],
    }
    
    # Add CNN configuration if using flexible CNN
    if NETWORK_CONFIG.get('use_custom_cnn', False):
        base_kwargs.update({
            'use_flexible_cnn': True,
            'cnn_layers': NETWORK_CONFIG['cnn_layers'],
            'cnn_hidden_size': NETWORK_CONFIG['cnn_hidden_size'],
            'cnn_final_layers': NETWORK_CONFIG['cnn_final_layers'],
            'cnn_activation': NETWORK_CONFIG.get('cnn_activation', 'relu'),
        })
    
    # Add MLP hidden size (requires modifying warm_up.py to use this)
    base_kwargs['hidden_size'] = NETWORK_CONFIG['hidden_size']
    
    return base_kwargs

def print_network_summary():
    """Print a detailed summary of the current network configuration."""
    print("Network Architecture Summary:")
    print("=" * 50)
    
    if NETWORK_CONFIG.get('use_custom_cnn', False):
        print("🧠 Architecture Type: Flexible CNN")
        print(f"   📊 Conv Layers: {len(NETWORK_CONFIG['cnn_layers'])}")
        for i, layer in enumerate(NETWORK_CONFIG['cnn_layers']):
            if len(layer) == 3:
                channels, kernel, stride = layer
                padding = 0
            else:
                channels, kernel, stride, padding = layer
            print(f"      Layer {i+1}: {channels} filters, {kernel}x{kernel} kernel, stride {stride}, padding {padding}")
        
        print(f"   🔗 FC Hidden Size: {NETWORK_CONFIG['cnn_hidden_size']}")
        print(f"   📈 Final FC Layers: {NETWORK_CONFIG['cnn_final_layers']}")
        print(f"   ⚡ Activation: {NETWORK_CONFIG.get('cnn_activation', 'relu')}")
    else:
        if len(EXPECTED_OBS_SHAPE) == 3:
            print("🧠 Architecture Type: Standard CNN (Nature CNN)")
            print("   📊 Conv Layers: 3 (hardcoded)")
            print("      Layer 1: 32 filters, 8x8 kernel, stride 4")
            print("      Layer 2: 64 filters, 4x4 kernel, stride 2") 
            print("      Layer 3: 64 filters, 3x3 kernel, stride 1")
            print("   🔗 FC Hidden Size: 512 (hardcoded)")
        else:
            print("🧠 Architecture Type: MLP")
            print(f"   🔗 Hidden Size: {NETWORK_CONFIG['hidden_size']}")
            print(f"   📊 Layers: {NETWORK_CONFIG['num_layers']} (hardcoded)")
    
    print(f"   🔧 Layer Norm: {NETWORK_CONFIG['layernorm']}")
    print("=" * 50)

# Add expected observation shape for network selection
EXPECTED_OBS_SHAPE = (1,)  # Default to 1D for Deep Sea Treasure

def print_config_summary():
    """Print a summary of the current configuration."""
    print("=" * 60)
    print("DEEP SEA TREASURE TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Environment: {ENV_NAME}")
    print(f"Objectives: {OBJECTIVES_NUM}")
    print(f"Save Directory: {SAVE_DIR}")
    print()
    
    # Print detailed network summary
    print_network_summary()
    print()
    
    print("Training Parameters:")
    print(f"  Total Steps: {TRAINING_CONFIG['num_env_steps']:,}")
    print(f"  Parallel Envs: {TRAINING_CONFIG['num_processes']}")
    print(f"  Learning Rate: {PPO_CONFIG['lr']}")
    print(f"  PPO Epochs: {PPO_CONFIG['ppo_epoch']}")
    print()
    print("Multi-Objective Settings:")
    print(f"  Weight Range: [{MO_CONFIG['min_weight']}, {MO_CONFIG['max_weight']}]")
    print(f"  Weight Step: {MO_CONFIG['delta_weight']}")
    print(f"  Selection Method: {MO_CONFIG['selection_method']}")
    print()
    print("Key Flags:")
    print(f"  Raw Objectives: {EXPERIMENTAL_CONFIG['raw']}")
    print(f"  Save Models: {not EXPERIMENTAL_CONFIG['no_save_models']}")
    print(f"  Obj Normalization: {NORMALIZATION_CONFIG['obj_rms']}")
    print()
    print("Expected Pareto Front:")
    for point in EXPECTED_PARETO_FRONT:
        print(f"  Treasure: {point['treasure']}, Steps: {point['steps']} ({point['description']})")
    print("=" * 60)

# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

def apply_fast_test_config():
    """Apply configuration for fast testing (shorter training)."""
    global TRAINING_CONFIG
    TRAINING_CONFIG['num_env_steps'] = 5000
    TRAINING_CONFIG['warmup_iter'] = 5
    TRAINING_CONFIG['update_iter'] = 5
    print("Applied FAST TEST configuration (5,000 steps)")

def apply_full_training_config():
    """Apply configuration for full training."""
    global TRAINING_CONFIG
    TRAINING_CONFIG['num_env_steps'] = 100000
    TRAINING_CONFIG['warmup_iter'] = 40
    TRAINING_CONFIG['update_iter'] = 20
    print("Applied FULL TRAINING configuration (100,000 steps)")

def apply_large_network_config():
    """Apply configuration for larger neural network."""
    global NETWORK_CONFIG
    NETWORK_CONFIG['hidden_size'] = 128
    NETWORK_CONFIG['layernorm'] = True
    print("Applied LARGE NETWORK configuration (128 hidden units with LayerNorm)")

def apply_small_network_config():
    """Apply configuration for smaller neural network."""
    global NETWORK_CONFIG
    NETWORK_CONFIG['hidden_size'] = 32
    NETWORK_CONFIG['layernorm'] = False
    print("Applied SMALL NETWORK configuration (32 hidden units)")

def apply_cnn_preset(preset_name):
    """Apply a CNN preset configuration."""
    global NETWORK_CONFIG
    if preset_name not in NETWORK_CONFIG['cnn_presets']:
        available = list(NETWORK_CONFIG['cnn_presets'].keys())
        raise ValueError(f"Unknown CNN preset '{preset_name}'. Available: {available}")
    
    preset = NETWORK_CONFIG['cnn_presets'][preset_name]
    NETWORK_CONFIG['use_custom_cnn'] = True
    NETWORK_CONFIG['cnn_layers'] = preset['cnn_layers']
    NETWORK_CONFIG['cnn_hidden_size'] = preset['cnn_hidden_size']
    NETWORK_CONFIG['cnn_final_layers'] = preset['cnn_final_layers']
    print(f"Applied CNN preset: {preset_name}")

def apply_atari_cnn_config():
    """Apply Atari-style CNN configuration."""
    apply_cnn_preset('atari')

def apply_meltingpot_cnn_config():
    """Apply MeltingPot-style CNN configuration."""
    apply_cnn_preset('meltingpot')

def apply_deep_nature_cnn_config():
    """Apply Deep Nature CNN configuration."""
    apply_cnn_preset('deep_nature')

def apply_lightweight_cnn_config():
    """Apply Lightweight CNN configuration."""
    apply_cnn_preset('lightweight')

def create_custom_cnn_config(cnn_layers, cnn_hidden_size=512, cnn_final_layers=None):
    """Create a custom CNN configuration.
    
    Args:
        cnn_layers: List of [channels, kernel_size, stride, padding] for each conv layer
        cnn_hidden_size: Size of first FC layer after conv layers
        cnn_final_layers: List of sizes for additional FC layers
    
    Example:
        create_custom_cnn_config([
            [32, 8, 4, 0],      # 32 filters, 8x8 kernel, stride 4
            [64, 4, 2, 0],      # 64 filters, 4x4 kernel, stride 2
            [128, 3, 1, 1]      # 128 filters, 3x3 kernel, stride 1, padding 1
        ], cnn_hidden_size=1024, cnn_final_layers=[1024, 512])
    """
    global NETWORK_CONFIG
    NETWORK_CONFIG['use_custom_cnn'] = True
    NETWORK_CONFIG['cnn_layers'] = cnn_layers
    NETWORK_CONFIG['cnn_hidden_size'] = cnn_hidden_size
    NETWORK_CONFIG['cnn_final_layers'] = cnn_final_layers or [cnn_hidden_size]
    print(f"Applied custom CNN configuration with {len(cnn_layers)} conv layers")
