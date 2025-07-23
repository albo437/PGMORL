# Deep Sea Treasure Configuration System

This directory now includes a flexible configuration system that separates all training parameters from the main code, making it much easier to experiment with different settings.

## Files Overview

- **`deep_sea_treasure_config.py`** - Main configuration file with all parameters
- **`example_deep_sea_treasure.py`** - Updated training script that reads from config
- **`config_demo.py`** - Demonstration script showing different configurations
- **`CONFIG_README.md`** - This documentation file

## Quick Start

### 1. Basic Usage
```bash
# Run with default configuration
python example_deep_sea_treasure.py

# See configuration examples
python config_demo.py
```

### 2. Modify Parameters
Edit `deep_sea_treasure_config.py` to change any parameter:

```python
# Example: Change training duration
TRAINING_CONFIG = {
    'num_env_steps': 100000,     # Was 50000
    'num_processes': 8,          # Was 4
    # ... other parameters
}

# Example: Use larger neural network
NETWORK_CONFIG = {
    'hidden_size': 128,          # Was 64
    'layernorm': True,           # Was False
    # ... other parameters
}
```

### 3. Use Preset Configurations
Uncomment preset functions in `main()`:

```python
def main():
    # Uncomment one of these:
    apply_fast_test_config()        # Quick testing (5,000 steps)
    # apply_full_training_config()    # Full training (100,000 steps)
    # apply_large_network_config()    # Larger network (128 hidden units)
    # apply_small_network_config()    # Smaller network (32 hidden units)
```

## Configuration Sections

### 🏛️ Environment Configuration
- Environment name and objectives
- Save directory location

### 🧠 Neural Network Architecture
- Hidden layer size (currently 64 units)
- Layer normalization settings
- Number of layers (fixed at 2 in current implementation)

### 🔄 PPO Algorithm Parameters
- Learning rate, epochs, mini-batches
- Discount factor, GAE lambda
- Clipping and loss coefficients

### 📈 Training Schedule
- Total environment steps
- Parallel environments
- Warmup and update iterations

### 🎯 Multi-Objective Settings
- Weight ranges for scalarization
- Selection method (PGMORL)
- Pareto buffer configuration

### 📊 Normalization Settings
- Objective, observation, and return normalization
- Critical `raw` flag for Deep Sea Treasure

### 🔬 Experimental Settings
- Model saving options
- CUDA usage
- Random seeds

## Key Parameters Explained

### Most Important Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_env_steps` | 50,000 | Total training steps |
| `hidden_size` | 64 | Neural network size |
| `num_processes` | 4 | Parallel environments |
| `raw` | True | **Critical**: Use undiscounted objectives |
| `no_save_models` | True | Skip saving .pt/.pkl files |

### Performance Tuning

**For Faster Training:**
```python
apply_fast_test_config()  # 5,000 steps
TRAINING_CONFIG['num_processes'] = 8  # More parallel envs
```

**For Better Results:**
```python
apply_full_training_config()  # 100,000 steps
apply_large_network_config()  # 128 hidden units
PPO_CONFIG['lr'] = 1e-4  # Lower learning rate
```

**For Resource Constraints:**
```python
apply_small_network_config()  # 32 hidden units
TRAINING_CONFIG['num_processes'] = 2  # Fewer parallel envs
```

## Configuration Examples

### Example 1: Quick Testing
```python
# In deep_sea_treasure_config.py
TRAINING_CONFIG['num_env_steps'] = 5000
TRAINING_CONFIG['num_processes'] = 2
NETWORK_CONFIG['hidden_size'] = 32
```

### Example 2: Production Training
```python
# In deep_sea_treasure_config.py
TRAINING_CONFIG['num_env_steps'] = 200000
TRAINING_CONFIG['num_processes'] = 16
NETWORK_CONFIG['hidden_size'] = 128
NETWORK_CONFIG['layernorm'] = True
```

### Example 3: Hyperparameter Search
```python
# Create multiple config files
# deep_sea_treasure_config_lr1e3.py
# deep_sea_treasure_config_lr1e4.py
# deep_sea_treasure_config_lr3e4.py

# Then modify imports in training script
```

## Expected Results

The configuration includes expected Pareto front for Deep Sea Treasure:

- **Treasure: 1.0, Steps: 1** (Quick & Small)
- **Treasure: 8.0, Steps: 6** (Medium Effort & Reward)  
- **Treasure: 20.0, Steps: 14** (High Effort & High Reward)

## Advanced Usage

### Creating Custom Presets
```python
def apply_my_custom_config():
    global TRAINING_CONFIG, NETWORK_CONFIG
    TRAINING_CONFIG['num_env_steps'] = 150000
    NETWORK_CONFIG['hidden_size'] = 96
    print("Applied custom configuration")

# Use in main():
apply_my_custom_config()
```

### Environment-Specific Configs
```python
# Create different config files for different environments
# deep_sea_treasure_config.py  - For DST
# minecart_config.py           - For Minecart
# four_room_config.py          - For Four Room

# Import the appropriate one in your training script
```

### Batch Experiments
```python
# configs/experiment_1.py
from deep_sea_treasure_config import *
TRAINING_CONFIG['num_env_steps'] = 50000
NETWORK_CONFIG['hidden_size'] = 64

# configs/experiment_2.py  
from deep_sea_treasure_config import *
TRAINING_CONFIG['num_env_steps'] = 50000
NETWORK_CONFIG['hidden_size'] = 128

# Run multiple experiments
for config in ['experiment_1', 'experiment_2']:
    # Import config and run training
```

## Tips

1. **Always check the `raw` flag** - It's critical for Deep Sea Treasure
2. **Start with fast test config** - Verify everything works before long training
3. **Monitor GPU memory** - Larger networks and more processes use more memory
4. **Save configs with results** - Copy config file to results directory for reproducibility
5. **Use version control** - Track config changes with git

## Troubleshooting

### Common Issues

**Training too slow:**
- Reduce `num_env_steps`
- Reduce `hidden_size` 
- Reduce `num_processes`

**Poor results:**
- Increase `num_env_steps`
- Increase `hidden_size`
- Enable `layernorm`
- Check `raw` flag is True

**Out of memory:**
- Reduce `num_processes`
- Reduce `hidden_size`
- Reduce `num_mini_batch`

**Wrong objectives:**
- Ensure `raw: True`
- Check `obj_rms` settings
- Verify environment registration

---

This configuration system makes it much easier to experiment with different settings without modifying the main training code. Happy experimenting! 🚀
