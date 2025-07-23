# PGMORL-MeltingPot Integration

This integration enables training multi-objective reinforcement learning agents on MeltingPot's multi-agent environments using the PGMORL framework.

## Overview

The integration bridges two powerful frameworks:
- **PGMORL**: Multi-objective reinforcement learning with Pareto-optimal solution discovery
- **MeltingPot**: Multi-agent environments for studying social dilemmas and cooperation

## Key Features

### 🎯 Multi-Objective Multi-Agent Learning
- Transform single-agent rewards into meaningful multi-objective signals
- Learn trade-offs between individual performance, cooperation, and efficiency
- Discover Pareto-optimal policies for social dilemmas

### 🧠 Optimized CNN Architectures
- Pre-configured CNN architectures for MeltingPot's visual observations
- Sprite-aligned convolutions for competitive scenarios
- Deep feature extraction for collaborative scenarios

### 🔄 Flexible Training Modes
- **Single Agent**: One PGMORL agent vs. random/baseline agents
- **Multi-Agent Decentralized**: Multiple independent PGMORL agents
- **Multi-Agent Centralized**: Coordinated training with shared observations

### 🎮 Supported Environments
- Collaborative Cooking (cooperation, efficiency)
- Bach or Stravinsky (coordination games)
- Prisoner's Dilemma (competition vs. cooperation)
- Clean Up (commons dilemma)

## Installation

### Prerequisites
1. **PGMORL**: Already available in this workspace
2. **MeltingPot**: Install with:
   ```bash
   pip install dm-meltingpot
   ```

### Verify Installation
```bash
python meltingpot_demo.py
```

## Quick Start

### 1. Explore Available Configurations
```bash
python meltingpot_demo.py
```

### 2. List Available Substrates
```bash
python example_meltingpot_training.py --list-substrates
```

### 3. Test Environment
```bash
python example_meltingpot_training.py --substrate collaborative_cooking__asymmetric --test-env
```

### 4. Run Training
```bash
# Single agent learning cooperation
python example_meltingpot_training.py --substrate collaborative_cooking__asymmetric --mode single_agent

# Two agents in competitive scenario
python example_meltingpot_training.py --substrate bach_or_stravinsky_in_the_matrix__repeated --mode multi_agent_decentralized
```

## Configuration System

### CNN Presets
- **meltingpot_competitive**: Optimized for competitive scenarios
- **meltingpot_collaborative**: Deep features for collaborative tasks
- **meltingpot_general**: Balanced architecture for any substrate

### Multi-Objective Transformations

#### Collaborative Scenarios
- **Individual Performance**: Direct agent reward
- **Team Cooperation**: Average of all agent rewards
- **Efficiency**: Reward per timestep ratio

#### Competitive Scenarios
- **Individual Performance**: Direct agent reward
- **Relative Performance**: Agent reward vs. others
- **Consistency**: Reward variance (negative for stability)

### Training Modes

#### Single Agent Mode
- One PGMORL agent, others are random/baseline
- Good for: Initial testing, baseline comparison
- Environment: `MeltingPotSingleAgentWrapper`

#### Multi-Agent Decentralized
- Multiple independent PGMORL agents
- Good for: Competitive scenarios, emergent behaviors
- Environment: `MeltingPotMultiAgentWrapper` (decentralized)

#### Multi-Agent Centralized
- Coordinated training with shared observations
- Good for: Collaborative scenarios, team coordination
- Environment: `MeltingPotMultiAgentWrapper` (centralized)

## Example Usage

### Configuration Examples

```python
from deep_sea_treasure_config import apply_meltingpot_preset, get_meltingpot_base_kwargs

# Get configuration for collaborative cooking
config = apply_meltingpot_preset('collaborative_cooking__asymmetric', 'single_agent')
base_kwargs = get_meltingpot_base_kwargs('collaborative_cooking__asymmetric', 'single_agent')

# Override specific parameters
config = apply_meltingpot_preset(
    'bach_or_stravinsky_in_the_matrix__repeated', 
    'multi_agent_decentralized',
    num_env_steps=200000,
    lr=5e-5
)
```

### Environment Creation

```python
from environments.meltingpot_wrapper import MeltingPotSingleAgentWrapper

# Create single agent environment
env = MeltingPotSingleAgentWrapper(
    substrate_name='collaborative_cooking__asymmetric',
    num_objectives=3
)

# Use with PGMORL
obs = env.reset()
action = env.action_space.sample()
obs, multi_obj_reward, done, info = env.step(action)
print(f"Multi-objective reward: {multi_obj_reward}")  # [individual, cooperation, efficiency]
```

## Available Substrates

### Collaborative Cooking (Asymmetric)
- **Type**: Collaborative
- **Agents**: 4
- **Objectives**: Individual reward, team cooperation, efficiency
- **Description**: Asymmetric collaborative cooking scenario

### Bach or Stravinsky (Repeated)
- **Type**: Competitive
- **Agents**: 2
- **Objectives**: Individual performance, relative performance
- **Description**: Coordination game with multiple equilibria

### Prisoner's Dilemma (Repeated)
- **Type**: Competitive
- **Agents**: 2
- **Objectives**: Individual reward, cooperation level
- **Description**: Classic social dilemma

### Clean Up
- **Type**: Mixed
- **Agents**: 4
- **Objectives**: Individual reward, collective cleanup, sustainability
- **Description**: Commons dilemma with resource management

## Training Results

Results are saved to the specified directory with:
- Pareto fronts for each objective combination
- Training progress plots
- Model checkpoints (if enabled)
- Configuration files

### Analyzing Results
```bash
python analyze_results.py --save-dir ./meltingpot_results
python visualize_results.py --save-dir ./meltingpot_results
```

## Architecture Details

### Environment Wrapper Architecture
```
MeltingPot Substrate (dm_env)
           ↓
   MeltingPot*Wrapper (Gym)
           ↓
  Multi-Objective Rewards
           ↓
      PGMORL Training
```

### Multi-Objective Reward Flow
```
Single Agent Reward (scalar)
           ↓
  Reward Transformer
           ↓
Multi-Objective Vector [obj1, obj2, obj3]
           ↓
   PGMORL Scalarization
           ↓
    Policy Update
```

### CNN Architecture Flow
```
Visual Observation (RGB)
         ↓
   CNN Layers (preset)
         ↓
 Feature Extraction
         ↓
  Fully Connected
         ↓
Multi-Objective Q-values
```

## Advanced Usage

### Custom Reward Transformations
```python
class CustomRewardTransformer(MultiObjectiveRewardTransformer):
    def _collaborative_transform(self, agent_reward, all_rewards, info):
        # Custom transformation logic
        obj1 = agent_reward
        obj2 = custom_cooperation_metric(all_rewards)
        obj3 = custom_efficiency_metric(info)
        return np.array([obj1, obj2, obj3])
```

### Custom CNN Architectures
```python
from deep_sea_treasure_config import create_custom_cnn_config

custom_cnn = create_custom_cnn_config(
    cnn_layers=[[32, 8, 4, 0], [64, 4, 2, 0], [128, 3, 1, 0]],
    cnn_hidden_size=1024,
    cnn_final_layers=[1024, 512, 256]
)
```

## Troubleshooting

### Common Issues

1. **MeltingPot not installed**
   ```bash
   pip install dm-meltingpot
   ```

2. **GPU memory issues**
   - Reduce `num_processes`
   - Use smaller CNN architectures
   - Reduce batch sizes

3. **Slow training**
   - Use GPU (`--cuda`)
   - Increase `num_processes`
   - Use lightweight CNN preset

4. **Poor convergence**
   - Adjust learning rate
   - Change weight candidates
   - Try different CNN presets

### Debug Mode
```bash
python example_meltingpot_training.py --debug --test-env
```

## Research Applications

This integration enables research in:
- Multi-objective social dilemmas
- Cooperation vs. competition trade-offs
- Emergent multi-agent behaviors
- Pareto-optimal policy discovery
- Visual multi-agent reinforcement learning

## Contributing

To add new substrates or features:
1. Add substrate configuration to `MELTINGPOT_CONFIG`
2. Define CNN preset if needed
3. Implement custom reward transformation
4. Add to documentation

## References

- **PGMORL**: [Multi-Objective Reinforcement Learning Paper]
- **MeltingPot**: [MeltingPot: A Suite of Multi-Agent Reinforcement Learning Environments]
- **CNN Architectures**: Based on DeepMind's multi-agent RL papers
