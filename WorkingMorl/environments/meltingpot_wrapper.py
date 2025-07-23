"""
MeltingPot Environment Wrapper for PGMORL Multi-Objective Learning

This module provides wrapper classes to integrate MeltingPot multi-agent substrates
with PGMORL's multi-objective reinforcement learning framework.

Key Features:
- Bridges MeltingPot dm_env interface with Gym interface
- Converts single-agent rewards to multi-objective rewards
- Supports both competitive and cooperative scenarios
- Maintains compatibility with PGMORL's existing training pipeline
"""

import sys
import os
import numpy as np
import gym
from gym import spaces
from typing import Dict, List, Tuple, Optional, Any
import importlib.util

# Try to import MeltingPot components
try:
    from meltingpot import substrate
    import dm_env
    MELTINGPOT_AVAILABLE = True
except ImportError:
    MELTINGPOT_AVAILABLE = False
    print("Warning: MeltingPot not found. Install MeltingPot to use this wrapper.")


class MultiObjectiveRewardTransformer:
    """
    Transforms single-agent rewards into multi-objective rewards for PGMORL.
    
    Different transformation strategies for different substrate types:
    - Competitive: Individual reward vs. Collective performance
    - Cooperative: Task completion vs. Efficiency/cooperation measures
    - Mixed: Individual vs. group vs. task-specific objectives
    """
    
    def __init__(self, substrate_name: str, num_objectives: int = 2):
        self.substrate_name = substrate_name
        self.num_objectives = num_objectives
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_step = 0
        
    def transform_reward(self, 
                        agent_reward: float, 
                        all_rewards: List[float], 
                        timestep_info: Dict) -> np.ndarray:
        """
        Transform single reward into multi-objective reward vector.
        
        Args:
            agent_reward: Individual agent's reward
            all_rewards: Rewards for all agents
            timestep_info: Additional environment information
            
        Returns:
            Multi-objective reward vector
        """
        self.episode_step += 1
        
        if "collaborative" in self.substrate_name.lower():
            return self._collaborative_transform(agent_reward, all_rewards, timestep_info)
        elif "competition" in self.substrate_name.lower() or "bach" in self.substrate_name.lower():
            return self._competitive_transform(agent_reward, all_rewards, timestep_info)
        else:
            return self._general_transform(agent_reward, all_rewards, timestep_info)
    
    def _collaborative_transform(self, agent_reward: float, all_rewards: List[float], info: Dict) -> np.ndarray:
        """Transform rewards for collaborative scenarios."""
        # Objective 1: Individual task performance
        individual_obj = agent_reward
        
        # Objective 2: Team cooperation (sum of all rewards)
        cooperation_obj = sum(all_rewards) / len(all_rewards)
        
        if self.num_objectives == 2:
            return np.array([individual_obj, cooperation_obj], dtype=np.float32)
        elif self.num_objectives == 3:
            # Objective 3: Efficiency (reward per step)
            efficiency_obj = agent_reward / max(self.episode_step, 1) * 100
            return np.array([individual_obj, cooperation_obj, efficiency_obj], dtype=np.float32)
        else:
            # Pad or truncate to desired number of objectives
            base_rewards = [individual_obj, cooperation_obj]
            while len(base_rewards) < self.num_objectives:
                base_rewards.append(0.0)
            return np.array(base_rewards[:self.num_objectives], dtype=np.float32)
    
    def _competitive_transform(self, agent_reward: float, all_rewards: List[float], info: Dict) -> np.ndarray:
        """Transform rewards for competitive scenarios."""
        # Objective 1: Individual performance
        individual_obj = agent_reward
        
        # Objective 2: Relative performance (vs others)
        other_rewards = [r for r in all_rewards if r != agent_reward]
        if other_rewards:
            relative_obj = agent_reward - np.mean(other_rewards)
        else:
            relative_obj = agent_reward
            
        if self.num_objectives == 2:
            return np.array([individual_obj, relative_obj], dtype=np.float32)
        elif self.num_objectives == 3:
            # Objective 3: Consistency (negative variance of recent rewards)
            self.episode_rewards.append(agent_reward)
            if len(self.episode_rewards) > 10:
                self.episode_rewards = self.episode_rewards[-10:]
            consistency_obj = -np.var(self.episode_rewards) if len(self.episode_rewards) > 1 else 0.0
            return np.array([individual_obj, relative_obj, consistency_obj], dtype=np.float32)
        else:
            base_rewards = [individual_obj, relative_obj]
            while len(base_rewards) < self.num_objectives:
                base_rewards.append(0.0)
            return np.array(base_rewards[:self.num_objectives], dtype=np.float32)
    
    def _general_transform(self, agent_reward: float, all_rewards: List[float], info: Dict) -> np.ndarray:
        """General transformation for unknown substrate types."""
        # Objective 1: Individual reward
        individual_obj = agent_reward
        
        # Objective 2: Social reward (collective performance)
        social_obj = sum(all_rewards) / len(all_rewards)
        
        return np.array([individual_obj, social_obj], dtype=np.float32)
    
    def reset(self):
        """Reset episode-specific tracking."""
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_step = 0


class MeltingPotSingleAgentWrapper(gym.Env):
    """
    Wraps a MeltingPot substrate to create a single-agent Gym environment for PGMORL.
    
    This wrapper:
    1. Extracts one agent's view from the multi-agent environment
    2. Converts dm_env TimeStep to Gym step format
    3. Transforms rewards to multi-objective format
    4. Handles observation/action space conversion
    """
    
    def __init__(self, 
                 substrate_name: str,
                 agent_index: int = 0,
                 num_objectives: int = 2,
                 roles: Optional[List[str]] = None):
        """
        Initialize the wrapper.
        
        Args:
            substrate_name: Name of MeltingPot substrate
            agent_index: Which agent to control (0-indexed)
            num_objectives: Number of objectives for multi-objective learning
            roles: Player roles for the substrate
        """
        if not MELTINGPOT_AVAILABLE:
            raise ImportError("MeltingPot not available. Please install MeltingPot.")
            
        self.substrate_name = substrate_name
        self.agent_index = agent_index
        self.num_objectives = num_objectives
        
        # Build the MeltingPot environment
        if roles is None:
            # Get default roles for this substrate
            config = substrate.get_config(substrate_name)
            roles = config.default_player_roles
            
        self.substrate_env = substrate.build(substrate_name, roles=roles)
        self.num_agents = len(roles)
        
        # Initialize reward transformer
        self.reward_transformer = MultiObjectiveRewardTransformer(
            substrate_name, num_objectives
        )
        
        # Set up observation and action spaces
        self._setup_spaces()
        
        # Episode tracking
        self.current_timestep = None
        self.episode_step = 0
        
    def _setup_spaces(self):
        """Setup Gym observation and action spaces."""
        # Get specs from the substrate
        obs_specs = self.substrate_env.observation_spec()
        action_specs = self.substrate_env.action_spec()
        
        # Extract agent's observation space
        agent_obs_spec = obs_specs[self.agent_index]
        self.observation_space = self._spec_to_gym_space(agent_obs_spec)
        
        # Extract agent's action space
        agent_action_spec = action_specs[self.agent_index]
        self.action_space = self._discrete_spec_to_gym_space(agent_action_spec)
        
    def _spec_to_gym_space(self, obs_spec: Dict) -> spaces.Dict:
        """Convert dm_env observation spec to Gym space."""
        gym_spaces = {}
        
        for name, spec in obs_spec.items():
            if hasattr(spec, 'shape') and hasattr(spec, 'dtype'):
                if 'RGB' in name or 'WORLD' in name:
                    # Image observations
                    gym_spaces[name] = spaces.Box(
                        low=0, high=255, 
                        shape=spec.shape, 
                        dtype=np.uint8
                    )
                else:
                    # Other observations
                    if hasattr(spec, 'minimum') and hasattr(spec, 'maximum'):
                        gym_spaces[name] = spaces.Box(
                            low=spec.minimum, high=spec.maximum,
                            shape=spec.shape, dtype=spec.dtype
                        )
                    else:
                        # Assume bounded by dtype limits
                        if np.issubdtype(spec.dtype, np.integer):
                            info = np.iinfo(spec.dtype)
                            low, high = info.min, info.max
                        else:
                            info = np.finfo(spec.dtype)
                            low, high = info.min, info.max
                        
                        gym_spaces[name] = spaces.Box(
                            low=low, high=high,
                            shape=spec.shape, dtype=spec.dtype
                        )
                        
        return spaces.Dict(gym_spaces)
    
    def _discrete_spec_to_gym_space(self, action_spec) -> spaces.Discrete:
        """Convert dm_env discrete action spec to Gym Discrete space."""
        if hasattr(action_spec, 'num_values'):
            return spaces.Discrete(action_spec.num_values)
        else:
            # Assume it's already a simple discrete space
            return spaces.Discrete(8)  # Common for MeltingPot
    
    def reset(self, **kwargs):
        """Reset the environment and return initial observation."""
        # Reset the substrate
        self.current_timestep = self.substrate_env.reset()
        self.episode_step = 0
        self.reward_transformer.reset()
        
        # Extract agent's observation
        agent_obs = self.current_timestep.observation[self.agent_index]
        
        return agent_obs
        
    def step(self, action):
        """Step the environment with agent's action."""
        # Create action list for all agents (others get random actions)
        actions = []
        for i in range(self.num_agents):
            if i == self.agent_index:
                actions.append(action)
            else:
                # Random action for other agents (can be improved)
                actions.append(self.action_space.sample())
        
        # Step the substrate
        self.current_timestep = self.substrate_env.step(actions)
        self.episode_step += 1
        
        # Extract agent's observation, reward, done
        agent_obs = self.current_timestep.observation[self.agent_index]
        agent_reward = self.current_timestep.reward[self.agent_index]
        done = self.current_timestep.last()
        
        # Transform to multi-objective reward
        all_rewards = list(self.current_timestep.reward)
        mo_reward = self.reward_transformer.transform_reward(
            agent_reward, all_rewards, {}
        )
        
        # Info dict
        info = {
            'agent_reward': agent_reward,
            'all_rewards': all_rewards,
            'episode_step': self.episode_step
        }
        
        return agent_obs, mo_reward, done, info
    
    def render(self, mode='rgb_array'):
        """Render the environment."""
        if mode == 'rgb_array':
            obs = self.substrate_env.observation()
            if obs and len(obs) > 0 and 'WORLD.RGB' in obs[0]:
                return obs[0]['WORLD.RGB']
        return None
        
    def close(self):
        """Close the environment."""
        if hasattr(self.substrate_env, 'close'):
            self.substrate_env.close()


class MeltingPotMultiAgentWrapper(gym.Env):
    """
    Advanced wrapper for multi-agent PGMORL training on MeltingPot substrates.
    
    This wrapper supports:
    - Multiple PGMORL agents training simultaneously
    - Centralized vs. decentralized training modes
    - Different multi-objective reward schemes per agent
    - Communication between agents (if supported by substrate)
    """
    
    def __init__(self, 
                 substrate_name: str,
                 num_pgmorl_agents: int = 1,
                 num_objectives: int = 2,
                 training_mode: str = "decentralized",
                 roles: Optional[List[str]] = None):
        """
        Initialize multi-agent wrapper.
        
        Args:
            substrate_name: Name of MeltingPot substrate
            num_pgmorl_agents: Number of agents to train with PGMORL
            num_objectives: Number of objectives for each agent
            training_mode: "centralized" or "decentralized"
            roles: Player roles for the substrate
        """
        if not MELTINGPOT_AVAILABLE:
            raise ImportError("MeltingPot not available. Please install MeltingPot.")
            
        self.substrate_name = substrate_name
        self.num_pgmorl_agents = num_pgmorl_agents
        self.num_objectives = num_objectives
        self.training_mode = training_mode
        
        # Build substrate
        if roles is None:
            config = substrate.get_config(substrate_name)
            roles = config.default_player_roles
            
        self.substrate_env = substrate.build(substrate_name, roles=roles)
        self.num_total_agents = len(roles)
        
        if num_pgmorl_agents > self.num_total_agents:
            raise ValueError(f"Cannot have more PGMORL agents ({num_pgmorl_agents}) than total agents ({self.num_total_agents})")
        
        # Setup reward transformers for each PGMORL agent
        self.reward_transformers = [
            MultiObjectiveRewardTransformer(substrate_name, num_objectives)
            for _ in range(num_pgmorl_agents)
        ]
        
        # Setup spaces
        self._setup_spaces()
        
        # Episode tracking
        self.current_timestep = None
        self.episode_step = 0
        
    def _setup_spaces(self):
        """Setup observation and action spaces for PGMORL agents."""
        obs_specs = self.substrate_env.observation_spec()
        action_specs = self.substrate_env.action_spec()
        
        # For centralized training, combine observations
        if self.training_mode == "centralized":
            # Combined observation space
            all_obs_spaces = {}
            for i in range(self.num_pgmorl_agents):
                agent_obs_spec = obs_specs[i]
                for name, spec in agent_obs_spec.items():
                    key = f"agent_{i}_{name}"
                    all_obs_spaces[key] = self._spec_to_gym_space({name: spec})[name]
            
            self.observation_space = spaces.Dict(all_obs_spaces)
            
            # Combined action space (MultiDiscrete)
            action_sizes = []
            for i in range(self.num_pgmorl_agents):
                action_spec = action_specs[i]
                if hasattr(action_spec, 'num_values'):
                    action_sizes.append(action_spec.num_values)
                else:
                    action_sizes.append(8)
            
            self.action_space = spaces.MultiDiscrete(action_sizes)
            
        else:  # decentralized
            # Use first agent's spaces as template
            agent_obs_spec = obs_specs[0]
            self.observation_space = self._spec_to_gym_space(agent_obs_spec)
            
            action_spec = action_specs[0]
            self.action_space = self._discrete_spec_to_gym_space(action_spec)
    
    def _spec_to_gym_space(self, obs_spec: Dict) -> spaces.Dict:
        """Convert dm_env observation spec to Gym space."""
        # Same implementation as single agent wrapper
        gym_spaces = {}
        
        for name, spec in obs_spec.items():
            if hasattr(spec, 'shape') and hasattr(spec, 'dtype'):
                if 'RGB' in name or 'WORLD' in name:
                    gym_spaces[name] = spaces.Box(
                        low=0, high=255, 
                        shape=spec.shape, 
                        dtype=np.uint8
                    )
                else:
                    if hasattr(spec, 'minimum') and hasattr(spec, 'maximum'):
                        gym_spaces[name] = spaces.Box(
                            low=spec.minimum, high=spec.maximum,
                            shape=spec.shape, dtype=spec.dtype
                        )
                    else:
                        if np.issubdtype(spec.dtype, np.integer):
                            info = np.iinfo(spec.dtype)
                            low, high = info.min, info.max
                        else:
                            info = np.finfo(spec.dtype)
                            low, high = info.min, info.max
                        
                        gym_spaces[name] = spaces.Box(
                            low=low, high=high,
                            shape=spec.shape, dtype=spec.dtype
                        )
                        
        return spaces.Dict(gym_spaces)
    
    def _discrete_spec_to_gym_space(self, action_spec) -> spaces.Discrete:
        """Convert dm_env discrete action spec to Gym Discrete space."""
        if hasattr(action_spec, 'num_values'):
            return spaces.Discrete(action_spec.num_values)
        else:
            return spaces.Discrete(8)
    
    def reset(self, **kwargs):
        """Reset environment."""
        self.current_timestep = self.substrate_env.reset()
        self.episode_step = 0
        
        for transformer in self.reward_transformers:
            transformer.reset()
        
        if self.training_mode == "centralized":
            # Return combined observations
            combined_obs = {}
            for i in range(self.num_pgmorl_agents):
                agent_obs = self.current_timestep.observation[i]
                for name, value in agent_obs.items():
                    combined_obs[f"agent_{i}_{name}"] = value
            return combined_obs
        else:
            # Return first agent's observation for decentralized training
            return self.current_timestep.observation[0]
    
    def step(self, action):
        """Step environment with PGMORL agent actions."""
        # Parse actions
        if self.training_mode == "centralized":
            pgmorl_actions = list(action)
        else:
            pgmorl_actions = [action]
        
        # Create full action list
        actions = []
        for i in range(self.num_total_agents):
            if i < len(pgmorl_actions):
                actions.append(pgmorl_actions[i])
            else:
                # Random action for non-PGMORL agents
                actions.append(self.action_space.sample() if self.training_mode != "centralized" 
                             else np.random.randint(8))
        
        # Step substrate
        self.current_timestep = self.substrate_env.step(actions)
        self.episode_step += 1
        
        # Process rewards for PGMORL agents
        all_rewards = list(self.current_timestep.reward)
        mo_rewards = []
        
        for i in range(self.num_pgmorl_agents):
            agent_reward = all_rewards[i]
            mo_reward = self.reward_transformers[i].transform_reward(
                agent_reward, all_rewards, {}
            )
            mo_rewards.append(mo_reward)
        
        # Process observations
        if self.training_mode == "centralized":
            combined_obs = {}
            for i in range(self.num_pgmorl_agents):
                agent_obs = self.current_timestep.observation[i]
                for name, value in agent_obs.items():
                    combined_obs[f"agent_{i}_{name}"] = value
            obs = combined_obs
            reward = np.mean(mo_rewards, axis=0)  # Average multi-objective rewards
        else:
            obs = self.current_timestep.observation[0]
            reward = mo_rewards[0]
        
        done = self.current_timestep.last()
        
        info = {
            'all_rewards': all_rewards,
            'mo_rewards': mo_rewards,
            'episode_step': self.episode_step,
            'training_mode': self.training_mode
        }
        
        return obs, reward, done, info
    
    def render(self, mode='rgb_array'):
        """Render the environment."""
        if mode == 'rgb_array':
            obs = self.substrate_env.observation()
            if obs and len(obs) > 0 and 'WORLD.RGB' in obs[0]:
                return obs[0]['WORLD.RGB']
        return None
        
    def close(self):
        """Close the environment."""
        if hasattr(self.substrate_env, 'close'):
            self.substrate_env.close()


def list_available_substrates():
    """List all available MeltingPot substrates."""
    if not MELTINGPOT_AVAILABLE:
        print("MeltingPot not available.")
        return []
    
    return substrate.SUBSTRATES


def get_substrate_info(substrate_name: str):
    """Get information about a specific substrate."""
    if not MELTINGPOT_AVAILABLE:
        print("MeltingPot not available.")
        return None
    
    try:
        config = substrate.get_config(substrate_name)
        return {
            'name': substrate_name,
            'default_player_roles': config.default_player_roles,
            'num_players': len(config.default_player_roles),
            'valid_roles': getattr(config, 'valid_roles', None),
            'action_set': getattr(config, 'action_set', None),
        }
    except Exception as e:
        print(f"Error getting info for substrate {substrate_name}: {e}")
        return None


# Environment registration functions
def register_meltingpot_environments():
    """Register MeltingPot environments with Gym."""
    if not MELTINGPOT_AVAILABLE:
        print("MeltingPot not available. Cannot register environments.")
        return
    
    # Register single-agent wrappers for popular substrates
    popular_substrates = [
        'collaborative_cooking__asymmetric',
        'bach_or_stravinsky_in_the_matrix__repeated',
        'prisoners_dilemma_in_the_matrix__repeated',
        'clean_up',
    ]
    
    for substrate_name in popular_substrates:
        try:
            # Single agent version
            gym.register(
                id=f'MeltingPot-{substrate_name}-SingleAgent-v0',
                entry_point='environments.meltingpot_wrapper:MeltingPotSingleAgentWrapper',
                kwargs={'substrate_name': substrate_name, 'num_objectives': 2},
                max_episode_steps=1000
            )
            
            # Multi-agent version
            gym.register(
                id=f'MeltingPot-{substrate_name}-MultiAgent-v0',
                entry_point='environments.meltingpot_wrapper:MeltingPotMultiAgentWrapper',
                kwargs={'substrate_name': substrate_name, 'num_objectives': 2},
                max_episode_steps=1000
            )
            
        except gym.error.Error:
            pass  # Already registered


if __name__ == "__main__":
    # Test the wrapper
    if MELTINGPOT_AVAILABLE:
        print("Available substrates:", list_available_substrates())
        
        # Test single agent wrapper
        try:
            env = MeltingPotSingleAgentWrapper(
                substrate_name='collaborative_cooking__asymmetric',
                num_objectives=2
            )
            print(f"Created environment with observation space: {env.observation_space}")
            print(f"Action space: {env.action_space}")
            
            obs = env.reset()
            print(f"Initial observation keys: {obs.keys() if isinstance(obs, dict) else type(obs)}")
            
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            print(f"Step result - Reward shape: {reward.shape}, Done: {done}")
            
            env.close()
            print("Single agent test successful!")
            
        except Exception as e:
            print(f"Single agent test failed: {e}")
            
    else:
        print("MeltingPot not available. Please install MeltingPot to test the wrapper.")
