# Dummy Multi-Objective Environment
# A simple test environment for multi-objective RL without Mujoco dependencies

import numpy as np
import gym
from gym import spaces

class DummyMOEnv(gym.Env):
    """
    A simple 2D navigation environment with multiple conflicting objectives.
    
    Objectives:
    1. Maximize distance traveled (exploration)
    2. Minimize energy consumption (efficiency) 
    3. (Optional) Stay near center (for 3+ objective version)
    
    The environment is designed to create trade-offs between these objectives.
    """
    
    def __init__(self, obj_num=2, max_steps=500):
        super(DummyMOEnv, self).__init__()
        
        self.obj_num = obj_num
        self.max_steps = max_steps
        self.current_step = 0
        
        # Action space: continuous actions for movement in 2D
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float64
        )
        
        # Observation space: position, velocity, and some additional features
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(6,), dtype=np.float64
        )
        
        # Environment state
        self.position = np.zeros(2, dtype=np.float64)
        self.velocity = np.zeros(2, dtype=np.float64)
        self.last_position = np.zeros(2, dtype=np.float64)
        
        # For objective tracking
        self.total_distance = 0.0
        self.total_energy = 0.0
        
    def reset(self):
        """Reset the environment to a random initial state."""
        self.current_step = 0
        self.position = np.random.uniform(-1.0, 1.0, 2).astype(np.float64)
        self.velocity = np.zeros(2, dtype=np.float64)
        self.last_position = self.position.copy()
        self.total_distance = 0.0
        self.total_energy = 0.0
        
        return self._get_observation()
    
    def step(self, action):
        """Execute one step in the environment."""
        self.current_step += 1
        
        # Ensure action is the right type and shape
        action = np.array(action, dtype=np.float64).flatten()  # Flatten to handle any extra dimensions
        action = np.clip(action, -1.0, 1.0)
        
        # Ensure action has correct shape (2,)
        if action.shape[0] != 2:
            raise ValueError(f"Expected action shape (2,), got {action.shape}")
        
        # Store previous position
        self.last_position = self.position.copy()
        
        # Update velocity and position (simple physics)
        self.velocity += action * 0.1  # Action affects acceleration
        self.velocity *= 0.95  # Damping
        self.position += self.velocity
        
        # Calculate distance moved this step
        step_distance = np.linalg.norm(self.position - self.last_position)
        self.total_distance += step_distance
        
        # Calculate energy consumption (based on action magnitude)
        step_energy = np.linalg.norm(action)
        self.total_energy += step_energy
        
        # Calculate objectives
        if self.obj_num == 2:
            # Objective 1: Maximize distance (exploration)
            obj1 = step_distance
            # Objective 2: Maximize efficiency (inverse of energy) - make it positive
            # Add a small constant to avoid division by zero and ensure positivity
            obj2 = 1.0 / (step_energy + 0.1)  # Higher value = more efficient
            objectives = np.array([obj1, obj2], dtype=np.float64)
            
        elif self.obj_num == 3:
            obj1 = step_distance
            obj2 = 1.0 / (step_energy + 0.1)  # Efficiency (positive)
            # Objective 3: Stay near center - use max distance minus current distance
            max_distance = 5.0  # Environment boundary
            obj3 = max_distance - np.linalg.norm(self.position)
            objectives = np.array([obj1, obj2, obj3], dtype=np.float64)
            
        elif self.obj_num >= 4:
            obj1 = step_distance
            obj2 = 1.0 / (step_energy + 0.1)  # Efficiency (positive)
            max_distance = 5.0
            obj3 = max_distance - np.linalg.norm(self.position)
            # Objective 4: Stability (inverse of velocity magnitude)
            obj4 = 1.0 / (np.linalg.norm(self.velocity) + 0.1)
            
            objectives = np.array([obj1, obj2, obj3, obj4], dtype=np.float64)
            
            # Add more objectives if needed (up to obj_num)
            if self.obj_num > 4:
                extra_objs = []
                for i in range(4, self.obj_num):
                    # Create synthetic positive objectives
                    if i == 4:
                        extra_objs.append(1.0 - np.abs(self.position[0]) / 5.0)  # Minimize |x|
                    elif i == 5:
                        extra_objs.append(1.0 - np.abs(self.position[1]) / 5.0)  # Minimize |y|
                    else:
                        # Random synthetic positive objectives for higher dimensions
                        extra_objs.append(0.5 + 0.5 * np.sin(self.total_distance * (i - 5)))
                
                objectives = np.concatenate([objectives, np.array(extra_objs, dtype=np.float64)])
        else:
            objectives = np.zeros(self.obj_num, dtype=np.float64)
        
        # Simple reward (can be weighted sum or just use one objective)
        reward = objectives[0] + 0.1 * objectives[1]
        
        # Done condition
        done = (self.current_step >= self.max_steps or 
                np.linalg.norm(self.position) > 5.0)
        
        # Info dict must contain 'obj' for multi-objective algorithms
        info = {
            'obj': objectives,
            'obj_raw': objectives.copy(),  # Some algorithms might want this
            'step_distance': step_distance,
            'step_energy': step_energy,
            'total_distance': self.total_distance,
            'total_energy': self.total_energy,
            'position': self.position.copy(),
            'velocity': self.velocity.copy()
        }
        
        return self._get_observation(), reward, done, info
    
    def _get_observation(self):
        """Get the current observation."""
        # Observation: [pos_x, pos_y, vel_x, vel_y, total_distance, total_energy]
        obs = np.concatenate([
            self.position,
            self.velocity,
            np.array([self.total_distance, self.total_energy], dtype=np.float64)
        ], dtype=np.float64)
        
        return obs
    
    def render(self, mode='human'):
        """Render the environment (optional)."""
        if mode == 'human':
            print(f"Step {self.current_step}: Position=({self.position[0]:.3f}, {self.position[1]:.3f}), "
                  f"Distance={self.total_distance:.3f}, Energy={self.total_energy:.3f}")
        elif mode == 'rgb_array':
            # Could implement visual rendering here
            pass
    
    def close(self):
        """Clean up the environment."""
        pass

# Register the environment
if __name__ == "__main__":
    # Test the environment
    env = DummyMOEnv(obj_num=2)
    obs = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.3f}, objectives={info['obj']}, done={done}")
        if done:
            break
    
    env.close()
