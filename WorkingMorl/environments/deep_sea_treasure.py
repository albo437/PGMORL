# Deep Sea Treasure Environment
# A classic multi-objective reinforcement learning benchmark
# Redesigned for PGMORL compatibility (ALL OBJECTIVES MUST BE POSITIVE)

import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt

class DeepSeaTreasureEnv(gym.Env):
    """
    Deep Sea Treasure Environment - PGMORL Compatible Version
    
    A submarine navigates a grid to collect treasures while managing fuel efficiency.
    
    Key Changes for PGMORL:
    - ALL objectives are POSITIVE (algorithm maximizes everything)
    - Step-based objectives (not cumulative)
    - Clear trade-offs between objectives
    
    Objectives:
    1. Treasure collection rate (positive step rewards)
    2. Fuel efficiency (positive, decreases with time)
    
    The environment creates natural Pareto trade-offs.
    """
    
    def __init__(self, max_steps=50):  # Much shorter episodes!
        super(DeepSeaTreasureEnv, self).__init__()
        
        self.max_steps = max_steps
        self.current_step = 0
        
        # Simplified but effective treasure layout
        self.height, self.width = 11, 11
        
        # Initialize map: 0 = water, -10 = wall, positive = treasure
        self.treasure_map = np.full((self.height, self.width), -10.0)  # Start with walls
        
        # Create the classic Deep Sea Treasure water areas
        # Surface row is all water
        self.treasure_map[0, :] = 0.0
        
        # Create diagonal water pattern (upper triangle)
        for row in range(self.height):
            for col in range(self.width):
                if row <= col:  # Upper triangle and diagonal are water
                    self.treasure_map[row, col] = 0.0
        
        # Use new Gym API by default (required for MeltingPot compatibility)
        self._new_step_api = True
        
        # Place treasures at strategic locations for clear Pareto front
        # Close treasures = low value, far treasures = high value
        self.treasure_map[1, 0] = 1.0    # Easy treasure (1 step down)
        self.treasure_map[2, 1] = 3.0    # Close medium treasure 
        self.treasure_map[4, 3] = 8.0    # Medium treasure
        self.treasure_map[7, 6] = 20.0   # Hard treasure
        
        # Action space: 0=up, 1=down, 2=left, 3=right
        self.action_space = spaces.Discrete(4)
        
        # Observation space: [row, col, remaining_steps_normalized] 
        self.observation_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(3,),
            dtype=np.float32
        )
        
        # Starting position
        self.start_pos = (0, 0)
        self.pos = None
        
        # Track treasure collection for this episode
        self.total_treasure_collected = 0.0
        self.steps_taken = 0
        
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        if seed is not None:
            self.seed(seed)
            
        self.current_step = 0
        self.pos = list(self.start_pos)
        self.total_treasure_collected = 0.0
        self.steps_taken = 0
        
        # Reset treasure map to original state
        self.treasure_map = np.full((self.height, self.width), -10.0)
        
        # Recreate water areas
        self.treasure_map[0, :] = 0.0
        for row in range(self.height):
            for col in range(self.width):
                if row <= col:
                    self.treasure_map[row, col] = 0.0
        
        # Replace treasures
        self.treasure_map[1, 0] = 1.0
        self.treasure_map[2, 1] = 3.0
        self.treasure_map[4, 3] = 8.0
        self.treasure_map[7, 6] = 20.0
        
        observation = self._get_observation()
        info = {'position': self.pos.copy(), 'step': self.current_step}
        
        # Always use new Gym API format (observation, info)
        return observation, info
    
    def step(self, action):
        """Execute an action and return the results."""
        # Check for valid action
        if action not in range(4):
            action = 0  # Default to "up" if invalid
        
        self.current_step += 1
        old_pos = self.pos.copy()
        
        # Execute action
        if action == 0:  # up
            if self.pos[0] > 0 and self.treasure_map[self.pos[0] - 1, self.pos[1]] != -10:
                self.pos[0] -= 1
        elif action == 1:  # down
            if self.pos[0] < self.height - 1 and self.treasure_map[self.pos[0] + 1, self.pos[1]] != -10:
                self.pos[0] += 1
        elif action == 2:  # left
            if self.pos[1] > 0 and self.treasure_map[self.pos[0], self.pos[1] - 1] != -10:
                self.pos[1] -= 1
        elif action == 3:  # right
            if self.pos[1] < self.width - 1 and self.treasure_map[self.pos[0], self.pos[1] + 1] != -10:
                self.pos[1] += 1
        
        # Calculate rewards (PGMORL-compatible: positive objectives)
        current_cell = self.treasure_map[self.pos[0], self.pos[1]]
        
        # Scalar reward: Just treasure collection
        if current_cell > 0:  # Found treasure
            treasure_reward = current_cell  # Direct value
            self.treasure_map[self.pos[0], self.pos[1]] = 0.0  # Remove collected treasure
        else:
            treasure_reward = 0.0
        
        # Multi-objective format for PGMORL
        # Objective 1: Treasure value (only when collected)
        obj1 = treasure_reward
        
        # Objective 2: Fuel efficiency = 10/(total_steps + 1) - only at episode end
        if treasure_reward > 0:  # Episode ends when treasure found
            obj2 = 10.0 / (self.current_step + 1)
        else:
            obj2 = 0.0  # No efficiency reward during intermediate steps
        
        # Objectives array for PGMORL
        objectives = np.array([obj1, obj2])
        
        # Check termination conditions - episode ends when treasure is collected OR max steps reached
        treasure_found = treasure_reward > 0
        terminated = bool(treasure_found or self.current_step >= self.max_steps)  # Use bool() to avoid np.bool8 warning
        
        # Create info dictionary
        info = {
            'position': self.pos.copy(),
            'step': self.current_step,
            'treasure_collected': treasure_found,
            'old_position': old_pos,
            'obj': objectives  # Required by PGMORL for multi-objective rewards
        }
        
        # Always use new Gym API format (5 values)
        truncated = False  # We don't use truncation in this environment
        
        # Return scalar reward (just treasure value) and objectives in info
        return self._get_observation(), treasure_reward, terminated, truncated, info
    
    def _get_observation(self):
        """Get the current observation - normalized for better learning."""
        # Normalize position and time for consistent scales
        row_norm = self.pos[0] / (self.height - 1)
        col_norm = self.pos[1] / (self.width - 1)
        time_norm = self.current_step / self.max_steps
        
        return np.array([row_norm, col_norm, time_norm], dtype=np.float32)
    
    def render(self, mode='human'):
        """Render the environment."""
        if mode == 'human':
            print(f"\\nStep {self.current_step}")
            print(f"Position: ({self.pos[0]}, {self.pos[1]})")
            print(f"Total treasure collected: {self.total_treasure_collected}")
            print(f"Steps taken: {self.steps_taken}")
            
            # Create visual representation
            visual = np.zeros((self.height, self.width), dtype=str)
            for i in range(self.height):
                for j in range(self.width):
                    cell_val = self.treasure_map[i, j]
                    if cell_val == -10:
                        visual[i, j] = '#'  # Wall
                    elif cell_val > 0:
                        visual[i, j] = 'T'  # Treasure
                    else:
                        visual[i, j] = '.'  # Water
            
            # Mark submarine position
            visual[self.pos[0], self.pos[1]] = 'S'
            
            print("Map (S=submarine, T=treasure, #=wall, .=water):")
            for row in visual:
                print(' '.join(row))
                
        elif mode == 'rgb_array':
            # Could implement graphical rendering here
            pass
    
    def close(self):
        """Clean up the environment."""
        pass
    
    def seed(self, seed=None):
        """Set the random seed for the environment."""
        if seed is not None:
            np.random.seed(seed)
        return [seed]
    
    def plot_treasure_map(self):
        """Plot the treasure map for visualization."""
        plt.figure(figsize=(10, 10))
        
        # Create color map
        display_map = self.treasure_map.copy()
        
        # Create custom colormap
        plt.imshow(display_map, cmap='RdYlBu_r', interpolation='nearest')
        plt.colorbar(label='Treasure Value')
        
        # Add treasure value annotations
        for i in range(self.height):
            for j in range(self.width):
                value = self.treasure_map[i, j]
                if value > 0:
                    plt.text(j, i, f'{value}', ha='center', va='center', 
                            color='white', fontweight='bold', fontsize=10)
                elif value == -10:
                    plt.text(j, i, 'WALL', ha='center', va='center', 
                            color='white', fontweight='bold', fontsize=8)
        
        # Mark starting position
        plt.plot(self.start_pos[1], self.start_pos[0], 'go', markersize=15, 
                label='Start Position')
        
        plt.title('Deep Sea Treasure Map')
        plt.xlabel('Column')
        plt.ylabel('Row (Depth)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# Register the environment variants
def register_environments():
    """Register Deep Sea Treasure environment variants."""
    
    # Check if already registered to avoid conflicts
    try:
        # Standard version
        gym.register(
            id='MO-DeepSeaTreasure-v0',
            entry_point='environments.deep_sea_treasure:DeepSeaTreasureEnv',
            kwargs={'max_steps': 50},  # Shorter episodes
            max_episode_steps=50
        )
    except gym.error.Error:
        pass  # Already registered
    
    try:
        # Shorter episodes for faster training
        gym.register(
            id='MO-DeepSeaTreasure-Short-v0', 
            entry_point='environments.deep_sea_treasure:DeepSeaTreasureEnv',
            kwargs={'max_steps': 30},  # Even shorter
            max_episode_steps=30
        )
    except gym.error.Error:
        pass  # Already registered
    
    try:
        # Longer episodes for thorough exploration
        gym.register(
            id='MO-DeepSeaTreasure-Long-v0',
            entry_point='environments.deep_sea_treasure:DeepSeaTreasureEnv', 
            kwargs={'max_steps': 100},  # Still reasonable
            max_episode_steps=100
        )
    except gym.error.Error:
        pass  # Already registered

# Only auto-register when run as main script, not when imported
if __name__ == "__main__":
    register_environments()
else:
    # Auto-register when imported (needed for training scripts)
    register_environments()

if __name__ == "__main__":
    # Register environments for this test
    register_environments()
    
    # Test the environment
    env = DeepSeaTreasureEnv()
    
    print("Deep Sea Treasure Environment Test")
    print("=" * 40)
    
    # Show the treasure map
    env.plot_treasure_map()
    
    # Run a simple episode
    obs = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test some actions
    actions = [1, 3, 1, 3, 1]  # down, right, down, right, down
    
    for i, action in enumerate(actions):
        print(f"\\n--- Step {i+1} ---")
        print(f"Action: {['up', 'down', 'left', 'right'][action]}")
        
        obs, reward, done, info = env.step(action)
        
        print(f"New observation: {obs}")
        print(f"Reward: {reward:.3f}")
        print(f"Objectives: {info['obj']}")
        print(f"Done: {done}")
        
        env.render()
        
        if done:
            print("Episode finished!")
            break
    
    env.close()
    print("\\nEnvironment test completed!")
