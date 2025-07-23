#!/usr/bin/env python3
"""
CNN Architecture Examples for PGMORL
====================================

This script demonstrates how to configure different CNN architectures for 
image-based environments in PGMORL. It shows various presets and how to 
create custom architectures.
"""

import sys
import os

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from deep_sea_treasure_config import *

def demo_cnn_architectures():
    """Demonstrate different CNN architecture configurations."""
    
    print("=" * 80)
    print("CNN ARCHITECTURE EXAMPLES FOR PGMORL")
    print("=" * 80)
    
    # Default MLP (for 1D observations like Deep Sea Treasure)
    print("\n1. DEFAULT MLP ARCHITECTURE (for 1D observations):")
    print("-" * 50)
    print_network_summary()
    
    # Atari-style CNN
    print("\n2. ATARI-STYLE CNN:")
    print("-" * 50)
    global EXPECTED_OBS_SHAPE
    EXPECTED_OBS_SHAPE = (4, 84, 84)  # 4 stacked frames, 84x84 resolution
    apply_atari_cnn_config()
    print_network_summary()
    
    # MeltingPot-style CNN
    print("\n3. MELTINGPOT-STYLE CNN:")
    print("-" * 50)
    EXPECTED_OBS_SHAPE = (3, 88, 104)  # RGB, MeltingPot dimensions
    apply_meltingpot_cnn_config()
    print_network_summary()
    
    # Deep Nature CNN
    print("\n4. DEEP NATURE CNN:")
    print("-" * 50)
    EXPECTED_OBS_SHAPE = (4, 84, 84)
    apply_deep_nature_cnn_config()
    print_network_summary()
    
    # Lightweight CNN
    print("\n5. LIGHTWEIGHT CNN:")
    print("-" * 50)
    EXPECTED_OBS_SHAPE = (3, 64, 64)
    apply_lightweight_cnn_config()
    print_network_summary()

def demo_custom_cnn():
    """Demonstrate creating custom CNN architectures."""
    
    print("\n" + "=" * 80)
    print("CUSTOM CNN ARCHITECTURE EXAMPLES")
    print("=" * 80)
    
    # Custom architecture for high-resolution images
    print("\n1. HIGH-RESOLUTION CUSTOM CNN:")
    print("-" * 50)
    global EXPECTED_OBS_SHAPE
    EXPECTED_OBS_SHAPE = (3, 128, 128)
    
    create_custom_cnn_config([
        [32, 8, 4, 0],      # 32 filters, 8x8 kernel, stride 4 -> 31x31
        [64, 4, 2, 0],      # 64 filters, 4x4 kernel, stride 2 -> 14x14  
        [128, 3, 2, 1],     # 128 filters, 3x3 kernel, stride 2, padding 1 -> 7x7
        [256, 3, 1, 1]      # 256 filters, 3x3 kernel, stride 1, padding 1 -> 7x7
    ], cnn_hidden_size=2048, cnn_final_layers=[2048, 1024, 512])
    
    print_network_summary()
    
    # Custom architecture for small images
    print("\n2. SMALL IMAGE CUSTOM CNN:")
    print("-" * 50)
    EXPECTED_OBS_SHAPE = (1, 32, 32)  # Grayscale 32x32
    
    create_custom_cnn_config([
        [16, 5, 2, 2],      # 16 filters, 5x5 kernel, stride 2, padding 2 -> 16x16
        [32, 3, 2, 1],      # 32 filters, 3x3 kernel, stride 2, padding 1 -> 8x8
        [64, 3, 2, 1]       # 64 filters, 3x3 kernel, stride 2, padding 1 -> 4x4
    ], cnn_hidden_size=512, cnn_final_layers=[512, 256])
    
    print_network_summary()
    
    # Custom architecture with residual-like connections (simulated with more layers)
    print("\n3. DEEP CUSTOM CNN:")
    print("-" * 50)
    EXPECTED_OBS_SHAPE = (3, 96, 96)
    
    create_custom_cnn_config([
        [32, 7, 2, 3],      # 32 filters, 7x7 kernel, stride 2, padding 3 -> 48x48
        [64, 3, 1, 1],      # 64 filters, 3x3 kernel, stride 1, padding 1 -> 48x48
        [64, 3, 2, 1],      # 64 filters, 3x3 kernel, stride 2, padding 1 -> 24x24
        [128, 3, 1, 1],     # 128 filters, 3x3 kernel, stride 1, padding 1 -> 24x24
        [128, 3, 2, 1],     # 128 filters, 3x3 kernel, stride 2, padding 1 -> 12x12
        [256, 3, 2, 1]      # 256 filters, 3x3 kernel, stride 2, padding 1 -> 6x6
    ], cnn_hidden_size=1024, cnn_final_layers=[1024, 512, 256])
    
    print_network_summary()

def show_environment_examples():
    """Show examples for different types of environments."""
    
    print("\n" + "=" * 80)
    print("ENVIRONMENT-SPECIFIC ARCHITECTURE EXAMPLES")
    print("=" * 80)
    
    print("\n🏛️ DEEP SEA TREASURE (1D observations):")
    print("-" * 50)
    print("Observation: Agent position (x, y) -> Shape: (2,)")
    print("Recommended: MLP architecture")
    print("Configuration: Default NETWORK_CONFIG")
    global EXPECTED_OBS_SHAPE
    EXPECTED_OBS_SHAPE = (2,)
    NETWORK_CONFIG['use_custom_cnn'] = False
    print_network_summary()
    
    print("\n🎮 ATARI GAMES (Image observations):")
    print("-" * 50)
    print("Observation: Stacked frames (4, 84, 84)")
    print("Recommended: Atari CNN preset")
    EXPECTED_OBS_SHAPE = (4, 84, 84)
    apply_atari_cnn_config()
    print_network_summary()
    
    print("\n🌊 MELTINGPOT (RGB observations):")
    print("-" * 50)
    print("Observation: RGB images with sprites")
    print("Recommended: MeltingPot CNN preset")
    EXPECTED_OBS_SHAPE = (3, 88, 104)
    apply_meltingpot_cnn_config()
    print_network_summary()
    
    print("\n🤖 CUSTOM ROBOTICS (High-res RGB):")
    print("-" * 50)
    print("Observation: High-resolution RGB camera")
    print("Recommended: Custom deep CNN")
    EXPECTED_OBS_SHAPE = (3, 128, 128)
    create_custom_cnn_config([
        [32, 8, 4, 0], [64, 4, 2, 0], [128, 3, 2, 1], [256, 3, 1, 1]
    ], cnn_hidden_size=1024, cnn_final_layers=[1024, 512])
    print_network_summary()

def show_usage_examples():
    """Show how to use the configurations in training scripts."""
    
    print("\n" + "=" * 80)
    print("USAGE EXAMPLES")
    print("=" * 80)
    
    print("""
🔧 HOW TO USE THESE CONFIGURATIONS:

1. EDIT THE CONFIG FILE:
   ```python
   # In deep_sea_treasure_config.py
   NETWORK_CONFIG['use_custom_cnn'] = True
   apply_atari_cnn_config()  # Use preset
   ```

2. USE PRESET FUNCTIONS:
   ```python
   # In your training script or main()
   apply_atari_cnn_config()
   apply_meltingpot_cnn_config()
   apply_deep_nature_cnn_config()
   apply_lightweight_cnn_config()
   ```

3. CREATE CUSTOM ARCHITECTURES:
   ```python
   create_custom_cnn_config([
       [32, 8, 4, 0],  # [channels, kernel_size, stride, padding]
       [64, 4, 2, 0],
       [128, 3, 1, 1]
   ], cnn_hidden_size=512, cnn_final_layers=[512, 256])
   ```

4. ENVIRONMENT-SPECIFIC SETUP:
   ```python
   # For Atari games
   ENV_NAME = 'BreakoutNoFrameskip-v4'
   apply_atari_cnn_config()
   
   # For MeltingPot
   ENV_NAME = 'bach_or_stravinsky_in_the_matrix__repeated'
   apply_meltingpot_cnn_config()
   
   # For custom environments
   create_custom_cnn_config(your_layers)
   ```

5. ARCHITECTURE PARAMETERS:
   Layer format: [out_channels, kernel_size, stride, padding]
   - out_channels: Number of output feature maps
   - kernel_size: Size of convolution kernel (int or [h, w])
   - stride: Stride of convolution
   - padding: Padding around input

📊 TIPS FOR DESIGNING ARCHITECTURES:
- Start with proven architectures (Atari, Nature CNN)
- Reduce channels for smaller/simpler images
- Increase depth for complex visual patterns
- Use stride=2 to reduce spatial dimensions
- Add padding to preserve spatial information
- Final spatial size should be small (e.g., 7x7 or smaller)

🎯 MULTI-OBJECTIVE SUPPORT:
All CNN architectures automatically support multi-objective learning
when obj_num > 1. The critic will output multiple values.

⚡ PERFORMANCE CONSIDERATIONS:
- Larger networks = more parameters = slower training
- Deeper networks = better features but harder to train
- Use lightweight presets for fast experimentation
- Use deep presets for final performance runs
""")

if __name__ == "__main__":
    demo_cnn_architectures()
    demo_custom_cnn()
    show_environment_examples()
    show_usage_examples()
    
    print("\n" + "=" * 80)
    print("CNN ARCHITECTURE DEMO COMPLETED")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Choose an appropriate architecture for your environment")
    print("2. Edit deep_sea_treasure_config.py with your choice")
    print("3. Run your training script")
    print("4. Experiment with different architectures for best performance")
