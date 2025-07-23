# 🧠 CNN Architecture System for PGMORL

## ✅ What We've Built

### **1. Flexible CNN Architecture System**
Added support for completely configurable CNN architectures in PGMORL:

- **FlexibleCNNBase** - Custom CNN class with configurable layers
- **MOFlexibleCNNBase** - Multi-objective version for MORL
- **Automatic architecture selection** based on observation space
- **Multiple CNN presets** for different use cases

### **2. Configuration System**
Extended the configuration file with CNN support:

```python
# CNN Configuration in deep_sea_treasure_config.py
NETWORK_CONFIG = {
    'use_custom_cnn': True,
    'cnn_layers': [[32, 8, 4, 0], [64, 4, 2, 0], [64, 3, 1, 0]],
    'cnn_hidden_size': 512,
    'cnn_final_layers': [512],
    'cnn_activation': 'relu'
}
```

### **3. Built-in CNN Presets**
Ready-to-use architectures for common scenarios:

| Preset | Use Case | Architecture |
|--------|----------|--------------|
| `atari` | Atari games (84x84) | 3 conv layers → 512 FC |
| `meltingpot` | MeltingPot sprites | 2 conv layers → 256 FC |
| `deep_nature` | Complex images | 4 conv layers → 1024 FC |
| `lightweight` | Small/fast training | 2 conv layers → 256 FC |

### **4. Integration with PGMORL**
- **Modified Policy class** to support flexible CNN selection
- **Updated warm_up.py** to use configuration-based network creation
- **Added command line arguments** for CNN control
- **Automatic multi-objective support** for all architectures

## 🎯 How to Use

### **Method 1: Configuration File**
```python
# Edit deep_sea_treasure_config.py
apply_atari_cnn_config()  # Use preset

# Or configure manually
NETWORK_CONFIG['use_custom_cnn'] = True
NETWORK_CONFIG['cnn_layers'] = [[32, 8, 4, 0], [64, 4, 2, 0]]
```

### **Method 2: Preset Functions**
```python
# In your training script
from deep_sea_treasure_config import *

apply_atari_cnn_config()        # For Atari-style games
apply_meltingpot_cnn_config()   # For MeltingPot environments  
apply_deep_nature_cnn_config()  # For complex visual tasks
apply_lightweight_cnn_config()  # For fast experimentation
```

### **Method 3: Custom Architectures**
```python
create_custom_cnn_config([
    [32, 8, 4, 0],    # 32 filters, 8x8 kernel, stride 4, padding 0
    [64, 4, 2, 0],    # 64 filters, 4x4 kernel, stride 2, padding 0
    [128, 3, 1, 1]    # 128 filters, 3x3 kernel, stride 1, padding 1
], cnn_hidden_size=512, cnn_final_layers=[512, 256])
```

### **Method 4: Command Line**
```bash
# Automatic detection based on environment
python example_flexible_training.py --env BreakoutNoFrameskip-v4

# Explicit CNN preset
python example_flexible_training.py --env MyEnv-v0 --cnn-preset atari

# Custom CNN
python example_flexible_training.py --env MyEnv-v0 --custom-cnn
```

## 📋 Architecture Specification Format

### **CNN Layer Format:**
```python
[out_channels, kernel_size, stride, padding]
```

- **out_channels**: Number of output feature maps (e.g., 32, 64, 128)
- **kernel_size**: Convolution kernel size (int or [height, width])
- **stride**: Stride of convolution (reduces spatial dimensions)
- **padding**: Padding around input (preserves spatial dimensions)

### **Example Architectures:**

**Atari-style (Nature CNN):**
```python
cnn_layers = [
    [32, 8, 4, 0],   # 32 filters, 8x8 kernel, stride 4 → reduce by 4x
    [64, 4, 2, 0],   # 64 filters, 4x4 kernel, stride 2 → reduce by 2x  
    [64, 3, 1, 0]    # 64 filters, 3x3 kernel, stride 1 → same size
]
```

**High-resolution images:**
```python
cnn_layers = [
    [32, 8, 4, 0],    # Initial large reduction
    [64, 4, 2, 0],    # Medium reduction
    [128, 3, 2, 1],   # Small reduction with padding
    [256, 3, 1, 1]    # Feature refinement with padding
]
```

**Small images (32x32):**
```python
cnn_layers = [
    [16, 5, 2, 2],    # 16 filters, 5x5 kernel, stride 2, padding 2
    [32, 3, 2, 1],    # 32 filters, 3x3 kernel, stride 2, padding 1
    [64, 3, 2, 1]     # 64 filters, 3x3 kernel, stride 2, padding 1
]
```

## 🔧 Files Modified/Created

### **New Files:**
- `cnn_demo.py` - CNN architecture demonstration
- `example_flexible_training.py` - Flexible training script

### **Modified Files:**
- `externals/pytorch-a2c-ppo-acktr-gail/a2c_ppo_acktr/model.py` - Added FlexibleCNNBase
- `deep_sea_treasure_config.py` - Added CNN configuration
- `morl/warm_up.py` - Updated to use base_kwargs
- `morl/arguments.py` - Added CNN command line arguments

## 🎮 Environment Examples

### **1D Observations (Vector-based):**
```python
# Deep Sea Treasure, CartPole, etc.
obs_shape = (4,)  # Agent state vector
# → Automatically uses MLP architecture
```

### **3D Observations (Image-based):**
```python
# Atari games
obs_shape = (4, 84, 84)  # 4 stacked grayscale frames
apply_atari_cnn_config()

# MeltingPot 
obs_shape = (3, 88, 104)  # RGB sprites
apply_meltingpot_cnn_config()

# Custom high-res
obs_shape = (3, 128, 128)  # RGB camera
create_custom_cnn_config(custom_layers)
```

## ⚡ Performance Considerations

### **Architecture Size vs Performance:**
- **Lightweight**: Fast training, basic features
- **Atari**: Good balance for most games  
- **Deep Nature**: Best features, slower training
- **Custom**: Tailored to your specific needs

### **Tips for Good Performance:**
1. **Start with proven architectures** (Atari, Nature CNN)
2. **Match network size to problem complexity**
3. **Use stride=2 to reduce spatial dimensions efficiently**
4. **Keep final spatial size small** (7x7 or smaller)
5. **Add more channels for complex visual patterns**

## 🎯 Multi-Objective Support

All CNN architectures automatically support multi-objective learning:

```python
# Single objective (obj_num=1)
critic_output = [value]  # Shape: (batch_size, 1)

# Multi-objective (obj_num=2+) 
critic_output = [obj1, obj2, ...]  # Shape: (batch_size, obj_num)
```

## 🚀 Next Steps

1. **Choose appropriate architecture** for your environment
2. **Start with presets** for quick experimentation
3. **Create custom architectures** for specific needs
4. **Tune hyperparameters** based on results
5. **Scale up** for final performance runs

## 🔍 Demo Scripts

```bash
# See all CNN architecture examples
python cnn_demo.py

# Try flexible training with different environments
python example_flexible_training.py --env YourEnv-v0 --cnn-preset atari
```

---

**You now have complete control over the neural network architecture in PGMORL!** 🎉

The system automatically detects whether your environment uses vector or image observations and applies the appropriate network type. You can easily experiment with different architectures without modifying the core training code.
