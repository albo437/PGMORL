"""
Custom Neural Network Architectures for PGMORL
==============================================

This module provides flexible neural network architectures that can be configured
through the configuration system, including custom CNN architectures.
"""

import torch
import torch.nn as nn
import numpy as np

# Import from the pytorch a2c-ppo-acktr-gail external library
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'externals', 'pytorch-a2c-ppo-acktr-gail'))

from a2c_ppo_acktr.utils import init

class Flatten(nn.Module):
    """Flattens input tensor except batch dimension."""
    def forward(self, x):
        return x.view(x.size(0), -1)

class CustomCNNBase(nn.Module):
    """
    Flexible CNN base class that can be configured with arbitrary CNN architectures.
    
    This class supports:
    - Any number of convolutional layers
    - Configurable kernel sizes, strides, and padding
    - Multiple fully connected layers after feature extraction
    - Multi-objective outputs
    """
    
    def __init__(self, num_inputs, recurrent=False, cnn_layers=None, 
                 cnn_hidden_size=512, cnn_final_layers=None, cnn_activation='relu',
                 obj_num=1, layernorm=False):
        super(CustomCNNBase, self).__init__()
        
        self.is_recurrent = recurrent
        self.recurrent_hidden_state_size = cnn_hidden_size if recurrent else 1
        self._hidden_size = cnn_final_layers[-1] if cnn_final_layers else cnn_hidden_size
        self.obj_num = obj_num
        
        # Default CNN layers if none provided (Atari-style)
        if cnn_layers is None:
            cnn_layers = [[32, 8, 4, 0], [64, 4, 2, 0], [64, 3, 1, 0]]
        
        if cnn_final_layers is None:
            cnn_final_layers = [cnn_hidden_size]
        
        # Initialize conv layers
        init_relu = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                                   constant_(x, 0), nn.init.calculate_gain('relu'))
        init_tanh = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                                   constant_(x, 0), np.sqrt(2))
        
        # Build convolutional layers
        conv_layers = []
        in_channels = num_inputs
        
        for i, (out_channels, kernel_size, stride, padding) in enumerate(cnn_layers):
            # Handle different kernel_size formats
            if isinstance(kernel_size, list):
                kernel_size = tuple(kernel_size)
            elif isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)
            
            conv_layers.extend([
                init_relu(nn.Conv2d(in_channels, out_channels, kernel_size, 
                                   stride=stride, padding=padding)),
                nn.ReLU() if cnn_activation == 'relu' else nn.Tanh()
            ])
            in_channels = out_channels
        
        conv_layers.append(Flatten())
        self.conv_net = nn.Sequential(*conv_layers)
        
        # Calculate the size of flattened features
        # We need to do a forward pass to determine this
        with torch.no_grad():
            # Assume input is 84x84 (common for Atari) or adapt based on environment
            dummy_input = torch.zeros(1, num_inputs, 84, 84)
            conv_output = self.conv_net(dummy_input)
            conv_output_size = conv_output.shape[1]
        
        # Build fully connected layers
        fc_layers = []
        fc_sizes = [conv_output_size] + cnn_final_layers
        
        for i in range(len(fc_sizes) - 1):
            fc_layers.extend([
                init_relu(nn.Linear(fc_sizes[i], fc_sizes[i + 1])),
                nn.ReLU() if cnn_activation == 'relu' else nn.Tanh()
            ])
            
            if layernorm:
                fc_layers.append(nn.LayerNorm(fc_sizes[i + 1]))
        
        self.fc_net = nn.Sequential(*fc_layers)
        
        # GRU for recurrent processing (if needed)
        if recurrent:
            self.gru = nn.GRU(self._hidden_size, cnn_hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)
        
        # Output layers
        init_output = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                                    constant_(x, 0))
        
        # Multi-objective critic (for MORL)
        if obj_num > 1:
            self.critic_linear = init_output(nn.Linear(self._hidden_size, obj_num))
        else:
            self.critic_linear = init_output(nn.Linear(self._hidden_size, 1))
        
        self.train()
    
    @property
    def output_size(self):
        return self._hidden_size
    
    def _forward_gru(self, x, hxs, masks):
        """Forward pass through GRU for recurrent processing."""
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs
    
    def forward(self, inputs, rnn_hxs, masks):
        # Extract features with CNN
        x = self.conv_net(inputs / 255.0)  # Normalize image inputs
        
        # Process through fully connected layers
        x = self.fc_net(x)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MOCustomCNNBase(CustomCNNBase):
    """Multi-objective version of CustomCNNBase."""
    
    def __init__(self, num_inputs, obj_num=2, **kwargs):
        kwargs['obj_num'] = obj_num
        super(MOCustomCNNBase, self).__init__(num_inputs, **kwargs)


def calculate_conv_output_size(input_size, kernel_size, stride, padding):
    """Calculate output size after convolution."""
    if isinstance(input_size, int):
        input_size = (input_size, input_size)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    
    h_out = (input_size[0] + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
    w_out = (input_size[1] + 2 * padding[1] - kernel_size[1]) // stride[1] + 1
    
    return (h_out, w_out)


def estimate_conv_output_size(input_shape, cnn_layers):
    """
    Estimate the total flattened size after all convolutional layers.
    
    Args:
        input_shape: (channels, height, width)
        cnn_layers: List of [out_channels, kernel_size, stride, padding]
    
    Returns:
        Total flattened size
    """
    _, h, w = input_shape
    
    for out_channels, kernel_size, stride, padding in cnn_layers:
        h, w = calculate_conv_output_size((h, w), kernel_size, stride, padding)
        if h <= 0 or w <= 0:
            raise ValueError(f"Invalid conv layer configuration - output size becomes {h}x{w}")
    
    # Final channels is the last layer's output channels
    final_channels = cnn_layers[-1][0]
    
    return final_channels * h * w


def create_cnn_architecture_from_config(config):
    """
    Create CNN architecture parameters from configuration.
    
    Args:
        config: NETWORK_CONFIG dictionary
    
    Returns:
        Dictionary with CNN parameters
    """
    if config.get('use_custom_cnn', False):
        return {
            'cnn_layers': config['cnn_layers'],
            'cnn_hidden_size': config['cnn_hidden_size'],
            'cnn_final_layers': config['cnn_final_layers'],
            'cnn_activation': config.get('cnn_activation', 'relu'),
            'layernorm': config.get('layernorm', False)
        }
    else:
        # Use default CNN (original CNNBase style)
        return {
            'cnn_layers': [[32, 8, 4, 0], [64, 4, 2, 0], [64, 3, 1, 0]],
            'cnn_hidden_size': 512,
            'cnn_final_layers': [512],
            'cnn_activation': 'relu',
            'layernorm': config.get('layernorm', False)
        }
