#!/usr/bin/env python3
"""
Test script to check weight generation in MORL
"""

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'morl'))

from utils import generate_weights_batch_dfs

def test_weight_generation():
    print("Testing weight generation...")
    
    # Test parameters from example_training.py
    obj_num = 2
    min_weight = 0.0
    max_weight = 1.0
    delta_weight = 0.2
    
    weights_batch = []
    generate_weights_batch_dfs(0, obj_num, min_weight, max_weight, delta_weight, [], weights_batch)
    
    print(f"Generated {len(weights_batch)} weight combinations:")
    for i, weights in enumerate(weights_batch):
        print(f"  {i}: {weights} (sum={sum(weights):.3f})")
    
    # Test with smaller delta_weight
    print(f"\nTesting with smaller delta_weight (0.1):")
    weights_batch_small = []
    generate_weights_batch_dfs(0, obj_num, min_weight, max_weight, 0.1, [], weights_batch_small)
    
    print(f"Generated {len(weights_batch_small)} weight combinations:")
    for i, weights in enumerate(weights_batch_small):
        print(f"  {i}: {weights} (sum={sum(weights):.3f})")

if __name__ == "__main__":
    test_weight_generation()
