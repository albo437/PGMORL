#!/usr/bin/env python3
"""
Results analysis script for MORL training outputs.
This script helps analyze and visualize the training results.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

def analyze_results(results_dir):
    """Analyze MORL training results."""
    
    if not os.path.exists(results_dir):
        print(f"Results directory {results_dir} not found!")
        return False
    
    print(f"Analyzing results in: {results_dir}")
    print("=" * 60)
    
    # Find all iteration directories
    iteration_dirs = []
    for item in os.listdir(results_dir):
        if item.isdigit():
            iteration_dirs.append(int(item))
    
    iteration_dirs.sort()
    print(f"Found {len(iteration_dirs)} iteration directories: {iteration_dirs}")
    
    if not iteration_dirs:
        print("No iteration directories found!")
        return False
    
    # Analyze each iteration
    total_elites = 0
    total_offsprings = 0
    
    for iteration in iteration_dirs:
        iter_dir = os.path.join(results_dir, str(iteration))
        print(f"\n--- Iteration {iteration} ---")
        
        # Check elites
        elites_file = os.path.join(iter_dir, 'elites', 'elites.txt')
        if os.path.exists(elites_file):
            with open(elites_file, 'r') as f:
                elites_lines = f.readlines()
            elite_count = len([line for line in elites_lines if line.strip()])
            total_elites += elite_count
            print(f"  Elites: {elite_count}")
            
            if elite_count > 0:
                # Parse elite objectives
                elite_objs = []
                for line in elites_lines:
                    if line.strip():
                        objs = [float(x) for x in line.strip().split(',')]
                        elite_objs.append(objs)
                print(f"    Elite objectives range: {np.min(elite_objs, axis=0)} to {np.max(elite_objs, axis=0)}")
        
        # Check offsprings
        offspring_file = os.path.join(iter_dir, 'elites', 'offsprings.txt')
        if os.path.exists(offspring_file):
            with open(offspring_file, 'r') as f:
                offspring_lines = f.readlines()
            offspring_count = len([line for line in offspring_lines if line.strip()])
            total_offsprings += offspring_count
            print(f"  Offsprings: {offspring_count}")
        
        # Check optimization graph
        optgraph_file = os.path.join(iter_dir, 'population', 'optgraph.txt')
        if os.path.exists(optgraph_file):
            with open(optgraph_file, 'r') as f:
                optgraph_lines = f.readlines()
            optgraph_count = len([line for line in optgraph_lines if line.strip() and not line.strip().isdigit()])
            print(f"  OptGraph entries: {optgraph_count}")
    
    print(f"\n--- Summary ---")
    print(f"Total elites across all iterations: {total_elites}")
    print(f"Total offsprings across all iterations: {total_offsprings}")
    
    # Check final results
    final_dir = os.path.join(results_dir, 'final')
    if os.path.exists(final_dir):
        print(f"\n--- Final Results ---")
        
        # Count final policies
        policy_files = [f for f in os.listdir(final_dir) if f.startswith('EP_policy_')]
        print(f"Final policies saved: {len(policy_files)}")
        
        # Check final objectives
        final_objs_file = os.path.join(final_dir, 'objs.txt')
        if os.path.exists(final_objs_file):
            with open(final_objs_file, 'r') as f:
                final_objs_lines = f.readlines()
            print(f"Final objective entries: {len(final_objs_lines)}")
            
            if len(final_objs_lines) > 0:
                # Parse and show final Pareto front
                final_objs = []
                for line in final_objs_lines:
                    if line.strip():
                        objs = [float(x) for x in line.strip().split(',')]
                        final_objs.append(objs)
                
                if final_objs:
                    final_objs = np.array(final_objs)
                    print(f"Final Pareto front:")
                    print(f"  Objectives shape: {final_objs.shape}")
                    print(f"  Objective ranges:")
                    for i in range(final_objs.shape[1]):
                        print(f"    Obj {i+1}: [{np.min(final_objs[:, i]):.3f}, {np.max(final_objs[:, i]):.3f}]")
                    
                    # Create visualizations
                    create_visualizations(final_objs, results_dir)
    
    return True

def create_visualizations(objectives, results_dir):
    """Create visualizations of the Pareto front."""
    
    print(f"\n--- Creating Visualizations ---")
    
    # Set up matplotlib style
    plt.style.use('default')
    
    num_objectives = objectives.shape[1]
    num_points = objectives.shape[0]
    
    print(f"Creating plots for {num_objectives}D Pareto front with {num_points} points...")
    
    if num_objectives == 2:
        # 2D Pareto front
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        scatter = ax.scatter(objectives[:, 0], objectives[:, 1], 
                           c=range(len(objectives)), cmap='viridis', 
                           alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('Objective 1', fontsize=12)
        ax.set_ylabel('Objective 2', fontsize=12)
        ax.set_title(f'2D Pareto Front ({num_points} points)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Policy Index', fontsize=10)
        
        # Save plot
        plot_path = os.path.join(results_dir, 'pareto_front_2d.png')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"2D Pareto front saved: {plot_path}")
        
    elif num_objectives == 3:
        # 3D Pareto front
        fig = plt.figure(figsize=(15, 5))
        
        # 3D scatter plot
        ax1 = fig.add_subplot(131, projection='3d')
        scatter = ax1.scatter(objectives[:, 0], objectives[:, 1], objectives[:, 2],
                            c=range(len(objectives)), cmap='viridis',
                            alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
        
        ax1.set_xlabel('Objective 1', fontsize=10)
        ax1.set_ylabel('Objective 2', fontsize=10)
        ax1.set_zlabel('Objective 3', fontsize=10)
        ax1.set_title(f'3D Pareto Front\n({num_points} points)', fontsize=12, fontweight='bold')
        
        # 2D projections
        ax2 = fig.add_subplot(132)
        scatter2 = ax2.scatter(objectives[:, 0], objectives[:, 1],
                             c=range(len(objectives)), cmap='viridis',
                             alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
        ax2.set_xlabel('Objective 1', fontsize=10)
        ax2.set_ylabel('Objective 2', fontsize=10)
        ax2.set_title('Projection: Obj1 vs Obj2', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        ax3 = fig.add_subplot(133)
        scatter3 = ax3.scatter(objectives[:, 0], objectives[:, 2],
                             c=range(len(objectives)), cmap='viridis',
                             alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
        ax3.set_xlabel('Objective 1', fontsize=10)
        ax3.set_ylabel('Objective 3', fontsize=10)
        ax3.set_title('Projection: Obj1 vs Obj3', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = fig.colorbar(scatter, ax=[ax1, ax2, ax3], shrink=0.8, aspect=20, pad=0.1)
        cbar.set_label('Policy Index', fontsize=10)
        
        plt.tight_layout()
        plot_path = os.path.join(results_dir, 'pareto_front_3d.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"3D Pareto front saved: {plot_path}")
        
        # Additional detailed 2D plots
        fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # All 2D projections with better formatting
        scatter1 = ax1.scatter(objectives[:, 0], objectives[:, 1], 
                              c=range(len(objectives)), cmap='plasma',
                              alpha=0.8, s=60, edgecolors='black', linewidth=0.5)
        ax1.set_xlabel('Objective 1', fontsize=11)
        ax1.set_ylabel('Objective 2', fontsize=11)
        ax1.set_title('Objective 1 vs Objective 2', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        scatter2 = ax2.scatter(objectives[:, 0], objectives[:, 2], 
                              c=range(len(objectives)), cmap='plasma',
                              alpha=0.8, s=60, edgecolors='black', linewidth=0.5)
        ax2.set_xlabel('Objective 1', fontsize=11)
        ax2.set_ylabel('Objective 3', fontsize=11)
        ax2.set_title('Objective 1 vs Objective 3', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        scatter3 = ax3.scatter(objectives[:, 1], objectives[:, 2], 
                              c=range(len(objectives)), cmap='plasma',
                              alpha=0.8, s=60, edgecolors='black', linewidth=0.5)
        ax3.set_xlabel('Objective 2', fontsize=11)
        ax3.set_ylabel('Objective 3', fontsize=11)
        ax3.set_title('Objective 2 vs Objective 3', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Objective distributions
        ax4.hist([objectives[:, 0], objectives[:, 1], objectives[:, 2]], 
                bins=15, alpha=0.7, label=['Obj 1', 'Obj 2', 'Obj 3'],
                color=['red', 'green', 'blue'])
        ax4.set_xlabel('Objective Value', fontsize=11)
        ax4.set_ylabel('Frequency', fontsize=11)
        ax4.set_title('Objective Value Distributions', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path2 = os.path.join(results_dir, 'pareto_projections_3d.png')
        plt.savefig(plot_path2, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"3D projections saved: {plot_path2}")
        
    elif num_objectives > 3:
        # High-dimensional visualization using parallel coordinates
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Parallel coordinates plot
        for i in range(len(objectives)):
            ax1.plot(range(num_objectives), objectives[i], 
                    alpha=0.6, linewidth=1, color=plt.cm.viridis(i/len(objectives)))
        
        ax1.set_xlabel('Objective Index', fontsize=12)
        ax1.set_ylabel('Objective Value', fontsize=12)
        ax1.set_title(f'{num_objectives}D Pareto Front - Parallel Coordinates ({num_points} points)', 
                     fontsize=14, fontweight='bold')
        ax1.set_xticks(range(num_objectives))
        ax1.set_xticklabels([f'Obj {i+1}' for i in range(num_objectives)])
        ax1.grid(True, alpha=0.3)
        
        # Pairwise scatter plot matrix (first 4 objectives if more than 4)
        max_obj_show = min(4, num_objectives)
        n_pairs = max_obj_show * (max_obj_show - 1) // 2
        
        if n_pairs > 0:
            pair_idx = 0
            colors = plt.cm.Set1(np.linspace(0, 1, n_pairs))
            
            for i in range(max_obj_show):
                for j in range(i+1, max_obj_show):
                    ax2.scatter(objectives[:, i], objectives[:, j], 
                              alpha=0.6, s=30, label=f'Obj{i+1} vs Obj{j+1}',
                              color=colors[pair_idx % len(colors)])
                    pair_idx += 1
            
            ax2.set_xlabel('Objective Values', fontsize=12)
            ax2.set_ylabel('Objective Values', fontsize=12)
            ax2.set_title(f'Pairwise Objectives (first {max_obj_show} objectives)', fontsize=12, fontweight='bold')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(results_dir, f'pareto_front_{num_objectives}d.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{num_objectives}D Pareto front saved: {plot_path}")
    
    # Training progress visualization (if iterations are available)
    create_training_progress_plot(results_dir)

def create_training_progress_plot(results_dir):
    """Create a plot showing training progress over iterations."""
    
    print("Creating training progress visualization...")
    
    # Find all iteration directories
    iteration_dirs = []
    for item in os.listdir(results_dir):
        if item.isdigit():
            iteration_dirs.append(int(item))
    
    iteration_dirs.sort()
    
    if len(iteration_dirs) == 0:
        print("No iteration data found for progress plot.")
        return
    
    # Collect data over iterations
    iterations = []
    elite_counts = []
    max_obj1 = []
    max_obj2 = []
    max_obj3 = []
    
    for iteration in iteration_dirs:
        iter_dir = os.path.join(results_dir, str(iteration))
        elites_file = os.path.join(iter_dir, 'elites', 'elites.txt')
        
        if os.path.exists(elites_file):
            with open(elites_file, 'r') as f:
                elites_lines = f.readlines()
            
            elite_objs = []
            for line in elites_lines:
                if line.strip():
                    objs = [float(x) for x in line.strip().split(',')]
                    elite_objs.append(objs)
            
            if elite_objs:
                elite_objs = np.array(elite_objs)
                iterations.append(iteration)
                elite_counts.append(len(elite_objs))
                max_obj1.append(np.max(elite_objs[:, 0]))
                max_obj2.append(np.max(elite_objs[:, 1]))
                if elite_objs.shape[1] > 2:
                    max_obj3.append(np.max(elite_objs[:, 2]))
    
    if len(iterations) > 0:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Elite count over iterations
        ax1.plot(iterations, elite_counts, 'bo-', linewidth=2, markersize=6)
        ax1.set_xlabel('Iteration', fontsize=11)
        ax1.set_ylabel('Number of Elites', fontsize=11)
        ax1.set_title('Elite Count Progress', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Objective progress
        ax2.plot(iterations, max_obj1, 'r-o', linewidth=2, markersize=4, label='Max Obj 1')
        ax2.plot(iterations, max_obj2, 'g-s', linewidth=2, markersize=4, label='Max Obj 2')
        if max_obj3:
            ax2.plot(iterations, max_obj3, 'b-^', linewidth=2, markersize=4, label='Max Obj 3')
        ax2.set_xlabel('Iteration', fontsize=11)
        ax2.set_ylabel('Maximum Objective Value', fontsize=11)
        ax2.set_title('Objective Progress', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Normalized objectives
        if len(max_obj1) > 1:
            norm_obj1 = np.array(max_obj1) / np.max(max_obj1)
            norm_obj2 = np.array(max_obj2) / np.max(max_obj2)
            
            ax3.plot(iterations, norm_obj1, 'r-o', linewidth=2, markersize=4, label='Norm Obj 1')
            ax3.plot(iterations, norm_obj2, 'g-s', linewidth=2, markersize=4, label='Norm Obj 2')
            if max_obj3:
                norm_obj3 = np.array(max_obj3) / np.max(max_obj3)
                ax3.plot(iterations, norm_obj3, 'b-^', linewidth=2, markersize=4, label='Norm Obj 3')
            
            ax3.set_xlabel('Iteration', fontsize=11)
            ax3.set_ylabel('Normalized Max Objective', fontsize=11)
            ax3.set_title('Normalized Progress', fontsize=12, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Elite count histogram
        ax4.hist(elite_counts, bins=max(1, len(set(elite_counts))), 
                alpha=0.7, color='skyblue', edgecolor='black')
        ax4.set_xlabel('Number of Elites', fontsize=11)
        ax4.set_ylabel('Frequency', fontsize=11)
        ax4.set_title('Elite Count Distribution', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        progress_path = os.path.join(results_dir, 'training_progress.png')
        plt.savefig(progress_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training progress saved: {progress_path}")
    else:
        print("No valid iteration data found for progress plot.")

def main():
    """Main analysis function."""
    
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        # Default to checking common result directories
        possible_dirs = ['./detailed_results', './example_3d_results', './example_results']
        results_dir = None
        
        for dir_path in possible_dirs:
            if os.path.exists(dir_path):
                results_dir = dir_path
                break
        
        if results_dir is None:
            print("No results directory found!")
            print("Usage: python analyze_results.py [results_directory]")
            print("Or make sure one of these directories exists:")
            for d in possible_dirs:
                print(f"  - {d}")
            return False
    
    print("MORL Results Analysis")
    print("=" * 60)
    
    return analyze_results(results_dir)

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
