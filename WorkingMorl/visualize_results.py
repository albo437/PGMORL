#!/usr/bin/env python3
"""
Convenience script to visualize PGMORL results in WorkingMorl.
Automatically detects result type and creates appropriate visualizations.
"""
import argparse
import os
import sys
import subprocess

def run_visualization(script_name, log_dir, extra_args=None, save_fig=False):
    """Run a visualization script with error handling"""
    script_path = os.path.join('plot', script_name)
    cmd = ['python', script_path, '--log-dir', log_dir]
    
    if save_fig:
        cmd.append('--save-fig')
    
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        if result.returncode == 0:
            print(f"✓ {script_name} completed successfully")
            if result.stdout:
                print(result.stdout)
        else:
            print(f"✗ {script_name} failed:")
            print(result.stderr)
    except Exception as e:
        print(f"✗ Error running {script_name}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Visualize PGMORL results')
    parser.add_argument('log_dir', type=str, help='Results directory (e.g., example_2d_results)')
    parser.add_argument('--env', type=str, default=None, help='Environment name (auto-detected if not provided)')
    parser.add_argument('--title', type=str, default=None, help='Plot title')
    parser.add_argument('--obj', type=str, nargs='+', default=None, help='Objective names')
    parser.add_argument('--save-fig', action='store_true', help='Save figures to disk')
    parser.add_argument('--all', action='store_true', help='Generate all visualizations')
    parser.add_argument('--pareto', action='store_true', help='Generate Pareto front analysis')
    parser.add_argument('--batch', action='store_true', help='Generate batch comparison')
    parser.add_argument('--training', action='store_true', help='Generate training visualization (interactive)')
    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        print(f"Error: Directory {args.log_dir} does not exist")
        sys.exit(1)

    # Auto-detect environment from directory name or results
    if args.env is None:
        if '2d' in args.log_dir.lower():
            args.env = 'MO-Dummy-v0'
        elif '3d' in args.log_dir.lower():
            args.env = 'MO-Dummy3-v0'
        else:
            # Try to detect from results
            objs_file = os.path.join(args.log_dir, 'final', 'objs.txt')
            if os.path.exists(objs_file):
                with open(objs_file, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line:
                        num_obj = len(first_line.split(','))
                        if num_obj == 2:
                            args.env = 'MO-Dummy-v0'
                        elif num_obj == 3:
                            args.env = 'MO-Dummy3-v0'
                        else:
                            args.env = f'MO-Dummy{num_obj}-v0'
            else:
                args.env = 'MO-Dummy-v0'  # Default

    print(f"Visualizing results from: {args.log_dir}")
    print(f"Environment: {args.env}")
    print("=" * 50)

    # Prepare common extra arguments
    extra_args = []
    if args.env:
        extra_args.extend(['--env', args.env])
    if args.title:
        extra_args.extend(['--title', args.title])
    if args.obj:
        extra_args.extend(['--obj'] + args.obj)

    # Run requested visualizations
    if args.all or args.pareto:
        print("\\n1. Generating Pareto front analysis...")
        run_visualization('ep_obj_visualize_2d.py', args.log_dir, extra_args, args.save_fig)

    if args.all or args.batch:
        print("\\n2. Generating batch comparison...")
        run_visualization('ep_batch_visualize_2d.py', args.log_dir, extra_args, args.save_fig)

    if args.training:
        print("\\n3. Generating interactive training visualization...")
        # For training visualization, don't use --save-fig to keep it interactive
        script_path = os.path.join('plot', 'training_visualize_2d.py')
        cmd = ['python', script_path, '--log-dir', args.log_dir] + extra_args
        if args.save_fig:
            cmd.append('--save-fig')
        print(f"Running: {' '.join(cmd)}")
        print("Note: This will open an interactive window. Use arrow keys to navigate.")
        subprocess.run(cmd, cwd=os.getcwd())

    if not (args.all or args.pareto or args.batch or args.training):
        print("No visualization type specified. Use --all, --pareto, --batch, or --training")
        print("Examples:")
        print(f"  python {sys.argv[0]} {args.log_dir} --all")
        print(f"  python {sys.argv[0]} {args.log_dir} --pareto")
        print(f"  python {sys.argv[0]} {args.log_dir} --training")
        print(f"  python {sys.argv[0]} {args.log_dir} --all --save-fig  # Save figures to disk")

    print("\\n" + "=" * 50)
    if args.save_fig:
        print("Visualization complete! Figures saved to the results directory.")
    else:
        print("Visualization complete! Interactive plots displayed (not saved).")

if __name__ == '__main__':
    main()
