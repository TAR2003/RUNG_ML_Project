#!/usr/bin/env python
"""
test_rung_learnable_distance_combined.py

Run clean training and attack evaluation in sequence.
Single command replaces needing to run clean.py and attack.py separately.

Usage:
    # Test cosine mode on cora with budgets 0.05, 0.10, 0.20
    python test_rung_learnable_distance_combined.py

    # Test projection mode on citeseer
    python test_rung_learnable_distance_combined.py projection citeseer

    # Full help
    python test_rung_learnable_distance_combined.py --help
"""

import subprocess
import sys
import argparse

def run_command(cmd, description):
    """Run a command and report results."""
    print("\n" + "=" * 70)
    print(f"  {description}")
    print("=" * 70)
    print(f"Command: {' '.join(cmd)}")
    print("")
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n✗ {description} FAILED")
        sys.exit(1)
    print(f"\n✓ {description} COMPLETE")
    return result

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end test: train RUNG_learnable_distance + attack evaluation"
    )
    parser.add_argument(
        'distance_mode', nargs='?', default='cosine',
        choices=['cosine', 'projection', 'bilinear'],
        help='Distance metric to test (default: cosine)'
    )
    parser.add_argument(
        'dataset', nargs='?', default='cora',
        choices=['cora', 'citeseer', 'chameleon', 'squirrel'],
        help='Dataset to use (default: cora)'
    )
    parser.add_argument(
        '--budgets', type=float, nargs='+', default=[0.05, 0.10, 0.20, 0.30,0.40,0.60],
        help='Attack budgets (default: 0.05 0.10 0.20)'
    )
    parser.add_argument(
        '--max_epoch', type=int, default=300,
        help='Max training epochs (default: 300)'
    )
    parser.add_argument(
        '--skip_clean', action='store_true',
        help='Skip clean training (use pre-trained model)'
    )
    parser.add_argument(
        '--skip_attack', action='store_true',
        help='Skip attack evaluation'
    )
    parser.add_argument(
        '--skip_baseline', action='store_true',
        help='Skip baseline (Euclidean) comparison'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("RUNG_learnable_distance — Combined Test Suite")
    print("=" * 70)
    print(f"Distance Mode: {args.distance_mode}")
    print(f"Dataset:      {args.dataset}")
    print(f"Max Epochs:   {args.max_epoch}")
    print(f"Budgets:      {args.budgets}")
    print("")
    
    # ====================================================================
    # PHASE 1: Clean Training
    # ====================================================================
    if not args.skip_clean:
        cmd = [
            'python', 'clean.py',
            '--model', 'RUNG_learnable_distance',
            '--data', args.dataset,
            '--distance_mode', args.distance_mode,
            '--percentile_q', '0.75',
            '--max_epoch', str(args.max_epoch),
            '--lr', '0.05',
            '--weight_decay', '5e-4',
        ]
        run_command(cmd, "PHASE 1: Clean Training (RUNG_learnable_distance)")
    else:
        print("\n" + "=" * 70)
        print("  PHASE 1: Clean Training (SKIPPED)")
        print("=" * 70)
    
    # ====================================================================
    # PHASE 2: Attack Evaluation
    # ====================================================================
    if not args.skip_attack:
        budget_str = ' '.join(str(b) for b in args.budgets)
        cmd = [
            'python', 'attack.py',
            '--model', 'RUNG_learnable_distance',
            '--data', args.dataset,
            '--distance_mode', args.distance_mode,
            '--percentile_q', '0.75',
            '--budgets', *[str(b) for b in args.budgets],
        ]
        run_command(cmd, f"PHASE 2: Attack Evaluation (budgets: {budget_str})")
    else:
        print("\n" + "=" * 70)
        print("  PHASE 2: Attack Evaluation (SKIPPED)")
        print("=" * 70)
    
    # ====================================================================
    # PHASE 3: Baseline Comparison (Euclidean)
    # ====================================================================
    if not args.skip_baseline:
        print("\n" + "=" * 70)
        print("  PHASE 3: Baseline Comparison (Euclidean / RUNG_percentile_gamma)")
        print("=" * 70)
        
        cmd = [
            'python', 'clean.py',
            '--model', 'RUNG_percentile_gamma',
            '--data', args.dataset,
            '--percentile_q', '0.75',
            '--max_epoch', str(args.max_epoch),
            '--lr', '0.05',
            '--weight_decay', '5e-4',
        ]
        run_command(cmd, "  Training baseline (Euclidean)")
        
        cmd = [
            'python', 'attack.py',
            '--model', 'RUNG_percentile_gamma',
            '--data', args.dataset,
            '--percentile_q', '0.75',
            '--budgets', *[str(b) for b in args.budgets],
        ]
        run_command(cmd, "  Attacking baseline (Euclidean)")
    else:
        print("\n" + "=" * 70)
        print("  PHASE 3: Baseline Comparison (SKIPPED)")
        print("=" * 70)
    
    # ====================================================================
    # Summary
    # ====================================================================
    print("\n" + "=" * 70)
    print("✓ ALL PHASES COMPLETE")
    print("=" * 70)
    print("")
    print("Results saved to:")
    print(f"  - Logs: log/{args.dataset}/clean/")
    print(f"  - Logs: log/{args.dataset}/attack/")
    print("")
    print("To visualize results, run:")
    print("  python plot_logs.py")
    print("")

if __name__ == '__main__':
    main()
