#!/usr/bin/env python
"""
run_adversarial_quick_demo.py

Quick demonstration of adversarial training test suite.
Runs 2 quick tests (baseline) and saves results.

Usage:
    python run_adversarial_quick_demo.py

This is a lightweight version of run_adversarial_test_suite.py
that completes faster and shows the basic workflow.
"""

import os
import sys
import subprocess
import json
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
RESULTS_DIR = PROJECT_ROOT / "exp/results/adversarial_demo"
LOG_DIR = RESULTS_DIR / "logs"

# Create directories
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Quick demo configs (2 short runs)
DEMO_CONFIGS = [
    {
        'model': 'RUNG_percentile_adv',
        'dataset': 'cora',
        'alpha': 0.7,
        'attack_freq': 5,
        'train_pgd_steps': 5,
        'max_epoch': 2,
        'description': 'Demo 1: percentile_adv',
    },
    {
        'model': 'RUNG_parametric_adv',
        'dataset': 'cora',
        'alpha': 0.7,
        'attack_freq': 5,
        'train_pgd_steps': 5,
        'max_epoch': 2,
        'description': 'Demo 2: parametric_adv',
    },
]


def run_demo():
    """Run the demo."""
    print("\n" + "="*80)
    print("ADVERSARIAL TRAINING - QUICK DEMO")
    print("="*80)
    print(f"Results directory: {RESULTS_DIR}\n")
    
    results = []
    total = len(DEMO_CONFIGS)
    
    for idx, config in enumerate(DEMO_CONFIGS, 1):
        print(f"\n[{idx}/{total}] {config['description']}")
        print(f"  Model: {config['model']}, Dataset: {config['dataset']}")
        print(f"  Alpha: {config['alpha']}, Freq: {config['attack_freq']}, "
              f"Steps: {config['train_pgd_steps']}, Epochs: {config['max_epoch']}")
        
        # Build command
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / 'clean.py'),
            f"--model={config['model']}",
            f"--data={config['dataset']}",
            f"--adv_alpha={config['alpha']}",
            f"--attack_freq={config['attack_freq']}",
            f"--train_pgd_steps={config['train_pgd_steps']}",
            f"--max_epoch={config['max_epoch']}",
        ]
        
        log_file = LOG_DIR / f"{idx}_{config['model']}_e{config['max_epoch']}.log"
        
        start = time.time()
        try:
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    timeout=3600,
                )
            elapsed = time.time() - start
            success = result.returncode == 0
            
            if success:
                print(f"  ✓ SUCCESS ({elapsed:.1f}s)")
            else:
                print(f"  ✗ FAILED ({elapsed:.1f}s, exit code {result.returncode})")
            
            results.append({
                'config': config,
                'success': success,
                'elapsed': elapsed,
                'log': str(log_file),
            })
            
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            print(f"  ✗ TIMEOUT ({elapsed:.1f}s)")
            results.append({
                'config': config,
                'success': False,
                'elapsed': elapsed,
                'log': str(log_file),
            })
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            results.append({
                'config': config,
                'success': False,
                'elapsed': 0,
                'log': str(log_file),
            })
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total:      {len(results)}")
    print(f"Successful: {sum(1 for r in results if r['success'])}")
    print(f"Failed:     {sum(1 for r in results if not r['success'])}")
    print(f"Time:       {sum(r['elapsed'] for r in results):.1f}s\n")
    
    for idx, result in enumerate(results, 1):
        status = "✓" if result['success'] else "✗"
        print(f"{status} [{idx}] {result['config']['description']:<40} "
              f"{result['elapsed']:>6.1f}s")
    
    print(f"\nLogs saved to: {LOG_DIR}/")
    
    # Save JSON
    json_file = RESULTS_DIR / "results.json"
    with open(json_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results,
        }, f, indent=2)
    print(f"Results JSON:  {json_file}\n")


if __name__ == '__main__':
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\n⚠ Demo interrupted")
        sys.exit(1)
