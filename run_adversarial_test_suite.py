#!/usr/bin/env python
"""
run_adversarial_test_suite.py

Comprehensive test suite for adversarial training models.
Runs multiple configurations, saves results, and generates a report.

Usage:
    python run_adversarial_test_suite.py

This will:
    1. Test both RUNG_percentile_adv and RUNG_parametric_adv
    2. Sweep over key hyperparameters (alpha, attack_freq)
    3. Save logs to exp/results/adversarial_training/
    4. Generate summary report
"""

import os
import sys
import subprocess
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Configuration
PROJECT_ROOT = Path(__file__).parent.resolve()
RESULTS_DIR = PROJECT_ROOT / "exp/results/adversarial_training"
LOG_DIR = RESULTS_DIR / "logs"

# Create directories
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Test configurations
TEST_CONFIGS = [
    # Format: {model, dataset, alpha, attack_freq, train_pgd_steps, max_epoch, description}
    
    # ====== BASELINE: Quick validation (very short, just to verify it works) ======
    {
        'model': 'RUNG_percentile_adv',
        'dataset': 'cora',
        'alpha': 0.7,
        'attack_freq': 5,
        'train_pgd_steps': 5,
        'max_epoch': 2,
        'description': 'BASELINE: percentile_adv, quick (2 epochs)',
    },
    {
        'model': 'RUNG_parametric_adv',
        'dataset': 'cora',
        'alpha': 0.7,
        'attack_freq': 5,
        'train_pgd_steps': 5,
        'max_epoch': 2,
        'description': 'BASELINE: parametric_adv, quick (2 epochs)',
    },
    
    # ====== ALPHA SENSITIVITY: Find optimal clean/robust tradeoff ======
    {
        'model': 'RUNG_percentile_adv',
        'dataset': 'cora',
        'alpha': 0.5,
        'attack_freq': 5,
        'train_pgd_steps': 10,
        'max_epoch': 10,
        'description': 'ALPHA: percentile_adv, alpha=0.5 (more adversarial)',
    },
    {
        'model': 'RUNG_percentile_adv',
        'dataset': 'cora',
        'alpha': 0.9,
        'attack_freq': 5,
        'train_pgd_steps': 10,
        'max_epoch': 10,
        'description': 'ALPHA: percentile_adv, alpha=0.9 (more clean)',
    },
    
    # ====== ATTACK FREQUENCY: Runtime vs strength tradeoff ======
    {
        'model': 'RUNG_percentile_adv',
        'dataset': 'cora',
        'alpha': 0.7,
        'attack_freq': 1,
        'train_pgd_steps': 5,
        'max_epoch': 5,
        'description': 'FREQ: attack_freq=1 (every epoch, slowest)',
    },
    {
        'model': 'RUNG_percentile_adv',
        'dataset': 'cora',
        'alpha': 0.7,
        'attack_freq': 10,
        'train_pgd_steps': 5,
        'max_epoch': 5,
        'description': 'FREQ: attack_freq=10 (every 10 epochs, fastest)',
    },
    
    # ====== PARAMETRIC GAMMA: Compare with percentile ======
    {
        'model': 'RUNG_parametric_adv',
        'dataset': 'cora',
        'alpha': 0.7,
        'attack_freq': 5,
        'train_pgd_steps': 10,
        'max_epoch': 10,
        'description': 'PARAM: parametric_adv full config',
    },
]


class TestRunner:
    """Manages execution of adversarial training test suite."""
    
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        self.start_time = datetime.now()
        
    def run_config(self, config: Dict[str, Any], run_idx: int, total: int) -> Dict[str, Any]:
        """
        Run a single configuration.
        
        Args:
            config: Test configuration dict
            run_idx: Current run index (for progress display)
            total: Total number of runs
            
        Returns:
            Dict with run results
        """
        print(f"\n{'='*80}")
        print(f"RUN {run_idx}/{total}: {config['description']}")
        print(f"{'='*80}")
        print(f"  Model:           {config['model']}")
        print(f"  Dataset:         {config['dataset']}")
        print(f"  Alpha:           {config['alpha']}")
        print(f"  Attack freq:     {config['attack_freq']}")
        print(f"  PGD steps:       {config['train_pgd_steps']}")
        print(f"  Max epochs:      {config['max_epoch']}")
        
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
            f"--lr=5e-2",
            f"--weight_decay=5e-4",
        ]
        
        # Log file
        log_name = (
            f"{run_idx:02d}_{config['model']}_"
            f"alpha{config['alpha']:.1f}_freq{config['attack_freq']}_"
            f"steps{config['train_pgd_steps']}_ep{config['max_epoch']}.log"
        )
        log_file = LOG_DIR / log_name
        
        # Run
        start = time.time()
        try:
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    timeout=1800,  # 30 minute timeout per run
                )
            elapsed = time.time() - start
            success = result.returncode == 0
            
            # Extract results from log (simplified)
            output = ""
            try:
                with open(log_file, 'r') as f:
                    output = f.read()
            except:
                output = ""
            
            print(f"  ✓ Completed in {elapsed:.1f}s")
            if success:
                print(f"  ✓ SUCCESS (exit code 0)")
            else:
                print(f"  ✗ FAILED (exit code {result.returncode})")
            
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            print(f"  ✗ TIMEOUT after {elapsed:.1f}s")
            success = False
        except Exception as e:
            elapsed = time.time() - start
            print(f"  ✗ ERROR: {e}")
            success = False
        
        result_dict = {
            'run': run_idx,
            'config': config,
            'log_file': str(log_file),
            'success': success,
            'elapsed_seconds': elapsed,
            'timestamp': datetime.now().isoformat(),
        }
        
        self.results.append(result_dict)
        return result_dict
    
    def run_all(self):
        """Run all test configurations."""
        total = len(TEST_CONFIGS)
        
        print("\n" + "="*80)
        print("ADVERSARIAL TRAINING TEST SUITE")
        print("="*80)
        print(f"Total configurations to test: {total}")
        print(f"Results directory: {RESULTS_DIR}")
        print(f"Logs directory: {LOG_DIR}")
        print()
        
        for idx, config in enumerate(TEST_CONFIGS, 1):
            self.run_config(config, idx, total)
        
        self.generate_report()
    
    def generate_report(self):
        """Generate summary report."""
        report_file = RESULTS_DIR / "report.json"
        report_txt = RESULTS_DIR / "report.txt"
        
        # JSON report
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'start_time': self.start_time.isoformat(),
            'total_runs': len(self.results),
            'successful_runs': sum(1 for r in self.results if r['success']),
            'failed_runs': sum(1 for r in self.results if not r['success']),
            'total_time_seconds': sum(r['elapsed_seconds'] for r in self.results),
            'results': self.results,
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # Text report
        with open(report_txt, 'w') as f:
            f.write("="*80 + "\n")
            f.write("ADVERSARIAL TRAINING TEST SUITE - SUMMARY REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Timestamp:      {datetime.now().isoformat()}\n")
            f.write(f"Total runs:     {len(self.results)}\n")
            f.write(f"Successful:     {sum(1 for r in self.results if r['success'])}\n")
            f.write(f"Failed:         {sum(1 for r in self.results if not r['success'])}\n")
            f.write(f"Total time:     {sum(r['elapsed_seconds'] for r in self.results):.1f}s\n\n")
            
            f.write("-"*80 + "\n")
            f.write("DETAILED RESULTS\n")
            f.write("-"*80 + "\n\n")
            
            for result in self.results:
                config = result['config']
                status = "✓ PASS" if result['success'] else "✗ FAIL"
                f.write(f"{status} | Run {result['run']:02d}: {config['description']}\n")
                f.write(f"       Model: {config['model']}, Dataset: {config['dataset']}\n")
                f.write(f"       Alpha: {config['alpha']}, Freq: {config['attack_freq']}, "
                       f"Steps: {config['train_pgd_steps']}, Epochs: {config['max_epoch']}\n")
                f.write(f"       Time: {result['elapsed_seconds']:.1f}s\n")
                f.write(f"       Log: {result['log_file']}\n\n")
        
        # Print summary
        print("\n" + "="*80)
        print("TEST SUITE SUMMARY")
        print("="*80)
        print(f"Total runs:     {len(self.results)}")
        print(f"Successful:     {sum(1 for r in self.results if r['success'])}")
        print(f"Failed:         {sum(1 for r in self.results if not r['success'])}")
        print(f"Total time:     {sum(r['elapsed_seconds'] for r in self.results):.1f}s")
        print()
        print(f"Report saved to:   {report_txt}")
        print(f"JSON report:       {report_file}")
        print(f"Logs directory:    {LOG_DIR}")
        print("="*80 + "\n")
        
        # Print results table
        print("\nDETAILED RESULTS:\n")
        for result in self.results:
            config = result['config']
            status = "✓" if result['success'] else "✗"
            print(f"{status} [{result['run']:02d}] {config['description']:<50} "
                  f"{result['elapsed_seconds']:>6.1f}s")
        
        print(f"\nLog files location: {LOG_DIR}/")
        print("\nTo view individual logs:")
        for result in self.results:
            if result['success']:
                print(f"  tail -50 {result['log_file']}")


def main():
    """Main entry point."""
    runner = TestRunner()
    try:
        runner.run_all()
    except KeyboardInterrupt:
        print("\n\n⚠ Test suite interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
