#!/usr/bin/env python
"""
PHASE 1 Analysis Script - Analyze and visualize Phase 1 results
Generates plots, statistics, and validation reports
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_metrics(filename='phase1_metrics.json'):
    """Load metrics from JSON file."""
    try:
        with open(filename) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {filename} not found. Run simulation first!")
        return None

def analyze_convergence(data):
    """Analyze convergence of drag coefficient."""
    cd_values = np.array(data['cd'])
    time_steps = np.array(data['time_steps'])
    
    # Split into transient and steady state (50-50)
    split_idx = len(cd_values) // 2
    
    transient_cd = cd_values[:split_idx]
    steady_cd = cd_values[split_idx:]
    
    transient_mean = np.mean(transient_cd)
    steady_mean = np.mean(steady_cd)
    steady_std = np.std(steady_cd)
    
    # Convergence rate
    conv_rate = abs(steady_mean - transient_mean) / max(abs(transient_mean), 1e-10)
    
    return {
        'transient_mean': transient_mean,
        'steady_mean': steady_mean,
        'steady_std': steady_std,
        'convergence_rate': conv_rate,
        'converged': steady_std < 0.02,
    }

def analyze_oscillations(data):
    """Analyze Cl oscillations."""
    cl_values = np.array(data['cl'])
    time_steps = np.array(data['time_steps'])
    
    # Use steady state part
    split_idx = len(cl_values) // 2
    cl_steady = cl_values[split_idx:]
    
    cl_mean = np.mean(cl_steady)
    cl_rms = np.sqrt(np.mean(cl_steady**2))
    cl_amp = (np.max(cl_steady) - np.min(cl_steady)) / 2
    
    return {
        'mean': cl_mean,
        'rms': cl_rms,
        'amplitude': cl_amp,
        'oscillating': cl_rms > 0.1,
    }

def analyze_energy(data):
    """Analyze kinetic energy and enstrophy."""
    ke = np.array(data['ke'])
    time_steps = np.array(data['time_steps'])
    
    split_idx = len(ke) // 2
    ke_steady = ke[split_idx:]
    
    return {
        'mean': np.mean(ke_steady),
        'std': np.std(ke_steady),
        'settled': np.std(ke_steady) < 0.001,
    }

def generate_plots(data, output_prefix='phase1'):
    """Generate comprehensive analysis plots."""
    
    time_steps = np.array(data['time_steps'])
    cd_values = np.array(data['cd'])
    cl_values = np.array(data['cl'])
    ke_values = np.array(data['ke'])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Phase 1: Core Metrics Analysis', fontsize=16, fontweight='bold')
    
    # 1. Drag Coefficient Convergence
    ax = axes[0, 0]
    ax.plot(time_steps, cd_values, 'b-', linewidth=1.5, label='Cd(t)')
    ax.axhline(1.465, color='r', linestyle='--', linewidth=2, label='Benchmark (1.465)')
    split_idx = len(time_steps) // 2
    ax.axvline(time_steps[split_idx], color='gray', linestyle=':', alpha=0.5, label='Steady-state start')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Drag Coefficient (Cd)')
    ax.set_title('Drag Coefficient Convergence')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 2. Lift Coefficient Oscillations
    ax = axes[0, 1]
    ax.plot(time_steps, cl_values, 'g-', linewidth=1, label='Cl(t)')
    ax.axhline(0, color='k', linestyle='-', linewidth=0.5)
    ax.axvline(time_steps[split_idx], color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Lift Coefficient (Cl)')
    ax.set_title('Lift Coefficient Oscillations')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 3. Kinetic Energy
    ax = axes[1, 0]
    ax.plot(time_steps, ke_values, 'purple', linewidth=1.5, label='KE(t)')
    ax.axvline(time_steps[split_idx], color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Kinetic Energy')
    ax.set_title('Mean Kinetic Energy Evolution')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 4. Steady-state Cd Distribution
    ax = axes[1, 1]
    cd_steady = cd_values[split_idx:]
    ax.hist(cd_steady, bins=30, color='cyan', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(cd_steady), color='r', linestyle='-', linewidth=2, label=f'Mean: {np.mean(cd_steady):.4f}')
    ax.axvline(1.465, color='orange', linestyle='--', linewidth=2, label='Benchmark')
    ax.set_xlabel('Drag Coefficient')
    ax.set_ylabel('Frequency')
    ax.set_title('Steady-State Cd Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_analysis.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_prefix}_analysis.png")
    
    return fig

def print_report(data):
    """Print comprehensive validation report."""
    
    conv = analyze_convergence(data)
    cl_osc = analyze_oscillations(data)
    energy = analyze_energy(data)
    
    print("\n" + "="*60)
    print("PHASE 1 VALIDATION REPORT")
    print("="*60)
    
    print("\n📊 CONVERGENCE ANALYSIS:")
    print(f"  Transient Cd mean:     {conv['transient_mean']:.4f}")
    print(f"  Steady-state Cd mean:  {conv['steady_mean']:.4f} ± {conv['steady_std']:.4f}")
    print(f"  Convergence rate:      {conv['convergence_rate']:.2%}")
    print(f"  Converged:             {'✓ YES' if conv['converged'] else '✗ NO'}")
    
    print("\n🌊 OSCILLATION ANALYSIS:")
    print(f"  Cl mean:               {cl_osc['mean']:.4f}")
    print(f"  Cl RMS:                {cl_osc['rms']:.4f}")
    print(f"  Cl amplitude:          {cl_osc['amplitude']:.4f}")
    print(f"  Oscillating:           {'✓ YES' if cl_osc['oscillating'] else '✗ NO (flow may be too slow)'}")
    
    print("\n⚡ ENERGY ANALYSIS:")
    print(f"  Mean KE:               {energy['mean']:.6f}")
    print(f"  KE std:                {energy['std']:.6f}")
    print(f"  Energy settled:        {'✓ YES' if energy['settled'] else '✗ NO'}")
    
    print("\n🎯 BENCHMARK COMPARISON:")
    cd_error = abs(conv['steady_mean'] - 1.465) / 1.465 * 100
    print(f"  Expected Cd:           1.465")
    print(f"  Your Cd:               {conv['steady_mean']:.4f}")
    print(f"  Error:                 {cd_error:.2f}%")
    print(f"  Acceptable:            {'✓ YES (<2%)' if cd_error < 2 else '✗ NO (>2%)'}")
    
    print("\n" + "="*60)
    
    if cd_error < 2 and conv['converged'] and cl_osc['oscillating']:
        print("✅ PHASE 1 VALIDATION: PASSED")
    else:
        print("⚠️  PHASE 1 VALIDATION: NEEDS REVIEW")
        if cd_error >= 2:
            print("   - Cd error too large (>2%). May need finer grid or longer simulation.")
        if not conv['converged']:
            print("   - Cd not converged. Run longer simulation (100k+ iterations).")
        if not cl_osc['oscillating']:
            print("   - Cl not oscillating. Check if vortex shedding is occurring.")
    
    print("="*60 + "\n")

def main():
    """Main analysis routine."""
    
    # Load metrics
    data = load_metrics()
    if data is None:
        return
    
    # Generate plots
    print("\n🔍 Generating analysis plots...")
    generate_plots(data)
    
    # Print report
    print("\n📋 Analysis report:")
    print_report(data)
    
    print("✓ Analysis complete!")

if __name__ == '__main__':
    main()
