"""
GPU Acceleration Benchmark Suite

Compares performance between CPU (NumPy) and GPU (CuPy) implementations
of the 3D Lattice Boltzmann Method (D3Q27).

Usage:
    python benchmark_gpu.py [grid_size] [steps]
    
    Example:
    python benchmark_gpu.py 100 10  # 100³ grid, 10 steps
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import time
import numpy as np
from src.phase4.lbm_3d_gpu import LBM3DGPU, GPU_AVAILABLE
from src.phase4.mesh_loader import create_simple_cylinder
from src.phase4.voxelizer import Voxelizer


def create_benchmark_voxel_grid(size: int = 50) -> object:
    """Create a simple cylinder voxel grid for benchmarking"""
    print(f"Creating {size}³ benchmark grid with cylinder geometry...")
    
    # Create simple cylinder mesh
    mesh = create_simple_cylinder(radius=0.5, height=2.0)
    
    # Voxelize
    voxelizer = Voxelizer(resolution=1.0/size)
    voxel_grid = voxelizer.voxelize(mesh)
    
    # Adjust to target size
    print(f"Grid shape: {voxel_grid.shape}")
    
    return voxel_grid


def benchmark_single_device(
    voxel_grid,
    reynolds: float = 100.0,
    use_gpu: bool = True,
    num_steps: int = 10
) -> dict:
    """Run benchmark on single device"""
    device_name = "GPU (CuPy + CUDA)" if use_gpu else "CPU (NumPy)"
    print(f"\n{'='*70}")
    print(f"Benchmarking: {device_name}")
    print(f"{'='*70}")
    
    try:
        sim = LBM3DGPU(
            voxel_grid,
            reynolds=reynolds,
            inlet_velocity=0.1,
            use_gpu=use_gpu,
            verbose=True
        )
        
        results = sim.benchmark_speed(num_steps=num_steps)
        
        # Print detailed results
        print(f"\n📊 Detailed Results:")
        print(f"  Grid Size: {results['grid_size']}")
        print(f"  Total Cells: {results['cells']:,}")
        print(f"  Time per Step: {results['avg_time_per_step']*1000:.2f} ms")
        print(f"  Steps per Second: {results['steps_per_second']:.1f} FPS")
        print(f"  Throughput: {results['mlups']:.1f} Million LUPs/sec")
        
        return results
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None


def main():
    """Run full benchmark comparison"""
    
    # Parse arguments
    grid_size = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    num_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    print("\n" + "="*70)
    print("🚀 GPU ACCELERATION BENCHMARK - Phase 4.2")
    print("="*70)
    print(f"Date: April 27, 2026")
    print(f"Grid Size: {grid_size}³")
    print(f"Benchmark Steps: {num_steps}")
    print(f"GPU Available: {'✅ YES' if GPU_AVAILABLE else '❌ NO'}")
    print("="*70)
    
    # Create benchmark grid
    try:
        voxel_grid = create_benchmark_voxel_grid(size=grid_size)
    except Exception as e:
        print(f"❌ Error creating voxel grid: {e}")
        print("\nTrying with smaller grid...")
        voxel_grid = create_benchmark_voxel_grid(size=30)
    
    results = {}
    
    # CPU benchmark
    print("\n" + "="*70)
    print("PHASE 1: CPU (NumPy) Benchmark")
    print("="*70)
    results['cpu'] = benchmark_single_device(
        voxel_grid,
        use_gpu=False,
        num_steps=num_steps
    )
    
    # GPU benchmark (if available)
    if GPU_AVAILABLE:
        print("\n" + "="*70)
        print("PHASE 2: GPU (CuPy + CUDA) Benchmark")
        print("="*70)
        results['gpu'] = benchmark_single_device(
            voxel_grid,
            use_gpu=True,
            num_steps=num_steps
        )
    else:
        print("\n⚠️  GPU (CuPy) not available. Install with:")
        print("  pip install cupy-cuda12x  (replace 12x with your CUDA version)")
    
    # Comparison
    print("\n" + "="*70)
    print("SUMMARY & COMPARISON")
    print("="*70)
    
    if results['cpu']:
        print(f"\n✅ CPU (NumPy):")
        print(f"  Time per step: {results['cpu']['avg_time_per_step']*1000:.2f} ms")
        print(f"  Throughput: {results['cpu']['mlups']:.1f} MLUPS")
    
    if results.get('gpu'):
        print(f"\n✅ GPU (CuPy + CUDA):")
        print(f"  Time per step: {results['gpu']['avg_time_per_step']*1000:.2f} ms")
        print(f"  Throughput: {results['gpu']['mlups']:.1f} MLUPS")
        
        speedup = results['cpu']['avg_time_per_step'] / results['gpu']['avg_time_per_step']
        print(f"\n🚀 SPEEDUP: {speedup:.1f}×")
        print(f"   GPU is {speedup:.1f}x faster than CPU!")
        
        # Performance targets
        print(f"\n📈 vs Phase 4.1 (CPU only):")
        print(f"  Phase 4.1: 400 ms/step for 100³ (2.5 FPS)")
        print(f"  Phase 4.2 Goal: 5 ms/step for 100³ (200 FPS)")
        print(f"  Expected: 80× speedup")
    
    print("\n" + "="*70)
    print("✅ Benchmark Complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
