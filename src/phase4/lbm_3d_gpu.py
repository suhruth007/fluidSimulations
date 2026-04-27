"""
3D Lattice Boltzmann Method with GPU Acceleration (CuPy)

GPU-accelerated D3Q27 lattice Boltzmann solver using CuPy for NVIDIA GPUs.
Falls back to CPU (NumPy) if GPU is unavailable.

Performance targets:
  - GPU (RTX 4090): 100³ grid @ 5ms/step (200 FPS)
  - CPU (i7-12700): 100³ grid @ 400ms/step (2.5 FPS)
  - Expected speedup: 10-100× depending on grid size

Usage:
    from lbm_3d_gpu import LBM3DGPU
    
    # Automatically detects GPU
    sim = LBM3DGPU(voxel_grid, reynolds=100, use_gpu=True)
    
    # Step-by-step simulation
    for step in range(100):
        sim.step()
        print(f"Drag: {sim.get_drag():.4f}")
"""

import numpy as np
from typing import Tuple, Optional
import time
import warnings

# Try to import CuPy for GPU support
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None
    warnings.warn(
        "CuPy not installed. GPU acceleration unavailable. "
        "Install with: pip install cupy-cuda12x (replace 12x with your CUDA version)"
    )

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


# ============================================================
# D3Q27 LATTICE DEFINITIONS (GPU-compatible)
# ============================================================

# 27-velocity lattice (D3Q27) - works on both CPU and GPU
D3Q27_VELOCITIES = np.array([
    [0, 0, 0],      # 0: rest
    [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],  # 1-6: axis-aligned
    [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],                       # 7-10: xy diagonals
    [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],                       # 11-14: xz diagonals
    [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],                       # 15-18: yz diagonals
    [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],                       # 19-22: body diagonals
    [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]                    # 23-26: more body diagonals
], dtype=np.float32)

D3Q27_WEIGHTS = np.array([
    8/27,                                           # 0: rest
    2/27, 2/27, 2/27, 2/27, 2/27, 2/27,           # 1-6: axis-aligned
    1/54, 1/54, 1/54, 1/54,                        # 7-10: xy diagonals
    1/54, 1/54, 1/54, 1/54,                        # 11-14: xz diagonals
    1/54, 1/54, 1/54, 1/54,                        # 15-18: yz diagonals
    1/216, 1/216, 1/216, 1/216,                    # 19-22: body diagonals
    1/216, 1/216, 1/216, 1/216                     # 23-26: body diagonals
], dtype=np.float32)


# ============================================================
# NUMBA-COMPILED UTILITY FUNCTIONS
# ============================================================

if NUMBA_AVAILABLE:
    @jit(nopython=True)
    def compute_eq_velocity_numba(f, rho, u, velocities, weights, cs_sq):
        """Compute equilibrium distribution (Numba version)"""
        f_eq = np.zeros_like(f)
        cs2 = cs_sq
        cs4 = cs_sq ** 2
        
        for i in range(f.shape[0]):  # 27 velocities
            ei_u = velocities[i, 0] * u[0] + velocities[i, 1] * u[1] + velocities[i, 2] * u[2]
            u_sq = u[0]*u[0] + u[1]*u[1] + u[2]*u[2]
            f_eq[i] = weights[i] * rho * (1.0 + ei_u / cs2 + 
                                          (ei_u*ei_u) / (2*cs4) - u_sq / (2*cs2))
        return f_eq
else:
    def compute_eq_velocity_numba(f, rho, u, velocities, weights, cs_sq):
        """Fallback without Numba"""
        f_eq = np.zeros_like(f)
        cs2 = cs_sq
        cs4 = cs_sq ** 2
        
        for i in range(len(velocities)):
            ei_u = (velocities[i] * u).sum()
            u_sq = (u ** 2).sum()
            f_eq[i] = weights[i] * rho * (1.0 + ei_u / cs2 + 
                                          (ei_u*ei_u) / (2*cs4) - u_sq / (2*cs2))
        return f_eq


# ============================================================
# GPU VERSION (CuPy)
# ============================================================

class LBM3DGPU:
    """
    GPU-accelerated 3D Lattice Boltzmann Method (D3Q27)
    
    Supports both NVIDIA GPUs (via CuPy) and CPU fallback (via NumPy).
    Automatically detects and uses available hardware.
    
    Attributes:
        shape: Tuple (nx, ny, nz) - grid dimensions
        f: Distribution functions (27 × nx × ny × nz)
        rho: Macroscopic density field (nx × ny × nz)
        u: Macroscopic velocity field (3 × nx × ny × nz)
        obstacle: Obstacle mask (nx × ny × nz)
        use_gpu: Whether GPU is being used
        device: CuPy/NumPy array type
    """
    
    def __init__(
        self,
        voxel_grid,
        reynolds: float = 100.0,
        inlet_velocity: float = 0.1,
        use_gpu: bool = True,
        verbose: bool = True
    ):
        """
        Initialize GPU-accelerated LBM simulator
        
        Args:
            voxel_grid: VoxelGrid object with voxelization data
            reynolds: Reynolds number for the flow
            inlet_velocity: Inlet boundary velocity (m/s)
            use_gpu: Try to use GPU if available (falls back to CPU)
            verbose: Print GPU/CPU status
        """
        self.shape = voxel_grid.shape
        self.inlet_velocity = inlet_velocity
        self.reynolds = reynolds
        self.time_step = 0
        self.verbose = verbose
        
        # Determine execution device
        self.use_gpu = use_gpu and GPU_AVAILABLE
        if self.use_gpu:
            self.device = cp
            self.xp = cp
            device_name = "GPU (CuPy + CUDA)"
        else:
            self.device = np
            self.xp = np
            device_name = "CPU (NumPy)"
        
        if verbose:
            print(f"🖥️  LBM3D initialized on: {device_name}")
            if use_gpu and not GPU_AVAILABLE:
                print("⚠️  GPU requested but not available - using CPU fallback")
        
        # Lattice parameters
        self.cs_sq = 1.0 / 3.0  # Speed of sound squared
        self.cs4 = (self.cs_sq ** 2)
        
        # Calculate tau from Reynolds number
        nu = inlet_velocity * 2.0 / reynolds  # kinematic viscosity
        self.tau = 0.5 + 3.0 * nu
        
        # Velocities and weights
        self.velocities = self.xp.asarray(D3Q27_VELOCITIES, dtype=np.float32)
        self.weights = self.xp.asarray(D3Q27_WEIGHTS, dtype=np.float32)
        
        # Initialize distribution function
        nx, ny, nz = self.shape
        self.f = self.xp.ones((27, nx, ny, nz), dtype=np.float32)
        self.f_new = self.xp.ones((27, nx, ny, nz), dtype=np.float32)
        
        # Macroscopic variables
        self.rho = self.xp.ones((nx, ny, nz), dtype=np.float32)
        self.u = self.xp.zeros((3, nx, ny, nz), dtype=np.float32)
        
        # Obstacle mask (1 = fluid, 0 = solid)
        obstacle_cpu = voxel_grid.obstacle
        self.obstacle = self.xp.asarray(obstacle_cpu, dtype=np.int32)
        
        # Statistics tracking
        self.drag_history = []
        self.pressure_history = []
        self.velocity_history = []
        
        if verbose:
            gpu_memory = f"{(27 * nx * ny * nz * 4 / 1e9):.2f} GB" if self.use_gpu else "N/A"
            print(f"📊 Grid: {nx} × {ny} × {nz} = {nx*ny*nz:,} cells")
            print(f"💾 Memory (GPU): {gpu_memory}")
    
    def step(self) -> None:
        """Execute one LBM timestep with GPU acceleration"""
        # Inlet boundary condition
        self._apply_inlet_bc()
        
        # Collision (BGK operator)
        self._collision()
        
        # Streaming (pull algorithm)
        self._streaming()
        
        # Bounce-back boundary condition on obstacles
        self._bounce_back()
        
        # Update macroscopic variables
        self._update_macroscopic()
        
        # Outlet boundary condition
        self._apply_outlet_bc()
        
        self.time_step += 1
    
    def _collision(self) -> None:
        """BGK collision operator (GPU-accelerated)"""
        # Update macroscopic variables for collision
        rho = self.xp.sum(self.f, axis=0)
        u = self.xp.zeros((3, *self.shape), dtype=np.float32)
        
        for i in range(27):
            u[0] += self.f[i] * self.velocities[i, 0]
            u[1] += self.f[i] * self.velocities[i, 1]
            u[2] += self.f[i] * self.velocities[i, 2]
        
        u /= rho
        
        # Compute equilibrium distribution
        f_eq = self.xp.zeros_like(self.f)
        for i in range(27):
            ei_u = (self.velocities[i, 0] * u[0] + 
                   self.velocities[i, 1] * u[1] + 
                   self.velocities[i, 2] * u[2])
            u_sq = u[0]**2 + u[1]**2 + u[2]**2
            
            f_eq[i] = (self.weights[i] * rho * 
                      (1.0 + ei_u / self.cs_sq + 
                       (ei_u*ei_u) / (2 * self.cs4) - 
                       u_sq / (2 * self.cs_sq)))
        
        # BGK collision: f_new = f - (f - f_eq) / tau
        self.f = self.f - (self.f - f_eq) / self.tau
    
    def _streaming(self) -> None:
        """Pull-based streaming (GPU-accelerated with roll operations)"""
        for i in range(27):
            vel = self.velocities[i].astype(int)
            # Roll along each axis (pull algorithm)
            self.f[i] = self.xp.roll(
                self.xp.roll(
                    self.xp.roll(self.f[i], -vel[0], axis=0),
                    -vel[1], axis=1
                ),
                -vel[2], axis=2
            )
    
    def _bounce_back(self) -> None:
        """Bounce-back boundary condition on obstacles"""
        # For each velocity direction, reverse it on obstacles
        for i in range(27):
            # Find opposite direction
            opposite_vel = -self.velocities[i]
            for j in range(27):
                if self.xp.allclose(self.velocities[j], opposite_vel):
                    # Bounce: f_i(obstacle) = f_opp
                    self.f[i] = self.xp.where(
                        self.obstacle == 0,
                        self.f[j],
                        self.f[i]
                    )
                    break
    
    def _update_macroscopic(self) -> None:
        """Update density and velocity fields from distributions"""
        self.rho = self.xp.sum(self.f, axis=0)
        
        self.u[0] = self.xp.sum(self.f * self.velocities[:, 0:1], axis=0) / self.rho
        self.u[1] = self.xp.sum(self.f * self.velocities[:, 1:2], axis=0) / self.rho
        self.u[2] = self.xp.sum(self.f * self.velocities[:, 2:3], axis=0) / self.rho
    
    def _apply_inlet_bc(self) -> None:
        """Inlet boundary condition: constant velocity (x=0 plane)"""
        u_inlet = self.xp.zeros((3, *self.shape), dtype=np.float32)
        u_inlet[0] = self.inlet_velocity
        
        # Set distributions at inlet
        rho_inlet = 1.0
        for i in range(27):
            ei_u = (self.velocities[i, 0] * u_inlet[0] +
                   self.velocities[i, 1] * u_inlet[1] +
                   self.velocities[i, 2] * u_inlet[2])
            u_sq = self.xp.sum(u_inlet ** 2, axis=0)
            
            self.f[i, 0, :, :] = (self.weights[i] * rho_inlet *
                                  (1.0 + ei_u[0, :, :] / self.cs_sq +
                                   (ei_u[0, :, :] ** 2) / (2 * self.cs4) -
                                   u_sq[0, :, :] / (2 * self.cs_sq)))
    
    def _apply_outlet_bc(self) -> None:
        """Outlet boundary condition: zero-gradient (x=nx-1 plane)"""
        # Copy from previous layer (zero-gradient)
        self.f[:, -1, :, :] = self.f[:, -2, :, :]
    
    def get_pressure(self) -> np.ndarray:
        """Get pressure field (cs_sq * rho)"""
        pressure = self.cs_sq * self.xp.asnumpy(self.rho) if self.use_gpu else self.cs_sq * self.rho
        return pressure
    
    def get_velocity_magnitude(self) -> np.ndarray:
        """Get velocity magnitude field"""
        if self.use_gpu:
            u = self.xp.asnumpy(self.u)
        else:
            u = self.u
        return np.sqrt(u[0]**2 + u[1]**2 + u[2]**2)
    
    def get_drag(self) -> float:
        """Calculate drag force on obstacles (simplified)"""
        # Simple approach: pressure difference across obstacle
        if self.use_gpu:
            rho = self.xp.asnumpy(self.rho)
        else:
            rho = self.rho
        
        # Estimate drag from momentum transfer
        pressure = self.cs_sq * rho
        drag = float(self.xp.sum(pressure[self.obstacle == 0])) / float(self.xp.sum(self.obstacle == 0))
        return drag
    
    def get_statistics(self) -> dict:
        """Get flow statistics"""
        if self.use_gpu:
            rho = self.xp.asnumpy(self.rho)
            u = self.xp.asnumpy(self.u)
        else:
            rho = self.rho
            u = self.u
        
        u_mag = np.sqrt(u[0]**2 + u[1]**2 + u[2]**2)
        
        return {
            'timestep': self.time_step,
            'density_min': float(np.min(rho)),
            'density_max': float(np.max(rho)),
            'density_mean': float(np.mean(rho)),
            'velocity_min': float(np.min(u_mag)),
            'velocity_max': float(np.max(u_mag)),
            'velocity_mean': float(np.mean(u_mag)),
            'pressure_min': float(self.cs_sq * np.min(rho)),
            'pressure_max': float(self.cs_sq * np.max(rho)),
            'drag': self.get_drag(),
        }
    
    def benchmark_speed(self, num_steps: int = 10) -> dict:
        """Benchmark simulation speed"""
        print(f"\n⏱️  Benchmarking {num_steps} timesteps...")
        
        # Warmup
        for _ in range(2):
            self.step()
        
        # Time steps
        start = time.time()
        for _ in range(num_steps):
            self.step()
        elapsed = time.time() - start
        
        # Synchronize GPU if needed
        if self.use_gpu:
            cp.cuda.Stream.null.synchronize()
        
        cells_per_step = np.prod(self.shape)
        mlups = (cells_per_step * num_steps / elapsed) / 1e6  # Million lattice updates per second
        
        results = {
            'device': 'GPU (CuPy)' if self.use_gpu else 'CPU (NumPy)',
            'total_time': elapsed,
            'avg_time_per_step': elapsed / num_steps,
            'steps_per_second': num_steps / elapsed,
            'mlups': mlups,  # Million lattice updates per second
            'grid_size': self.shape,
            'cells': cells_per_step,
        }
        
        print(f"  Device: {results['device']}")
        print(f"  Time per step: {results['avg_time_per_step']*1000:.1f} ms")
        print(f"  Throughput: {results['mlups']:.1f} MLUPS")
        
        return results


# ============================================================
# HYBRID CPU/GPU COMPARISON
# ============================================================

def compare_cpu_gpu(voxel_grid, reynolds: float = 100.0, num_steps: int = 5) -> dict:
    """
    Compare CPU vs GPU performance
    
    Args:
        voxel_grid: VoxelGrid object
        reynolds: Reynolds number
        num_steps: Number of steps to benchmark
    
    Returns:
        Dictionary with CPU and GPU benchmark results
    """
    print("\n" + "="*70)
    print("🔬 CPU vs GPU BENCHMARK")
    print("="*70)
    
    results = {}
    
    # CPU benchmark
    print("\n📊 CPU (NumPy) Simulation:")
    sim_cpu = LBM3DGPU(voxel_grid, reynolds=reynolds, use_gpu=False, verbose=True)
    results['cpu'] = sim_cpu.benchmark_speed(num_steps=num_steps)
    
    # GPU benchmark (if available)
    if GPU_AVAILABLE:
        print("\n📊 GPU (CuPy + CUDA) Simulation:")
        sim_gpu = LBM3DGPU(voxel_grid, reynolds=reynolds, use_gpu=True, verbose=True)
        results['gpu'] = sim_gpu.benchmark_speed(num_steps=num_steps)
        
        # Calculate speedup
        speedup = results['cpu']['avg_time_per_step'] / results['gpu']['avg_time_per_step']
        results['speedup'] = speedup
        
        print(f"\n⚡ SPEEDUP: {speedup:.1f}× (GPU is {speedup:.1f}x faster)")
    else:
        print("\n⚠️  GPU not available for comparison")
    
    print("="*70 + "\n")
    return results


if __name__ == "__main__":
    # Quick test
    print("LBM3D GPU Module loaded successfully")
    print(f"GPU Available: {GPU_AVAILABLE}")
