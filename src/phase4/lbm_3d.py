"""
3D Lattice Boltzmann Method (D3Q27)

D3Q27 has 27 velocity vectors with weights optimized for 3D flows.
Implements BGK collision operator with standard boundary conditions.

Usage:
    sim = LBM3D(voxels, reynolds=100, inlet_velocity=0.1)
    for step in range(10000):
        sim.step()
        if step % 100 == 0:
            Cd = sim.get_drag()
"""

import numpy as np
from numba import jit, prange
from typing import Tuple
import time


# D3Q27 Lattice definitions
D3Q27_VELOCITIES = np.array([
    # Rest particle
    [0, 0, 0],
    # 6 axis-aligned (c=1)
    [1, 0, 0], [-1, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1],
    # 12 face diagonals (c=√2)
    [1, 1, 0], [-1, -1, 0],
    [1, -1, 0], [-1, 1, 0],
    [1, 0, 1], [-1, 0, -1],
    [1, 0, -1], [-1, 0, 1],
    [0, 1, 1], [0, -1, -1],
    [0, 1, -1], [0, -1, 1],
    # 8 body diagonals (c=√3)
    [1, 1, 1], [-1, -1, -1],
    [1, 1, -1], [-1, -1, 1],
    [1, -1, 1], [-1, 1, -1],
    [1, -1, -1], [-1, 1, 1],
], dtype=np.float32)

D3Q27_WEIGHTS = np.array([
    8/27,                           # c=0: 1
    2/27, 2/27,                     # c=1: 6 × (2/27)
    2/27, 2/27,
    2/27, 2/27,
    1/27, 1/27, 1/27, 1/27,        # c=√2: 12 × (1/27)
    1/27, 1/27, 1/27, 1/27,
    1/27, 1/27, 1/27, 1/27,
    1/54, 1/54, 1/54, 1/54,        # c=√3: 8 × (1/54)
    1/54, 1/54, 1/54, 1/54,
], dtype=np.float32)


@jit(nopython=True)
def compute_eq_velocity(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray, uz: np.ndarray,
                        c: np.ndarray, w: np.ndarray):
    """Compute equilibrium distribution function
    
    f_i^eq = w_i * ρ * (1 + 3(c_i·u)/cs² + 9(c_i·u)²/(2cs⁴) - 3u²/(2cs²))
    where cs² = 1/3 for D3Q27
    """
    Nx, Ny, Nz = rho.shape
    f_eq = np.zeros((27, Nx, Ny, Nz), dtype=np.float32)
    
    cs2 = 1.0 / 3.0
    cs4 = cs2 * cs2
    
    for i in range(27):
        ci_x, ci_y, ci_z = c[i, 0], c[i, 1], c[i, 2]
        wi = w[i]
        
        for x in prange(Nx):
            for y in range(Ny):
                for z in range(Nz):
                    ux_val = ux[x, y, z]
                    uy_val = uy[x, y, z]
                    uz_val = uz[x, y, z]
                    rho_val = rho[x, y, z]
                    
                    # c_i · u
                    cu = ci_x * ux_val + ci_y * uy_val + ci_z * uz_val
                    
                    # u²
                    u2 = ux_val**2 + uy_val**2 + uz_val**2
                    
                    # f_i^eq
                    term1 = 1.0 + 3.0 * cu / cs2
                    term2 = 9.0 * cu * cu / (2.0 * cs4)
                    term3 = 3.0 * u2 / (2.0 * cs2)
                    
                    f_eq[i, x, y, z] = wi * rho_val * (term1 + term2 - term3)
    
    return f_eq


class LBM3D:
    """3D Lattice Boltzmann Method solver"""
    
    def __init__(self, voxel_grid, reynolds: float = 100, inlet_velocity: float = 0.1,
                 viscosity: float = 1e-5):
        """
        Args:
            voxel_grid: VoxelGrid object (1=solid, 0=fluid)
            reynolds: Reynolds number
            inlet_velocity: Inlet flow velocity
            viscosity: Kinematic viscosity
        """
        self.voxel_grid = voxel_grid
        self.Nx, self.Ny, self.Nz = voxel_grid.shape
        
        self.reynolds = float(reynolds)
        self.inlet_velocity = float(inlet_velocity)
        self.viscosity = float(viscosity)
        
        # Lattice properties
        self.c = D3Q27_VELOCITIES.astype(np.float32)
        self.w = D3Q27_WEIGHTS.astype(np.float32)
        self.cs2 = 1.0 / 3.0
        
        # Relaxation time
        self.tau = self.viscosity / self.cs2 + 0.5
        
        print(f"LBM3D Simulator")
        print(f"  Grid: {self.Nx} × {self.Ny} × {self.Nz}")
        print(f"  Re: {self.reynolds}")
        print(f"  Inlet velocity: {self.inlet_velocity}")
        print(f"  Viscosity: {self.viscosity}")
        print(f"  Relaxation time: {self.tau:.4f}")
        
        # Initialize distributions
        self.f = np.zeros((27, self.Nx, self.Ny, self.Nz), dtype=np.float32)
        self.f_new = np.zeros_like(self.f)
        
        # Initialize macroscopic variables
        self.rho = np.ones((self.Nx, self.Ny, self.Nz), dtype=np.float32)
        self.ux = np.zeros((self.Nx, self.Ny, self.Nz), dtype=np.float32)
        self.uy = np.zeros((self.Nx, self.Ny, self.Nz), dtype=np.float32)
        self.uz = np.zeros((self.Nx, self.Ny, self.Nz), dtype=np.float32)
        
        # Initialize equilibrium distribution
        f_eq = compute_eq_velocity(self.rho, self.ux, self.uy, self.uz, self.c, self.w)
        self.f = f_eq.copy()
        
        # Force tracking
        self.force_x = np.zeros(self.Nx)
        self.step_count = 0
        self.time_per_step = 0.0
    
    def step(self):
        """Execute one LBM step (collision + streaming)"""
        t_start = time.time()
        
        # Inlet boundary condition
        self._apply_inlet_bc()
        
        # Collision
        self._collision()
        
        # Streaming
        self._streaming()
        
        # Outlet boundary condition (periodic or outflow)
        self._apply_outlet_bc()
        
        # Bounce-back on obstacles
        self._bounce_back()
        
        # Update macroscopic variables
        self._update_macroscopic()
        
        self.step_count += 1
        self.time_per_step = time.time() - t_start
    
    def _apply_inlet_bc(self):
        """Apply inlet boundary condition (constant velocity)"""
        # Set inlet face (x=0) to inlet velocity
        self.ux[0, :, :] = self.inlet_velocity
        self.uy[0, :, :] = 0.0
        self.uz[0, :, :] = 0.0
    
    def _apply_outlet_bc(self):
        """Apply outlet boundary condition (pressure outlet)"""
        # Simple zero-gradient extrapolation
        self.rho[-1, :, :] = self.rho[-2, :, :]
        self.ux[-1, :, :] = self.ux[-2, :, :]
        self.uy[-1, :, :] = self.uy[-2, :, :]
        self.uz[-1, :, :] = self.uz[-2, :, :]
    
    def _collision(self):
        """BGK collision operator"""
        # Compute equilibrium distribution
        f_eq = compute_eq_velocity(self.rho, self.ux, self.uy, self.uz, self.c, self.w)
        
        # BGK collision: f_new = f - (1/tau) * (f - f_eq)
        for i in range(27):
            self.f_new[i] = self.f[i] - (1.0 / self.tau) * (self.f[i] - f_eq[i])
    
    def _streaming(self):
        """Streaming step (propagate distributions)"""
        for i in range(27):
            cx, cy, cz = int(self.c[i, 0]), int(self.c[i, 1]), int(self.c[i, 2])
            
            if cx >= 0:
                x_range = range(self.Nx - 1, -1, -1)
            else:
                x_range = range(self.Nx)
            
            if cy >= 0:
                y_range = range(self.Ny - 1, -1, -1)
            else:
                y_range = range(self.Ny)
            
            if cz >= 0:
                z_range = range(self.Nz - 1, -1, -1)
            else:
                z_range = range(self.Nz)
            
            for x in x_range:
                x_src = x - cx
                if not (0 <= x_src < self.Nx):
                    continue
                
                for y in y_range:
                    y_src = y - cy
                    if not (0 <= y_src < self.Ny):
                        continue
                    
                    for z in z_range:
                        z_src = z - cz
                        if not (0 <= z_src < self.Nz):
                            continue
                        
                        self.f[i, x, y, z] = self.f_new[i, x_src, y_src, z_src]
    
    def _bounce_back(self):
        """Bounce-back boundary condition on obstacles"""
        solid = self.voxel_grid.grid
        
        for i in range(27):
            # Opposite direction index
            # Simple indexing: assume velocities are ordered such that
            # opposite of index i is some index j
            opposite_i = self._get_opposite(i)
            
            # Where there's solid, bounce back
            self.f[opposite_i, solid == 1] = self.f_new[i, solid == 1]
    
    def _get_opposite(self, i: int) -> int:
        """Get index of opposite velocity direction"""
        # Build opposite map
        opposites = {
            0: 0,   # Rest
            1: 2, 2: 1,    # ±x
            3: 4, 4: 3,    # ±y
            5: 6, 6: 5,    # ±z
            7: 8, 8: 7,    # Face diagonals
            9: 10, 10: 9,
            11: 12, 12: 11,
            13: 14, 14: 13,
            15: 16, 16: 15,
            17: 18, 18: 17,
            19: 20, 20: 19,   # Body diagonals
            21: 22, 22: 21,
            23: 24, 24: 23,
            25: 26, 26: 25,
        }
        return opposites.get(i, i)
    
    def _update_macroscopic(self):
        """Update density and velocity from distribution functions"""
        # Density: ρ = Σ f_i
        self.rho = np.sum(self.f, axis=0)
        
        # Velocity: ρu = Σ f_i * c_i
        self.ux = np.sum(self.f * self.c[:, 0, None, None, None], axis=0) / (self.rho + 1e-10)
        self.uy = np.sum(self.f * self.c[:, 1, None, None, None], axis=0) / (self.rho + 1e-10)
        self.uz = np.sum(self.f * self.c[:, 2, None, None, None], axis=0) / (self.rho + 1e-10)
        
        # Zero velocity in solid
        solid = self.voxel_grid.grid
        self.ux[solid == 1] = 0.0
        self.uy[solid == 1] = 0.0
        self.uz[solid == 1] = 0.0
    
    def get_velocity_magnitude(self):
        """Get velocity magnitude field"""
        return np.sqrt(self.ux**2 + self.uy**2 + self.uz**2)
    
    def get_pressure(self):
        """Get pressure field (p = ρ * cs²)"""
        return self.rho * self.cs2
    
    def get_vorticity(self):
        """Get vorticity field (∇ × u)"""
        # Simple finite differences
        Nx, Ny, Nz = self.shape
        vort_x = np.zeros((Nx, Ny, Nz))
        vort_y = np.zeros((Nx, Ny, Nz))
        vort_z = np.zeros((Nx, Ny, Nz))
        
        # ω_x = ∂u_z/∂y - ∂u_y/∂z
        # (not fully implemented in this example)
        
        return vort_x, vort_y, vort_z
    
    def get_drag(self):
        """Calculate drag force on obstacles"""
        solid = self.voxel_grid.grid
        pressure = self.get_pressure()
        
        # Integrate pressure on surfaces
        # Simplified: use pressure at surface voxels
        drag = np.sum(pressure[solid == 1])
        
        return drag
    
    def get_statistics(self):
        """Get simulation statistics"""
        vel_mag = self.get_velocity_magnitude()
        pressure = self.get_pressure()
        
        stats = {
            'step': self.step_count,
            'time_per_step': self.time_per_step,
            'max_velocity': np.max(vel_mag),
            'mean_velocity': np.mean(vel_mag),
            'max_pressure': np.max(pressure),
            'min_pressure': np.min(pressure),
            'drag': self.get_drag(),
        }
        
        return stats
    
    @property
    def shape(self):
        return (self.Nx, self.Ny, self.Nz)


# Example usage
if __name__ == '__main__':
    from mesh_loader import create_simple_cylinder
    from voxelizer import Voxelizer
    
    print("=" * 60)
    print("3D LBM Simulator Example")
    print("=" * 60)
    
    # Create test geometry
    print("\n1. Creating test cylinder...")
    mesh = create_simple_cylinder(radius=0.5, height=2.0, resolution=16)
    
    # Voxelize
    print("\n2. Voxelizing...")
    voxelizer = Voxelizer(resolution=0.1)
    voxel_grid = voxelizer.voxelize(mesh)
    
    # Create simulator
    print("\n3. Creating 3D LBM simulator...")
    sim = LBM3D(voxel_grid, reynolds=100, inlet_velocity=0.05)
    
    # Run a few steps
    print("\n4. Running simulation (10 steps)...")
    for step in range(10):
        sim.step()
        stats = sim.get_statistics()
        print(f"  Step {step:3d}: v_max={stats['max_velocity']:.4f}, "
              f"p_range=[{stats['min_pressure']:.4f}, {stats['max_pressure']:.4f}], "
              f"t={stats['time_per_step']*1000:.2f}ms")
    
    print("\n" + "=" * 60)
