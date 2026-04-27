# Lattice Boltzmann Method: Simulation Explanation

## Overview

This code implements a **2D Lattice Boltzmann Method (LBM)** simulation to model incompressible fluid flow around a cylindrical obstacle. The Lattice Boltzmann Method is a computational fluid dynamics (CFD) technique that simulates fluid dynamics by solving a discretized Boltzmann equation on a regular grid.

## Physical Principles

### What is the Lattice Boltzmann Method?

The Lattice Boltzmann Method is an alternative to solving the Navier-Stokes equations directly. Instead of computing velocity and pressure fields, LBM tracks probability distributions of particles moving in discrete directions on a lattice. The advantages include:

- **Parallelizable**: Each grid point updates independently
- **Simple implementation**: No need for complex solvers or Poisson equations
- **Handles complex boundaries**: Bounce-back boundary conditions are straightforward
- **Naturally captures turbulence features**: Vorticity and wake patterns emerge naturally

### Governing Equations

The LBM solves the discrete Boltzmann equation in the relaxation time approximation (BGK collision operator):

$$f_i(\mathbf{x} + \mathbf{c}_i \Delta t, t + \Delta t) = f_i(\mathbf{x}, t) - \frac{1}{\tau}[f_i(\mathbf{x}, t) - f_i^{eq}(\mathbf{x}, t)]$$

Where:
- $f_i$ = particle distribution function at lattice node in direction $i$
- $\mathbf{c}_i$ = lattice velocity vector in direction $i$
- $\tau$ = relaxation time (parameter that controls viscosity)
- $f_i^{eq}$ = equilibrium distribution function

### D2Q9 Lattice

This code uses the **D2Q9** lattice (2D, 9 velocities):
- 2D domain
- 9 discrete velocities per node

The lattice velocities are:

```
Direction  cx   cy   Weight (w)
   0       0    0    4/9    (rest particle)
   1       0    1    1/9    (north)
   2       1    1    1/36   (northeast)
   3       1    0    1/9    (east)
   4       1   -1    1/36   (southeast)
   5       0   -1    1/9    (south)
   6      -1   -1    1/36   (southwest)
   7      -1    0    1/9    (west)
   8      -1    1    1/36   (northwest)
```

The weights ensure conserved quantities (mass and momentum) are preserved.

## Simulation Algorithm

The lattice Boltzmann simulation follows a three-step process that repeats for each timestep:

### Step 1: Streaming (Advection)

Particles stream to neighboring nodes in the direction of their lattice velocity:

```python
for i in range(NL):  # For each lattice direction
    F[:, :, i] = np.roll(F[:, :, i], cxs[i], axis=1)  # Move in x-direction
    F[:, :, i] = np.roll(F[:, :, i], cys[i], axis=0)  # Move in y-direction
```

This corresponds to the left-hand side of the Boltzmann equation:
$$f_i(\mathbf{x} + \mathbf{c}_i \Delta t, t + \Delta t)$$

**Boundary Conditions (Edges):**
- Right edge (x = Nx-1): Copy from interior (inlet boundary)
- Left edge (x = 0): Copy from interior (outlet boundary)
- These create a continuous flow through the domain

### Step 2: Macroscopic Quantities Computation

Calculate density (ρ) and velocity (ux, uy) from particle distributions:

$$\rho = \sum_{i=0}^{8} f_i$$

$$u_x = \frac{1}{\rho} \sum_{i=0}^{8} c_{ix} f_i$$

$$u_y = \frac{1}{\rho} \sum_{i=0}^{8} c_{iy} f_i$$

```python
rho = np.sum(F, 2)                    # Density
ux = np.sum(F * cxs, 2) / rho         # x-velocity
uy = np.sum(F * cys, 2) / rho         # y-velocity
```

### Step 3: Collision (Relaxation)

Particle distributions relax toward equilibrium, governed by the relaxation time τ:

The equilibrium distribution for D2Q9 is:

$$f_i^{eq} = \rho \cdot w_i \left(1 + 3(\mathbf{c}_i \cdot \mathbf{u}) + \frac{9}{2}(\mathbf{c}_i \cdot \mathbf{u})^2 - \frac{3}{2}|\mathbf{u}|^2\right)$$

Where:
- $w_i$ = weight for direction $i$
- $\mathbf{c}_i \cdot \mathbf{u} = c_{ix} u_x + c_{iy} u_y$

The collision step:
$$f_i^{new} = f_i - \frac{1}{\tau}(f_i - f_i^{eq})$$

```python
cu = cx * ux + cy * uy
Feq = rho * w * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * (ux * ux + uy * uy))
F[:, :, i] += -(1.0 / tau) * (F[:, :, i] - Feq)
```

**Viscosity Relationship:**
The relaxation time τ controls the fluid viscosity:
$$\nu = c_s^2 \left(\tau - \frac{1}{2}\right)$$

where $c_s = 1/\sqrt{3}$ is the lattice speed of sound. In this code, τ = 0.53 gives a specific viscosity value.

## Boundary Conditions

### 1. Inlet/Outlet (Periodic-Style)

Dirichlet boundary conditions at edges establish continuous flow:
```python
F[:, -1, [6, 7, 8]] = F[:, -2, [6, 7, 8]]  # Right edge
F[:, 0, [2, 3, 4]] = F[:, 1, [2, 3, 4]]    # Left edge
```

### 2. Cylinder (No-Slip Bounce-Back)

The cylinder implements a **momentum transfer** (no-slip) boundary condition via bounce-back:

At obstacle nodes:
```python
# Swap opposite direction lattice populations
F[y, x, 1], F[y, x, 5] = F[y, x, 5], F[y, x, 1]  # North ↔ South
F[y, x, 2], F[y, x, 6] = F[y, x, 6], F[y, x, 2]  # NE ↔ SW
F[y, x, 3], F[y, x, 7] = F[y, x, 7], F[y, x, 3]  # East ↔ West
F[y, x, 4], F[y, x, 8] = F[y, x, 8], F[y, x, 4]  # SE ↔ NW
```

This enforces zero velocity at the cylinder surface: $\mathbf{u} = 0$ on boundary.

The physical interpretation: particles that would move into the obstacle are reflected backwards, creating a no-slip condition without explicit force computation.

## Simulation Parameters

```python
Nx = 400               # Grid width (lattice units)
Ny = 100               # Grid height (lattice units)
Nt = 30000             # Number of timesteps
tau = 0.53             # Relaxation time (controls viscosity)

# Cylinder properties
center = (Nx//4, Ny//2) = (100, 50)
radius = 13            # lattice units
```

### Domain Dimensions

- **Physical interpretation**: The grid is dimensionless in lattice units. To map to real units, scale by lattice spacing Δx (e.g., 1 cm per lattice unit).
- Grid size: 400×100 = 40,000 nodes
- Total computations: 40,000 nodes × 9 directions × 30,000 steps = **10.8 billion operations** (without optimization)

## Output: Vorticity Visualization

The code visualizes **vorticity** (curl of velocity field):

$$\omega = \frac{\partial u_y}{\partial x} - \frac{\partial u_x}{\partial y}$$

Computed via finite differences:
```python
dfydx = ux[2:, 1:-1] - ux[0:-2, 1:-1]  # ∂ux/∂y
dfxdy = uy[1:-1, 2:] - uy[1:-1, 0:-2]  # ∂uy/∂x
curl = dfydx - dfxdy                    # vorticity
```

**Physical meaning:**
- Red regions: Clockwise vorticity (fluid rotating clockwise)
- Blue regions: Counter-clockwise vorticity (fluid rotating counter-clockwise)
- White (zero vorticity): Irrotational flow

The von Kármán vortex street (alternating vortices behind the cylinder) naturally emerges from the simulation.

## Physical Phenomenon: Cylinder Wake

The simulation captures the **cylinder wake** phenomenon:

1. **Separation point**: Boundary layer separates on the top and bottom of cylinder
2. **Recirculation zone**: Behind cylinder, counter-rotating vortex pair
3. **Vortex shedding**: Vortices alternate (top then bottom), creating vortex street
4. **Drag and lift**: Pressure asymmetry creates forces on cylinder

All these emerge naturally from the LBM equations—no explicit force computation needed!

## Stability and Convergence

**CFL Condition (Courant-Friedrichs-Lewy):**
For stability, timestep must satisfy: $\Delta t < \frac{\Delta x}{c}$ where $c$ is maximum fluid speed.

In lattice units, this is automatically satisfied since maximum lattice velocity = 1.

**Convergence:**
- Initial transient (first ~5000 steps): Flow accelerates, vortex street develops
- Steady fluctuations (~5000-30000 steps): Periodic vortex shedding
- The frequency of shedding depends on Reynolds number

## Computational Complexity

| Phase | Complexity | Time |
|-------|-----------|------|
| Streaming | O(N) | ~15% of loop |
| Macroscopic (ρ, u) | O(N) | ~25% of loop |
| Collision | O(N) | ~55% of loop |
| Visualization | O(N) | Async (often ignored in timing) |

Where N = Nx × Ny = 40,000

## Reynolds Number

The Reynolds number characterizes the flow:

$$Re = \frac{u_{max} \cdot L}{\nu}$$

Where:
- $u_{max}$ = characteristic velocity
- $L$ = characteristic length (cylinder diameter ≈ 26 lattice units)
- $\nu$ = kinematic viscosity

In lattice units with τ=0.53, this simulation typically achieves Re ≈ 50-200, putting the cylinder in the laminar-transitional regime where vortex shedding occurs.

## Extensions and Improvements

### For Phase 3 Optimization:
1. **GPU acceleration**: CuPy or CUDA kernels for 10-100x speedup
2. **Numba parallelization**: Multi-core speedup with `parallel=True`
3. **Higher-order schemes**: Better accuracy but more computation
4. **Multi-phase flow**: Simulate fluid-fluid interfaces

### For Physics Enhancement:
1. **Moving cylinder**: Set u_cylinder ≠ 0 for dynamic analysis
2. **Non-Newtonian fluids**: Modify collision operator for viscoelasticity
3. **Turbulence modeling**: LES (Large Eddy Simulation) with subgrid model
4. **Pressure measurement**: Extract pressure field for drag/lift calculation

## References

- Krüger T, Kusumaatmaja H, Kuzmin A, Shardt O, Silva G, Summers EM. *The lattice Boltzmann method: principles and practice*. Springer 2017.
- Succi S. *The Lattice Boltzmann Equation for Fluid Mechanics and Beyond*. Oxford University Press 2001.
- Benzi R, Succi S, Vergassola M. *The lattice Boltzmann equation: towards turbulence*. Physics Reports 1992.

---

**Author Notes:**
This simulation demonstrates the power of the LBM: with ~130 lines of optimized Python code, we obtain physically accurate fluid dynamics around an obstacle, capturing complex phenomena like vortex shedding without explicitly solving Navier-Stokes or computing pressure Poisson equations.
