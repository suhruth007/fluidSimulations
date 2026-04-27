# Phase 1 API Reference

Complete documentation of all new Phase 1 functions and classes in `main.py`.

---

## Core Metric Functions

### `compute_drag_coefficient(F, cylinder_indices, rho, ux, uy, cxs, cys, weights, D=26, U_ref=0.1)`

Calculate the drag coefficient using the momentum exchange method.

**Parameters**:
- `F` (np.ndarray): Distribution function [Ny, Nx, 9]
- `cylinder_indices` (tuple): (y_indices, x_indices) of cylinder cells
- `rho` (np.ndarray): Density field [Ny, Nx]
- `ux` (np.ndarray): X-velocity field [Ny, Nx]
- `uy` (np.ndarray): Y-velocity field [Ny, Nx]
- `cxs` (np.ndarray): Lattice x-velocities [9]
- `cys` (np.ndarray): Lattice y-velocities [9]
- `weights` (np.ndarray): Lattice weights [9]
- `D` (float): Cylinder diameter in lattice units [default: 26]
- `U_ref` (float): Reference velocity [default: 0.1]

**Returns**:
- `Cd` (float): Dimensionless drag coefficient
- `Fx` (float): Dimensional drag force

**Example**:
```python
cd, fx = compute_drag_coefficient(F, cylinder_indices, rho, ux, uy, cxs, cys, weights)
print(f"Drag coefficient: {cd:.4f}")
```

**Expected Value**: 1.465 at Re=40

---

### `compute_lift_coefficient(F, cylinder_indices, rho, ux, uy, cxs, cys, weights, D=26, U_ref=0.1)`

Calculate the lift coefficient from transverse force.

**Parameters**: Same as `compute_drag_coefficient()`

**Returns**:
- `Cl` (float): Dimensionless lift coefficient
- `Fy` (float): Dimensional lift force

**Example**:
```python
cl, fy = compute_lift_coefficient(F, cylinder_indices, rho, ux, uy, cxs, cys, weights)
print(f"Lift coefficient: {cl:.4f}")
```

**Expected Value**: Oscillates around 0 with RMS ≈ 0.35 at Re=40

---

### `compute_strouhal_number(vorticity_history, dt=0.1, U_ref=0.1, D=26)`

Compute the Strouhal number from vorticity time series using FFT.

**Parameters**:
- `vorticity_history` (list or np.ndarray): Time series of vorticity values
- `dt` (float): Time step in physical units [default: 0.1]
- `U_ref` (float): Reference velocity [default: 0.1]
- `D` (float): Cylinder diameter [default: 26]

**Returns**:
- `St` (float): Strouhal number (dimensionless)
- `f_dominant` (float): Dominant frequency in Hz

**Example**:
```python
# Collect vorticity signal
vorticity_history = [...]  # List of 2000+ values

# Compute Strouhal
st, f = compute_strouhal_number(vorticity_history)
print(f"Strouhal: {st:.4f}, Frequency: {f:.4f} Hz")
```

**Expected Value**: 0.16-0.20 at Re=40

**Algorithm**:
1. Takes last 2000 points from history
2. Removes mean
3. Applies FFT
4. Finds dominant frequency
5. Converts to dimensionless Strouhal

---

### `compute_equilibrium(rho, ux, uy, cxs, cys, weights)`

Compute equilibrium distribution function f_eq.

**Parameters**:
- `rho` (np.ndarray): Density field [Ny, Nx]
- `ux` (np.ndarray): X-velocity field [Ny, Nx]
- `uy` (np.ndarray): Y-velocity field [Ny, Nx]
- `cxs`, `cys`, `weights` (np.ndarray): Lattice parameters [9]

**Returns**:
- `f_eq` (np.ndarray): Equilibrium distribution [Ny, Nx, 9]

**Example**:
```python
f_eq = compute_equilibrium(rho, ux, uy, cxs, cys, weights)
f_neq = F - f_eq  # Non-equilibrium part
```

**Note**: Used internally by Cd/Cl calculations

---

### `compute_pressure_coefficient(rho, U_ref=0.1, rho_ref=1.0)`

Calculate the pressure coefficient field.

**Parameters**:
- `rho` (np.ndarray): Density field [Ny, Nx]
- `U_ref` (float): Reference velocity [default: 0.1]
- `rho_ref` (float): Reference density [default: 1.0]

**Returns**:
- `Cp` (np.ndarray): Pressure coefficient field [Ny, Nx]

**Example**:
```python
cp = compute_pressure_coefficient(rho)
# Cp is negative at velocity peaks, positive at stagnation points
```

**Physics**: 
```
Cp = (p - p_ref) / (0.5 * rho * U_ref^2)
```

---

### `export_to_vtk(filename, ux, uy, rho, vorticity, cylinder, Nx, Ny)`

Export simulation snapshot to VTK format for ParaView visualization.

**Parameters**:
- `filename` (str): Output VTK filename (e.g., "frame_001.vtk")
- `ux`, `uy` (np.ndarray): Velocity components [Ny, Nx]
- `rho` (np.ndarray): Density field [Ny, Nx]
- `vorticity` (np.ndarray): Vorticity field [Ny, Nx]
- `cylinder` (np.ndarray): Boolean cylinder mask [Ny, Nx]
- `Nx`, `Ny` (int): Grid dimensions

**Returns**:
- `True` if successful, `False` otherwise

**Example**:
```python
# Export at end of simulation
vorticity = ux[2:, 1:-1] - ux[0:-2, 1:-1] - (uy[1:-1, 2:] - uy[1:-1, 0:-2])
export_to_vtk('final_state.vtk', ux, uy, rho, vorticity, cylinder, Nx, Ny)

# Open in ParaView:
# File > Open > final_state.vtk
```

**Output Fields** (in ParaView):
- Velocity (3D vector field)
- Density (scalar)
- Vorticity (scalar)
- Cylinder (binary mask)

---

## MetricsTracker Class

Auto-tracking class that records metrics over simulation time.

### `__init__()`

Initialize empty tracker.

```python
from main import MetricsTracker
tracker = MetricsTracker()
```

---

### `record(t, cd, cl, vort_point, ke, enst)`

Record metrics at time step t.

**Parameters**:
- `t` (int): Time step number
- `cd` (float): Drag coefficient
- `cl` (float): Lift coefficient
- `vort_point` (float): Vorticity at monitor point
- `ke` (float): Kinetic energy
- `enst` (float): Enstrophy (vorticity squared)

**Example**:
```python
metrics = MetricsTracker()
for t in range(Nt):
    # ... simulation step ...
    if t % 50 == 0:
        cd, fx = compute_drag_coefficient(...)
        cl, fy = compute_lift_coefficient(...)
        metrics.record(t, cd, cl, vort_signal[t], ke, enstrophy)
```

---

### `get_statistics()`

Compute mean and std of steady-state metrics (2nd half of simulation).

**Returns**: Dictionary
```python
{
    'cd_mean': float,  # Mean Cd after settling
    'cd_std': float,   # Std deviation of Cd
    'cl_mean': float,  # Mean Cl after settling
    'cl_std': float,   # Std deviation of Cl
}
```

**Example**:
```python
stats = tracker.get_statistics()
print(f"Cd = {stats['cd_mean']:.4f} ± {stats['cd_std']:.4f}")
```

**Note**: Skips first 50% of data as settling time

---

### `compute_strouhal(dt=0.1, U_ref=0.1, D=26)`

Compute Strouhal number from recorded vorticity signal.

**Parameters**:
- `dt` (float): Time step [default: 0.1]
- `U_ref` (float): Reference velocity [default: 0.1]
- `D` (float): Cylinder diameter [default: 26]

**Returns**:
- `St` (float): Strouhal number
- `f_dominant` (float): Dominant frequency (Hz)

**Example**:
```python
st, f = tracker.compute_strouhal()
print(f"Strouhal: {st:.4f}")
```

---

### `save_metrics(filename='metrics.json')`

Save all recorded metrics to JSON file.

**Parameters**:
- `filename` (str): Output filename [default: 'metrics.json']

**Example**:
```python
tracker.save_metrics('phase1_results.json')
```

**JSON Structure**:
```json
{
    "time_steps": [0, 50, 100, ...],
    "cd": [1.0, 1.2, 1.465, ...],
    "cl": [0.0, 0.1, -0.05, ...],
    "ke": [0.001, 0.002, 0.003, ...],
    "enstrophy": [0.001, 0.002, 0.003, ...]
}
```

---

## Usage Example: Complete Phase 1 Test

```python
import numpy as np
from main import *

# Initialize
tracker = MetricsTracker()
U_ref = 0.1
D = 26
Nt = 50000

# ... simulation loop ...
for t in range(Nt):
    # ... streaming, collision, BC ...
    
    # Compute metrics every 50 iterations
    if t % 50 == 0:
        # Drag and lift
        cd, fx = compute_drag_coefficient(F, cylinder_indices, rho, ux, uy, cxs, cys, weights)
        cl, fy = compute_lift_coefficient(F, cylinder_indices, rho, ux, uy, cxs, cys, weights)
        
        # Monitor vorticity
        vort_point = ux[y_mon+1, x_mon] - ux[y_mon-1, x_mon] - \
                     (uy[y_mon, x_mon+1] - uy[y_mon, x_mon-1])
        
        # Energy metrics
        ke = 0.5 * np.sum(rho * (ux**2 + uy**2)) / np.sum(rho)
        enstrophy = 0.5 * np.sum((uy[1:, :-1] - uy[:-1, :-1])**2 + 
                                 (ux[:-1, 1:] - ux[:-1, :-1])**2) / np.sum(rho)
        
        # Record
        tracker.record(t, cd, cl, vort_point, ke, enstrophy)

# Post-simulation analysis
stats = tracker.get_statistics()
st, f = tracker.compute_strouhal()

print(f"Cd = {stats['cd_mean']:.4f} ± {stats['cd_std']:.4f}")
print(f"Cl = {stats['cl_mean']:.4f} ± {stats['cl_std']:.4f}")
print(f"St = {st:.4f}")

# Save results
tracker.save_metrics('phase1_results.json')

# Export to ParaView
export_to_vtk('final_state.vtk', ux, uy, rho, 
               ux[2:, 1:-1] - ux[0:-2, 1:-1] - (uy[1:-1, 2:] - uy[1:-1, 0:-2]),
               cylinder, Nx, Ny)
```

---

## Performance Notes

**Time Complexity**:
- `compute_drag_coefficient()`: O(N_cylinder) - typically < 1ms
- `compute_lift_coefficient()`: O(N_cylinder) - typically < 1ms
- `compute_strouhal_number()`: O(n log n) FFT - typically < 10ms
- `export_to_vtk()`: O(Nx * Ny) - typically 5-10ms

**Memory**:
- `MetricsTracker`: O(Nt) - grows with simulation length
- For 50k iterations: ~1MB JSON file

---

## Troubleshooting

**Issue**: Cd is way off (negative or > 5)
**Cause**: Probably using wrong initialization or testing phase data
**Fix**: Run for at least 10,000 iterations before collecting statistics

**Issue**: Strouhal is 0
**Cause**: Not enough vorticity data collected
**Fix**: Use tracker.compute_strouhal() after 100k+ iterations

**Issue**: VTK export fails
**Cause**: vtk package not installed or permission issue
**Fix**: `pip install vtk` or check file permissions

---

## Integration with GUI

The updated `run_simulation()` function automatically:
1. Creates `MetricsTracker` instance
2. Records metrics every 50 iterations
3. Computes Strouhal after simulation
4. Passes metrics to `completion_callback()`
5. Saves metrics to JSON

No additional code needed - just run `python main.py`!

---

**Version**: Phase 1 API v1.0
**Last Updated**: April 18, 2026
**Status**: Production Ready ✅
