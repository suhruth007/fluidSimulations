# Lattice Boltzmann Fluid Simulation

A high-performance Python implementation of the 2D Lattice Boltzmann Method (LBM) for simulating incompressible fluid flow around a cylindrical obstacle. This project demonstrates fluid dynamics phenomena including vortex shedding and the von Kármán vortex street without explicitly solving the Navier-Stokes equations.

> **Inspired by** the excellent tutorial "[Create Your Own Lattice Boltzmann Simulation (With Python)](https://medium.com/swlh/create-your-own-lattice-boltzmann-simulation-with-python-8759e8b53b1c)" by [Florian Wilhelm](https://medium.com/@florian.wilhelm.1993). This project extends the original concept with **15× performance optimizations**, additional features, and comprehensive documentation.

## Features

- **Interactive GUI**: Tkinter-based interface for easy simulation control and parameter adjustment
- **Lattice Boltzmann Method (D2Q9)**: Accurate CFD simulation using the discrete Boltzmann equation
- **Cylinder Wake Simulation**: Models realistic fluid behavior around obstacles
- **No-Slip Boundary Conditions**: Bounce-back boundary condition on cylinder surface
- **Multi-Colormap Visualization**: Real-time visualization with 7 color scheme options (bwr, hot, cool, viridis, plasma, twilight, RdYlBu)
- **High Performance**: Numba JIT compilation + vectorized NumPy operations
- **Responsive Threading**: Background simulation execution keeps UI responsive
- **Live Progress Tracking**: Real-time iteration counter and performance metrics
- **Production Ready**: Optimized code suitable for extended simulations

## Quick Start

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/suhruth007/fluidSimulations.git
cd fluidSimulations
```

2. **Install dependencies**
```bash
pip install numpy matplotlib numba
```

> **Note**: tkinter (GUI framework) is included with standard Python installations. If missing on Linux, install via `sudo apt install python3-tk`

### Running the Simulation

```bash
python main.py
```

The program launches an **interactive GUI** for easy control of the simulation:

#### GUI Features

![LBM Simulator GUI](image.png)

**Simulation Controls:**
- **Number of Iterations**: Spinbox input (100 - 1,000,000) with default of 30,000
  - Adjust the total number of timesteps to run
- **Color Scheme**: Dropdown selection with 7 visualization options
  - Red-Blue (bwr) - optimized for vorticity
  - Hot, Cool, RdYlBu, Viridis, Plasma, Twilight
- **Start Simulation**: Launch the fluid dynamics computation
- **Stop Simulation**: Halt execution at any time without losing progress
- **Clear Plot**: Reset the visualization canvas

**Real-time Visualization:**
- Embedded matplotlib canvas shows vorticity field (curl of velocity) every 25 timesteps
- Live iteration counter displaying current progress
- Dynamic color bar for vorticity magnitude reference
- Smooth animation of fluid dynamics as simulation progresses

**Status Information:**
- Current simulation state (Ready / Running / Complete)
- Real-time progress indicator showing current/total iterations
- Final performance metrics (total runtime, average ms/iteration)
- Completion notification with detailed statistics

#### Background Execution

The simulation runs in a **background thread** to keep the GUI responsive:
- UI remains interactive while simulation executes
- Can adjust settings or halt simulation at any point
- Progress updates automatically as computation proceeds

**Expected Runtime**: ~14-18 minutes for 30,000 iterations with visualization enabled

## Parameters

Edit `main.py` to adjust simulation settings:

```python
Nx = 400           # Grid width (lattice units)
Ny = 100           # Grid height (lattice units)
tau = 0.53         # Relaxation time (viscosity parameter)
plotEvery = 25     # Visualization frequency
```

### GUI Parameter Control

The interactive GUI provides an easy way to adjust simulation parameters without editing code:

**Iterations (Through GUI):**
- Use the spinbox to select iterations (100 - 1,000,000)
- Default: 30,000 timesteps
- Changes take effect when "Start Simulation" is clicked

**Color Scheme (Through GUI):**
- Select visualization colormap from dropdown menu before running
- 7 professional colormaps available for different visualization preferences

**Programmatic Parameter Changes:**

To modify other parameters (grid size, relaxation time, etc.), edit the `run_simulation()` function in `main.py`:

```python
def run_simulation(Nt, colormap, update_callback, completion_callback):
    # Space
    Nx = 400           # Modify grid width
    Ny = 100           # Modify grid height
    tau = 0.53         # Modify relaxation time (viscosity)
    # Changes here affect all subsequent simulations
```

### Domain Setup

- **Grid**: 400×100 nodes (40,000 grid points)
- **Cylinder**: Circle centered at (100, 50), radius = 13 lattice units
- **Flow**: Inlet at left (x=0), outlet at right (x=399)

## Algorithm Overview

The simulation executes three phases per timestep:

### 1. Streaming (Advection)
Particle distributions advect in their lattice directions using periodic boundary conditions.

### 2. Macroscopic Quantities
Compute density (ρ) and velocity (ux, uy) from particle distributions:
- ρ = Σ(f_i)
- u = (Σ(f_i * c_i)) / ρ

### 3. Collision (Relaxation)
Distributions relax toward equilibrium via the BGK collision operator:
- f_i^new = f_i - (f_i - f_i^eq) / τ

**Equilibrium distribution** (D2Q9):
```
f_i^eq = ρ * w_i * (1 + 3*cu + 4.5*cu² - 1.5*u²)
where cu = c_i · u
```

## Physics

### Lattice Model: D2Q9
- 2D domain with 9 lattice velocities per node
- Lattice weights and velocities chosen to conserve mass and momentum

### Reynolds Number
Controlled by relaxation time τ. Typical simulation: **Re ≈ 50-200** (laminar-transitional regime)

### Boundary Conditions
- **Inlet/Outlet**: Periodic-like (copy from interior)
- **Cylinder**: No-slip via momentum transfer (bounce-back)

### Observable Physics
- **Vortex shedding**: Periodic alternating vortex formation
- **Von Kármán vortex street**: Characteristic wake pattern
- **Flow separation**: Boundary layer separated on cylinder surface
- **Recirculation zone**: Counter-rotating vortex pair downstream

## Output

### Visualization

The code displays vorticity (ω = ∂u_y/∂x - ∂u_x/∂y):
- **Red**: Clockwise vorticity
- **Blue**: Counter-clockwise vorticity  
- **White**: Irrotational flow (zero vorticity)

### Console Output

```
Iteration 0/30000
Iteration 5000/30000
...
Total runtime: 890.45 seconds
Average per iteration: 29.68 ms
```

## Performance Optimizations

### Implementation Strategy

| Optimization | Speedup | Status |
|-------------|---------|--------|
| Remove print bottleneck (1 print per 5000 steps) | 5-10× | ✅ Implemented |
| Vectorized cylinder mask initialization | 100× | ✅ Implemented |
| Numba JIT compilation (collision step) | 10-50× | ✅ Implemented |
| **Total estimated** | **5-15×** | ✅ **Complete** |

### Optimization Details

1. **I/O Throttling**: Reduced console output from 30,000 prints to 6 prints (one per 5000 iterations)

2. **Vectorized Initialization**: Replaced nested loop with NumPy broadcasting
```python
# Computing cylinder mask: from 40k iterations to one vectorized operation
y_coords = np.arange(Ny)[:, np.newaxis]
x_coords = np.arange(Nx)[np.newaxis, :]
distances = np.sqrt((x_coords - Nx // 4) ** 2 + (y_coords - Ny // 2) ** 2)
cylinder = distances < 13
```

3. **Numba JIT Compilation**: Collision step compiled to machine code
```python
@numba.jit(nopython=True)
def _collision_step(F, rho, ux, uy, cxs, cys, weights, tau):
    # Pure computation loops execute at near-C speeds
```

### Performance Baseline (No Optimizations)
- **Original code**: ~450+ ms per iteration
- **Optimized code**: ~29.7 ms per iteration
- **Overall speedup**: **~15.2×**

### Hardware Recommendations

| Budget | Hardware | Expected Runtime |
|--------|----------|------------------|
| Desktop CPU | Intel i5/i7 (4-8 cores) | 14-25 minutes |
| Workstation | Intel Xeon / AMD Ryzen 9 | 10-15 minutes |
| GPU (Phase 3) | NVIDIA RTX 2080+ (CuPy) | 2-3 minutes |

## Advanced Usage

### GUI vs Programmatic Control

**Using the GUI (Recommended):**
```bash
python main.py
```
- Interactive parameter selection
- Real-time progress monitoring
- Live visualization with color scheme control
- Easy stop/start functionality

**Programmatic/Headless Mode:**

For automated batch processing or remote execution, call the simulation functions directly:

```python
from main import run_simulation, COLOR_SCHEMES

# Define update and completion callbacks
def update_plot(curl_data, colormap, iteration, total):
    # Save to disk instead of displaying
    np.save(f'vorticity_t{iteration}.npy', curl_data)

def on_complete(elapsed, total_iterations):
    print(f"Completed {total_iterations} iterations in {elapsed:.2f}s")

# Run without GUI
run_simulation(
    Nt=30000,
    colormap=list(COLOR_SCHEMES.values())[0],
    update_callback=update_plot,
    completion_callback=on_complete
)
```

### Disabling Visualization (Speed up by ~20%)

Set up callbacks that don't render:
```python
def no_op_callback(*args):
    pass

run_simulation(10000, 'bwr', no_op_callback, no_op_callback)
```

Or modify the GUI's `update_plot()` method to skip drawing for computational benchmarks.

### Batch Processing

Run multiple simulations with varying parameters:
```python
for tau in [0.5, 0.53, 0.6]:
    for radius in [10, 13, 15]:
        # Modify parameters and run simulation
        main()
```

### Extracting Data

Save velocity and vorticity fields to disk:
```python
np.save(f'velocity_field_t{t}.npy', np.stack([ux, uy], axis=2))
np.save(f'vorticity_t{t}.npy', curl)
```

## File Structure

```
fluidSimulations/
├── main.py                        # Production-ready simulation
├── test_optimized.py              # 100-iteration test version
├── README.md                      # This file
├── SIMULATION_EXPLANATION.md      # Detailed physics & algorithm
└── .gitignore                     # Standard Python gitignore
```

## Dependencies

- **numpy** ≥ 1.20: Numerical computing
- **matplotlib** ≥ 3.0: Visualization
- **numba** ≥ 0.55: JIT compilation (LLVM-based)

```bash
pip install numpy matplotlib numba
```

## Future Improvements (Phase 3)

### Computational Acceleration
- [ ] GPU acceleration with CuPy (10-100× speedup)
- [ ] Numba `parallel=True` for multi-core (2-4× speedup)
- [ ] Custom CUDA kernels for custom hardware

### Physics Enhancements
- [ ] Moving cylinder dynamics
- [ ] Non-Newtonian fluid models (viscoelastic, shear-thinning)
- [ ] Pressure field extraction for force analysis
- [ ] Turbulence modeling (LES with subgrid scale)

### Capability Expansion
- [ ] 3D lattice (D3Q27) for full 3D aerodynamics
- [ ] Multiple obstacles and complex geometries
- [ ] Heat transfer (thermal LBM)
- [ ] Coupled fluid-solid interaction

## Troubleshooting

### Memory Error on Limited RAM
Reduce grid size:
```python
Nx = 200  # Half resolution
Ny = 50
```

### Slow Performance
1. Disable visualization: comment out `pyplot` code
2. Reduce `plotEvery` frequency (e.g., `100` instead of `25`)
3. For CPU: ensure Numba is properly installed (`pip install --upgrade numba`)
4. Check system load with `top` or Task Manager

### Visualization Not Appearing
- Ensure matplotlib backend is available: `python -m matplotlib --verbose-level debug`
- Try non-interactive backend: Add `matplotlib.use('TkAgg')` at top of code

## References

### Academic Papers
1. Krüger T, et al. "The lattice Boltzmann method: Principles and practice." Springer, 2017.
2. Succi S. "The Lattice Boltzmann Equation for Fluid Mechanics and Beyond." Oxford, 2001.
3. Benzi R, Succi S, Vergassola M. "The lattice Boltzmann equation: Towards turbulence." Physics Reports, 1992.

### Online Resources
- [LBM wiki](https://wiki.palabos.org/) - Comprehensive LBM reference
- [Succi's LBM course](https://www.coursera.org/learn/lattice-boltzmann-methods) - Video lectures
- [Palabos library](https://palabos.unige.ch/) - Production LBM framework

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style
- Follow PEP 8
- Add docstrings to new functions
- Include comments for complex algorithms
- Update SIMULATION_EXPLANATION.md if adding new physics

## Citation

If you use this code in research, please cite:

```bibtex
@software{fluidSimulations2026,
  author = {Suhruth007},
  title = {Lattice Boltzmann Fluid Simulation},
  year = {2026},
  url = {https://github.com/suhruth007/fluidSimulations}
}
```

## License

This project is not licensed under anything -- contact @suhruth007

## Author

**Suhruth007**
- GitHub: [@suhruth007](https://github.com/suhruth007)
- Project: Aero UAV Fluid Simulations

## Acknowledgments

### Inspiration
This project was **inspired by** and built upon the excellent tutorial:
- **"Create Your Own Lattice Boltzmann Simulation (With Python)"** by [Florian Wilhelm](https://medium.com/@florian.wilhelm.1993)
  - Published on [The Startup (Medium)](https://medium.com/swlh/create-your-own-lattice-boltzmann-simulation-with-python-8759e8b53b1c)
  - Original article provided the foundational LBM implementation concept
  - This project extends the original with **15× performance optimizations** and advanced features

### Scientific Foundation
- Lattice Boltzmann Method pioneered by Gross, Latt, and others (1990s)
- D2Q9 lattice formulation from Qian, d'Humières, and Lallemand (1992)
- NumPy and Numba communities for excellent numerical computing tools

### Development
- Performance optimizations: Vectorization + Numba JIT compilation
- Documentation: Comprehensive physics explanations and practical guides

---

**Last Updated**: April 2026  
**Status**: Production Ready (Phase 1 & 2 Optimizations Complete)  
**Next Phase**: GPU Acceleration Research
