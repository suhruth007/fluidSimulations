# CUDA Lattice Boltzmann Method Simulation

GPU-accelerated implementation of the Lattice Boltzmann Method (LBM) for 2D fluid dynamics simulation using NVIDIA CUDA.

## Quick Start

### Build (Automated)

**Windows:**
```powershell
.\build.bat
```

**Linux/macOS:**
```bash
chmod +x build.sh
./build.sh
```

### Run

```bash
cd build
./lbm_simulation_gpu         # Linux/macOS
Release\lbm_simulation_gpu.exe  # Windows
```

## Documentation

- **[CUDA_BUILD_GUIDE.md](CUDA_BUILD_GUIDE.md)** - Complete build & installation instructions
- **[../SIMULATION_EXPLANATION.md](../SIMULATION_EXPLANATION.md)** - Physics & algorithm details
- **[../README.md](../README.md)** - Project overview & performance metrics

## Requirements

- ✅ NVIDIA GPU (Compute Capability 3.0+)
- ✅ CUDA Toolkit 11.0 or later
- ✅ CMake 3.18+
- ✅ C++17 compatible compiler

## Performance

Expected execution time for 30,000 iterations on 400×100 grid:

| GPU | Time | ms/iteration |
|-----|------|-------------|
| RTX 3090 | 8-12s | 0.27-0.4 ms |
| RTX 2080 Ti | 12-18s | 0.4-0.6 ms |
| Tesla V100 | 15-20s | 0.5-0.67 ms |

**~1000× faster than original Python implementation**

## File Structure

```
cuda/
├── src/
│   ├── main.cu           # GPU program orchestration
│   └── lbm_kernels.cu    # 10 CUDA compute kernels
├── include/
│   └── lbm_kernels.h     # Kernel function declarations
├── CMakeLists.txt        # CMake build configuration
├── build.bat             # Windows build script
├── build.sh              # Linux/macOS build script
├── CUDA_BUILD_GUIDE.md   # Installation & troubleshooting
└── README.md             # This file
```

## Key Features

- ✅ **10 GPU kernels** - All LBM operations parallelized
- ✅ **Streaming & collision** - Optimized memory access patterns
- ✅ **Boundary conditions** - Cylinder obstacle + inlet/outlet BC
- ✅ **Device detection** - Automatic GPU capability reporting
- ✅ **Performance metrics** - Built-in timing & GFLOP calculations

## Implementation Highlights

**Kernel Summary:**

| Kernel | Purpose | Block Size |
|--------|---------|-----------|
| `kernel_streaming_phase()` | Particle advection | 8×8×2 |
| `kernel_collision_step()` | BGK collision operator (main compute) | 8×8×2 |
| `kernel_compute_macroscopic()` | ρ, ux, uy from distributions | 8×8×2 |
| `kernel_compute_vorticity()` | Curl of velocity field | 4×16 |
| `kernel_*_boundary_conditions()` | Obstacle & flow BC | 64 threads |

## Troubleshooting

**"nvcc not found"**
```bash
export CUDA_PATH=/usr/local/cuda    # Linux
export PATH=$CUDA_PATH/bin:$PATH
```

**GPU out of memory**
- Reduce domain size in `src/main.cu` (Nx, Ny parameters)

**Build fails**
- Verify CUDA installed: `nvcc --version`
- Verify GPU detected: `nvidia-smi`
- See [CUDA_BUILD_GUIDE.md](CUDA_BUILD_GUIDE.md) for full troubleshooting

## Performance Optimization

- Compute capability auto-detection
- Architecture targeting: 75, 80, 86 (RTX 2070 → RTX 3090)
- Release build optimization: `-O3 -use_fast_math`
- Parallel compilation

See [CUDA_BUILD_GUIDE.md](CUDA_BUILD_GUIDE.md#performance-tuning) for advanced tuning.

## Next Steps

1. **Build:** Run `./build.sh` or `build.bat`
2. **Run:** Execute the compiled binary
3. **Benchmark:** Compare with CPU version in `../`
4. **Optimize:** Adjust Nt (iterations) and grid size in `src/main.cu`

---

Written with CUDA 11.0+, NVIDIA GPU support, and modern C++17 practices.

For physics & algorithm background, see **[../SIMULATION_EXPLANATION.md](../SIMULATION_EXPLANATION.md)**
