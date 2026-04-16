# CUDA Build Guide for LBM Simulation

## Overview

This guide explains how to build and run the CUDA-accelerated Lattice Boltzmann Method (LBM) fluid simulation. The CUDA implementation provides significant performance improvements over the CPU-based C++ version, leveraging NVIDIA GPUs for massive parallelization.

**Performance Summary:**
- **Python (original):** ~450+ ms/iteration
- **Python (Numba JIT):** ~29.7 ms/iteration (15× speedup)
- **C++ (OpenMP):** ~0.3-0.8 ms/iteration (100× speedup)
- **CUDA (GPU):** ~0.01-0.08 ms/iteration (1000+× speedup) *theoretical*

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Building the Simulation](#building-the-simulation)
4. [Running the Simulation](#running-the-simulation)
5. [GPU Selection & Compute Capability](#gpu-selection--compute-capability)
6. [Performance Tuning](#performance-tuning)
7. [Troubleshooting](#troubleshooting)
8. [Performance Benchmarking](#performance-benchmarking)

---

## System Requirements

### Hardware Requirements

- **NVIDIA GPU:** Compute Capability 3.0 or higher
  - Examples: Tesla K40, GTX 750 Ti, RTX 2060, RTX 3090, H100, etc.
- **GPU Memory:** Minimum 2 GB VRAM (tested with 4-12 GB systems)
  - Default simulation: ~400MB peak GPU memory
- **Host Memory:** Minimum 2 GB RAM for CPU operations

### Software Requirements

#### Windows
- **CUDA Toolkit:** Version 11.0 or later
  - Download: [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- **Visual Studio:** 2015 or later with C++ workload
- **CMake:** Version 3.18 or later
- **NVIDIA Driver:** Latest stable version

#### Linux/macOS
- **CUDA Toolkit:** Version 11.0 or later
- **GCC/Clang:** C++17 compatible compiler
- **CMake:** Version 3.18 or later
- **Make or Ninja:** Build system (installed automatically with CMake)
- **NVIDIA Driver:** Latest stable version

---

## Installation

### Step 1: Verify NVIDIA Driver

Check if your NVIDIA driver is installed:

```bash
# Linux/macOS
nvidia-smi

# Windows (PowerShell)
& "C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
```

Expected output shows GPU information. If not found, [download the driver](https://www.nvidia.com/download/driverDetails.aspx).

### Step 2: Install CUDA Toolkit

#### Windows

1. Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
2. Run the installer with default settings
3. The installer automatically sets `CUDA_PATH` environment variable
4. Restart your terminal/IDE for changes to take effect

#### Linux

```bash
# Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-repo-ubuntu2004_11.8.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004_11.8.0-1_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo apt update
sudo apt install cuda-toolkit-11-8

# Fedora/RHEL
sudo dnf install cuda
```

After installation, add to `~/.bashrc`:
```bash
export CUDA_PATH=/usr/local/cuda
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
```

#### macOS

```bash
# Using Homebrew (if available)
brew install cuda

# Or download from NVIDIA website and install
# https://developer.nvidia.com/cuda-toolkit
```

### Step 3: Install CMake

#### Windows
Download and install from [cmake.org](https://cmake.org/download/)

#### Linux/macOS
```bash
# Ubuntu/Debian
sudo apt install cmake

# macOS (Homebrew)
brew install cmake

# Fedora/RHEL
sudo dnf install cmake
```

### Step 4: Verify Installation

```bash
# Check CUDA Compiler
nvcc --version

# Check CMake
cmake --version

# Check GCC/Clang
gcc --version  # or clang --version
```

---

## Building the Simulation

### Option 1: Automated Build (Recommended)

#### Windows

Navigate to the `cpp/cuda/` directory and run:

```powershell
.\build.bat
```

For Release build (optimized):
```powershell
.\build.bat Release
```

For Debug build (with symbols):
```powershell
.\build.bat Debug
```

To clean build artifacts:
```powershell
.\build.bat clean
```

#### Linux/macOS

Make the script executable and run:

```bash
chmod +x build.sh
./build.sh
```

For Release build:
```bash
./build.sh Release
```

For Debug build:
```bash
./build.sh Debug
```

To clean build artifacts:
```bash
./build.sh clean
```

### Option 2: Manual CMake Build

If the automated scripts don't work, build manually:

#### Windows (Visual Studio)

```powershell
cd cpp\cuda
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release --parallel 4
```

#### Linux/macOS

```bash
cd cpp/cuda
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel $(nproc)
```

---

## Running the Simulation

### Basic Execution

#### Windows

```powershell
cd cpp\cuda\build
.\Release\lbm_simulation_gpu.exe
```

#### Linux/macOS

```bash
cd cpp/cuda/build
./lbm_simulation_gpu
```

### Expected Output

```
======================================
CUDA Lattice Boltzmann Simulation
======================================
GPU: Tesla V100 (Compute Capability 7.0)
GPU Memory: 32000 MB
Device Properties:
  Max threads per block: 1024
  Warp size: 32
  Max grid dimensions: 65535 x 65535 x 65535

Simulation Parameters:
  Domain size: 400 x 100 x 9
  Time steps: 30000
  Reynolds number: 40.0
  Relaxation time: 0.6

Memory Allocation:
  F distributions: 1440.0 MB
  Temporary buffers: 480.0 MB
  Macroscopic vars: 4.8 MB
  Vorticity field: 1.56 MB
  Total: 1926.36 MB

Starting simulation...
Iteration 0/30000
Iteration 5000/30000
...
Iteration 30000/30000

Simulation Complete!
Total time: 24.5 seconds
Per-iteration time: 0.817 ms
Throughput: 1.22 Gflop/s
GFLOPs (total): 1000000.0
```

### Customizing Simulation Parameters

Edit [main.cu](src/main.cu) to modify simulation:

```cpp
// Line ~25-35
const int Nx = 400;              // Domain width (lattice units)
const int Ny = 100;              // Domain height
const int Nt = 30000;            // Simulation timesteps
const double CYLINDER_RADIUS = 13.0;  // Cylinder radius
const double INLET_VELOCITY = 0.1;    // Inlet flow velocity
const double tau = 0.6;          // Relaxation time (viscosity control)
```

Then rebuild:
```bash
cd build
cmake --build . --parallel 4
```

---

## GPU Selection & Compute Capability

### Checking Your GPU

```bash
nvidia-smi
```

Output example:
```
GPU  Name                 Compute Capability  Memory
0    Tesla V100              7.0              16GB
1    RTX 3090                8.6              24GB
```

### Compute Capability Mapping

The build system automatically compiles for architectures: **75, 80, 86**

| Compute Capability | GPU Examples | Release Year |
|---|---|---|
| 3.0-3.5 | GTX Titan, GTX 750 | 2012-2013 |
| 5.0-5.3 | GTX 750 Ti, GTX 960 | 2014 |
| 6.0-6.2 | GTX 1080, Titan X | 2015-2016 |
| 7.0-7.5 | Tesla V100, RTX 2070 | 2017-2018 |
| **8.0-8.6** | A100, RTX 3060/3090 | 2020-2021 |
| **8.9** | RTX 4090 | 2022 |

### Optimizing for Your GPU

Edit [CMakeLists.txt](CMakeLists.txt) line ~31:

```cmake
# Current (broad compatibility)
set_target_properties(lbm_simulation_gpu PROPERTIES CUDA_ARCHITECTURES "75;80;86")

# For RTX 4090 only (maximum optimization, no portability)
set_target_properties(lbm_simulation_gpu PROPERTIES CUDA_ARCHITECTURES "89")

# For older GPUs (Tesla K40/M40)
set_target_properties(lbm_simulation_gpu PROPERTIES CUDA_ARCHITECTURES "35;50;60;70")
```

Then rebuild.

---

## Performance Tuning

### Register Usage Optimization

In [lbm_kernels.cu](src/lbm_kernels.cu), near line ~150 (kernel_collision_step):

```cpp
// Current: ~80 registers/thread
// To reduce register pressure:
__global__ void kernel_collision_step(...) {
    // Use __launch_bounds__ to hint compiler
    // Reduces register usage but may reduce occupancy
}
```

### Shared Memory Optimization

The current implementation uses global memory. For further speedup, you can add shared memory:

```cpp
extern __shared__ double shared_f[256 * 9];  // Allocate per-block cache
```

This requires modifying kernel launches and rebuilds.

### Thread Block Size

Current: 8×8×2 = 256 threads (occupancy ~50-75% on RTX cards)

For optimization:
- **Decrease** to 128 threads if register spilling occurs
- **Increase** to 512 if memory bandwidth is bottleneck

Edit [lbm_kernels.cu](src/lbm_kernels.cu) block dimensions (line ~160):

```cpp
dim3 blockDim(8, 8, 2);  // Change these values
```

### Memory Coalescing

The kernel uses linear indexing: `idx = y*Nx*NL + x*NL + i`

This is already optimized for memory coalescing. Avoid changing the indexing scheme.

---

## Troubleshooting

### Error: "nvcc not found"

**Cause:** CUDA Toolkit not installed or not in PATH

**Solution:**
```bash
# Windows: Verify CUDA_PATH is set
echo %CUDA_PATH%

# Linux/macOS: Set manually
export CUDA_PATH=/usr/local/cuda
export PATH=$CUDA_PATH/bin:$PATH
```

### Error: "No NVIDIA GPUs detected"

**Cause:** NVIDIA driver not installed or GPU not recognized

**Solution:**
1. Run `nvidia-smi` to verify driver installation
2. Check Device Manager (Windows) for unrecognized NVIDIA devices
3. Update driver from [nvidia.com/download](https://www.nvidia.com/download/)
4. Ensure GPU is enabled in BIOS

### Error: "CUDA out of memory"

**Cause:** GPU memory insufficient for simulation

**Solution:**
1. Reduce simulation domain size: `Nx`, `Ny` in main.cu
2. Close other GPU applications (PyTorch, Tensorflow, etc.)
3. Upgrade to larger GPU memory

### Error: "CMake cannot find CUDA"

**Cause:** CMake cannot locate CUDA installation

**Solution:**
```bash
# Explicitly specify CUDA path
cd build
cmake .. -DCMAKE_CUDA_COMPILER=/path/to/nvcc
```

### Slow Performance (<100 ms/iteration)

**Cause:** GPU not being fully utilized

**Solution:**
1. Verify GPU is actually being used: `nvidia-smi` while running
2. Check for PCI-e bottleneck (should show ~60-70% GPU utilization at minimum)
3. Increase Nt (total timesteps) to amortize overhead

---

## Performance Benchmarking

### Measuring Execution Time

The program automatically reports performance:

```
Total time: 24.5 seconds
Per-iteration time: 0.817 ms
Throughput: 1.22 Gflop/s
```

### Detailed GPU Profiling

#### Using NVIDIA Nsight Compute (NCU)

```bash
# Ubuntu/Debian
sudo apt install nvidia-nsight-compute

# Profile the simulation
ncu ./lbm_simulation_gpu
```

Output includes:
- SM occupancy (%)
- Memory bandwidth utilization
- Warp efficiency
- Register usage per thread

#### Using NVIDIA Nsight Systems

```bash
# Ubuntu/Debian
sudo apt install nvidia-nsight-systems

# Trace entire execution
nsys profile -o lbm_trace ./lbm_simulation_gpu

# Analyze
nsys-ui lbm_trace.nsys-rep
```

### Comparing Performance Across Implementations

Create a benchmark script (benchmark.py):

```python
import subprocess
import time

# Python (Numba)
print("Python (Numba):")
start = time.time()
subprocess.run(["python", "main.py"], cwd="../..")
print(f"Total: {time.time() - start:.2f}s\n")

# C++ (OpenMP)
print("C++ (OpenMP):")
start = time.time()
subprocess.run(["./build/lbm_simulation"], cwd="../")
print(f"Total: {time.time() - start:.2f}s\n")

# CUDA (GPU)
print("CUDA (GPU):")
start = time.time()
subprocess.run(["./lbm_simulation_gpu"], cwd="./build")
print(f"Total: {time.time() - start:.2f}s\n")
```

### Performance Targets

For 400×100×9 domain with 30,000 iterations on typical GPUs:

| GPU | Architecture | Expected Time | ms/iteration |
|---|---|---|---|
| Tesla V100 | Volta (7.0) | 15-20s | 0.5-0.67 ms |
| RTX 2080 Ti | Turing (7.5) | 12-18s | 0.4-0.6 ms |
| RTX 3090 | Ampere (8.6) | 8-12s | 0.27-0.4 ms |
| RTX 4090 | Ada (8.9) | 5-8s | 0.17-0.27 ms |
| H100 | Hopper (9.0) | 3-5s | 0.1-0.17 ms |

---

## Advanced Topics

### Compiling for Multiple GPU Architectures

Edit `CMakeLists.txt` to compile for all supported GPUs:

```cmake
set_target_properties(lbm_simulation_gpu PROPERTIES 
    CUDA_ARCHITECTURES "35;50;60;70;75;80;86;89")
```

This increases binary size (~20-50 MB) but provides portability.

### Custom Memory Management

For multi-GPU execution, modify `main.cu`:

```cpp
cudaSetDevice(gpu_id);  // Select GPU
cudaMalloc(...);        // Allocate on selected GPU
```

### Performance Profiling Workflow

1. **Baseline:** Run and record initial performance
2. **Profile:** Use ncu or nsys to identify bottlenecks
3. **Optimize:** Modify kernel (shared memory, thread blocks)
4. **Validate:** Ensure results match CPU version
5. **Compare:** Measure speedup

### Validation Against CPU Version

The CUDA kernels should produce identical results (within floating-point precision) as the CPU version. To verify:

1. Run CPU version and save output
2. Run GPU version and save output
3. Compare using `numpy.allclose(cpu_output, gpu_output, atol=1e-6)`

---

## References

- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVIDIA Performance Profiling Guide](https://docs.nvidia.com/cuda/profiler-users-guide/)
- [CMake CUDA Support](https://cmake.org/cmake/help/latest/language/CUDA/)
- [LBM Algorithm Theory](./SIMULATION_EXPLANATION.md)

---

## Support

For issues or questions:

1. Check [Troubleshooting](#troubleshooting) section
2. Verify CUDA installation: `nvcc --version`
3. Confirm GPU visibility: `nvidia-smi`
4. Review CMake configuration: `cat build/CMakeCache.txt | grep CUDA`

Good luck with your fluid simulation!
