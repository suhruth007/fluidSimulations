#!/bin/bash

# CUDA LBM Simulation Build Script for Linux/macOS
# Usage: ./build.sh [clean] [Release/Debug]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'  # No Color

# Get build type from arguments, default to Release
BUILD_TYPE="Release"
if [ -n "$2" ]; then
    BUILD_TYPE="$2"
fi

echo ""
echo "========================================"
echo "CUDA LBM Simulation Build Script"
echo "========================================"
echo "Build Type: $BUILD_TYPE"
echo ""

# Check if CUDA is installed
if [ -z "$CUDA_PATH" ] && [ -z "$CUDA_HOME" ]; then
    # Try common installation paths
    if [ -d "/usr/local/cuda" ]; then
        export CUDA_PATH="/usr/local/cuda"
        echo -e "${YELLOW}[INFO]${NC} Found CUDA at /usr/local/cuda"
    elif [ -d "/opt/cuda" ]; then
        export CUDA_PATH="/opt/cuda"
        echo -e "${YELLOW}[INFO]${NC} Found CUDA at /opt/cuda"
    else
        echo -e "${RED}[ERROR]${NC} CUDA not found"
        echo "Please install NVIDIA CUDA Toolkit and set CUDA_PATH environment variable"
        echo "  export CUDA_PATH=/path/to/cuda"
        exit 1
    fi
else
    CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
    echo -e "${GREEN}[SUCCESS]${NC} Using CUDA: $CUDA_PATH"
fi

# Add CUDA to PATH for CMake detection
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LD_LIBRARY_PATH"

# Check if CMake is installed
if ! command -v cmake &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} CMake not found"
    echo "Please install CMake 3.18 or later"
    exit 1
fi
CMAKE_VERSION=$(cmake --version | head -n1)
echo -e "${GREEN}[SUCCESS]${NC} $CMAKE_VERSION"

# Check if NVIDIA compiler is available
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} NVIDIA CUDA Compiler (nvcc) not found"
    echo "Ensure CUDA_PATH is set correctly and nvcc is in PATH"
    exit 1
fi
NVCC_VERSION=$(nvcc --version | tail -n1)
echo -e "${GREEN}[SUCCESS]${NC} $NVCC_VERSION"
echo ""

# Create build directory
if [ ! -d "build" ]; then
    echo "Creating build directory..."
    mkdir build
fi

# Clean if requested
if [ "$1" = "clean" ]; then
    echo "Cleaning build directory..."
    rm -rf build/*
    echo -e "${GREEN}[SUCCESS]${NC} Clean complete"
    echo ""
fi

# Run CMake configuration
cd build
echo "Configuring with CMake..."

# Detect number of CPU cores for parallel builds
if command -v nproc &> /dev/null; then
    NUM_CORES=$(nproc)
elif [ "$(uname)" = "Darwin" ]; then
    NUM_CORES=$(sysctl -n hw.ncpu)
else
    NUM_CORES=4
fi

cmake .. \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DCMAKE_CUDA_ARCHITECTURES="75;80;86" \
    -DCMAKE_CUDA_COMPILER=$CUDA_PATH/bin/nvcc

if [ $? -ne 0 ]; then
    echo -e "${RED}[ERROR]${NC} CMake configuration failed"
    exit 1
fi
echo -e "${GREEN}[SUCCESS]${NC} CMake configuration complete"
echo ""

# Build
echo "Building (using $NUM_CORES cores)..."
cmake --build . --config $BUILD_TYPE --parallel $NUM_CORES

if [ $? -ne 0 ]; then
    echo -e "${RED}[ERROR]${NC} Build failed"
    exit 1
fi
echo -e "${GREEN}[SUCCESS]${NC} Build complete"
echo ""

# Check if executable was created
if [ -f "./lbm_simulation_gpu" ]; then
    echo -e "${GREEN}[SUCCESS]${NC} Executable created: ./lbm_simulation_gpu"
    echo ""
    echo "To run the simulation:"
    echo "  cd build"
    echo "  ./lbm_simulation_gpu"
    echo ""
    echo "For GPU profiling (NVIDIA GPUs):"
    echo "  ncu ./lbm_simulation_gpu  # NVIDIA Nsight Compute"
    echo "  nsys profile ./lbm_simulation_gpu  # NVIDIA Nsight Systems"
else
    echo -e "${YELLOW}[WARNING]${NC} Executable not found at expected location"
fi

cd ..
echo ""
echo "========================================"
echo "Build Complete!"
echo "========================================"
