#!/bin/bash
# Build script for macOS/Linux (CMake + Make/Ninja)
# Usage: ./build.sh [Release|Debug]

set -e

# Default to Release build
BUILD_TYPE="${1:-Release}"

echo "Building Lattice Boltzmann C++ Simulation..."
echo ""

# Check for CMake
if ! command -v cmake &> /dev/null; then
    echo "Error: CMake is not installed!"
    echo ""
    echo "Installation instructions:"
    echo "  macOS: brew install cmake"
    echo "  Ubuntu/Debian: sudo apt-get install cmake"
    echo "  Fedora: sudo dnf install cmake"
    exit 1
fi

# Check for C++ compiler
if ! command -v g++ &> /dev/null && ! command -v clang++ &> /dev/null; then
    echo "Error: C++ compiler not found!"
    echo ""
    echo "Installation instructions:"
    echo "  macOS: Install Xcode Command Line Tools: xcode-select --install"
    echo "  Ubuntu/Debian: sudo apt-get install build-essential"
    echo "  Fedora: sudo dnf groupinstall 'Development Tools'"
    exit 1
fi

# Check for OpenMP
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS specific OpenMP check
    if ! command -v libomp &> /dev/null; then
        echo "Warning: libomp may not be installed on macOS"
        echo "Installing libomp: brew install libomp"
        # Note: Don't exit, as system Clang may still have OpenMP
    fi
fi

# Create build directory
if [ ! -d "build" ]; then
    echo "Creating build directory..."
    mkdir build
fi

cd build

# Configure CMake
echo "Configuring CMake (BuildType: $BUILD_TYPE)..."
cmake .. -DCMAKE_BUILD_TYPE="$BUILD_TYPE"

if [ $? -ne 0 ]; then
    echo "CMake configuration failed!"
    exit 1
fi

# Build the project
echo ""
echo "Building project..."
cmake --build . --config "$BUILD_TYPE" --parallel "$(nproc 2>/dev/null || echo 4)"

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

# Print success message
echo ""
echo "==================================="
echo "Build successful!"
echo "==================================="
echo ""
echo "Executable location:"
echo "  ./lbm_simulation"
echo ""
echo "To run the simulation:"
echo "  ./lbm_simulation"
echo ""
echo "To profile with perf (Linux only):"
echo "  sudo perf record ./lbm_simulation"
echo "  sudo perf report"
echo ""

cd ..
