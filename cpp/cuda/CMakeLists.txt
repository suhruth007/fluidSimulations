cmake_minimum_required(VERSION 3.18)
project(LatticeBoltzmannCUDA CUDA CXX)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Add CUDA source files
set(CUDA_SOURCES
    src/main.cu
    src/lbm_kernels.cu
)

# Create executable
add_executable(lbm_simulation_gpu ${CUDA_SOURCES})

# Set CUDA compilation flags
set_target_properties(lbm_simulation_gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Optimization flags for Release build
if(MSVC)
    target_compile_options(lbm_simulation_gpu PRIVATE $<$<CONFIG:Release>:/O2>)
else()
    target_compile_options(lbm_simulation_gpu PRIVATE 
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-march=native>
    )
endif()

# CUDA-specific compilation flags
if(CMAKE_BUILD_TYPE MATCHES Release)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math")
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")  # Debug mode
endif()

# Set CUDA architecture (adjust based on your GPU)
# Common options:
# - 75 for RTX 2070/2080 (Turing)
# - 80 for A100, RTX 3090 (Ampere)
# - 86 for RTX 3060, 3070, 3080 (Ampere)
# - 89 for RTX 4090 (Ada)
set_target_properties(lbm_simulation_gpu PROPERTIES CUDA_ARCHITECTURES "75;80;86")

# Print configuration
message(STATUS "Building ${PROJECT_NAME}")
message(STATUS "CUDA Toolkit: ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "CUDA Architectures: 75;80;86")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Optional: Add profiling support
option(ENABLE_NVPROF "Enable NVIDIA profiling" OFF)
if(ENABLE_NVPROF)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    message(STATUS "NVIDIA Profiling: ENABLED")
endif()
