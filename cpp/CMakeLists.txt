cmake_minimum_required(VERSION 3.10)
project(LatticeBoltzmannSimulation CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenMP for parallelization
find_package(OpenMP REQUIRED)

# Add executable with source from src/main.cpp
add_executable(lbm_simulation src/main.cpp)

# Link OpenMP
target_link_libraries(lbm_simulation PRIVATE OpenMP::OpenMP_CXX)

# Optimization flags for Release build
if(MSVC)
    target_compile_options(lbm_simulation PRIVATE /O2 /W4)
else()
    target_compile_options(lbm_simulation PRIVATE -O3 -march=native -Wall)
endif()

# Print configuration
message(STATUS "Building ${PROJECT_NAME}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "OpenMP: Found")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
