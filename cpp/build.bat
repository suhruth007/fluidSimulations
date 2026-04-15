@echo off
REM Build script for Windows (CMake + MSVC)
REM Usage: build.bat [Release|Debug]

setlocal enabledelayedexpansion

REM Default to Release build
set BUILD_TYPE=Release
if not "%~1"=="" set BUILD_TYPE=%~1

echo Building Lattice Boltzmann C++ Simulation...
echo.

REM Create build directory
if not exist "build" (
    echo Creating build directory...
    mkdir build
)

cd build

REM Run CMake configuration
echo Configuring CMake...
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=%BUILD_TYPE%

if !errorlevel! neq 0 (
    echo CMake configuration failed!
    echo.
    echo Troubleshooting:
    echo 1. Install CMake from https://cmake.org/download/
    echo 2. Install Visual Studio 2019 or later with C++ workload
    echo 3. Ensure OpenMP is installed (included in VS C++ workload)
    pause
    exit /b 1
)

REM Build the project
echo.
echo Building project in %BUILD_TYPE% mode...
cmake --build . --config %BUILD_TYPE%

if !errorlevel! neq 0 (
    echo Build failed!
    pause
    exit /b 1
)

REM Print success message
echo.
echo ===================================
echo Build successful!
echo ===================================
echo.
echo Executable location:
echo   .\%BUILD_TYPE%\lbm_simulation.exe
echo.
echo To run the simulation:
echo   .\%BUILD_TYPE%\lbm_simulation.exe
echo.

cd ..
pause
