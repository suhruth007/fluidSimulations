@echo off
REM CUDA LBM Simulation Build Script for Windows
REM Usage: build.bat [clean] [Release/Debug]

setlocal enabledelayedexpansion

REM Get build type from arguments, default to Release
set BUILD_TYPE=Release
if not "%~2"=="" set BUILD_TYPE=%2

REM Color codes for output
set SUCCESS=[SUCCESS]
set ERROR=[ERROR]
set WARNING=[WARNING]

echo.
echo ========================================
echo CUDA LBM Simulation Build Script
echo ========================================
echo Build Type: %BUILD_TYPE%
echo.

REM Check if CUDA is installed
if "%CUDA_PATH%"=="" (
    echo %ERROR% CUDA_PATH environment variable not set
    echo Please install NVIDIA CUDA Toolkit and ensure CUDA_PATH is set
    exit /b 1
)

echo Using CUDA: %CUDA_PATH%

REM Create build directory
if not exist "build" (
    echo Creating build directory...
    mkdir build
)

REM Clean if requested
if "%1"=="clean" (
    echo Cleaning build directory...
    if exist "build" rmdir /s /q build
    mkdir build
    echo %SUCCESS% Clean complete
    echo.
)

REM Run CMake configuration
cd build
echo Configuring with CMake...
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_CUDA_ARCHITECTURES="75;80;86" -DCMAKE_BUILD_TYPE=%BUILD_TYPE%

if errorlevel 1 (
    echo %ERROR% CMake configuration failed
    exit /b 1
)
echo %SUCCESS% CMake configuration complete
echo.

REM Build
echo Building...
cmake --build . --config %BUILD_TYPE% --parallel 4
if errorlevel 1 (
    echo %ERROR% Build failed
    exit /b 1
)
echo %SUCCESS% Build complete
echo.

REM Check if executable was created
if exist "%BUILD_TYPE%\lbm_simulation_gpu.exe" (
    echo %SUCCESS% Executable created: %BUILD_TYPE%\lbm_simulation_gpu.exe
    echo.
    echo To run the simulation:
    echo   .\%BUILD_TYPE%\lbm_simulation_gpu.exe
) else (
    echo %WARNING% Executable not found at expected location
)

cd ..
echo.
echo ========================================
echo Build Complete!
echo ========================================

endlocal
