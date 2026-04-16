#ifndef LBM_KERNELS_H
#define LBM_KERNELS_H

#include <cuda_runtime.h>
#include <stdio.h>

// Grid and block size constants
#define BLOCK_SIZE 256

// Error checking macro
#define cudaCheckError() { \
    cudaError_t e = cudaGetLastError(); \
    if (e != cudaSuccess) { \
        printf("CUDA error: %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
}

/**
 * Initialize particle distributions on GPU
 * @param d_F Device pointer to F distributions [Ny][Nx][NL]
 * @param Nx Grid width
 * @param Ny Grid height
 * @param NL Number of lattice directions (9)
 */
void cuda_initialize_distributions(double* d_F, int Nx, int Ny, int NL);

/**
 * Create cylinder mask on GPU
 * @param d_cylinder Device pointer to cylinder mask [Ny][Nx]
 * @param Nx Grid width
 * @param Ny Grid height
 * @param centerX Cylinder center X
 * @param centerY Cylinder center Y
 * @param radius Cylinder radius
 */
void cuda_create_cylinder_mask(bool* d_cylinder, int Nx, int Ny, int centerX, int centerY, double radius);

/**
 * Streaming phase with boundary conditions (GPU)
 * @param d_F Device pointer to distributions (input/output)
 * @param d_cylinder Device pointer to cylinder mask
 * @param Nx Grid width
 * @param Ny Grid height
 * @param NL Number of lattice directions
 */
void cuda_streaming_phase(double* d_F, const bool* d_cylinder, int Nx, int Ny, int NL);

/**
 * Compute macroscopic quantities (density and velocity) on GPU
 * @param d_F Device pointer to distributions
 * @param d_rho Device pointer to density (output)
 * @param d_ux Device pointer to x-velocity (output)
 * @param d_uy Device pointer to y-velocity (output)
 * @param Nx Grid width
 * @param Ny Grid height
 * @param NL Number of lattice directions
 */
void cuda_compute_macroscopic(const double* d_F, double* d_rho, double* d_ux, double* d_uy, 
                               int Nx, int Ny, int NL);

/**
 * Apply velocity boundary conditions on GPU
 * @param d_ux Device pointer to x-velocity (input/output)
 * @param d_uy Device pointer to y-velocity (input/output)
 * @param d_cylinder Device pointer to cylinder mask
 * @param Nx Grid width
 * @param Ny Grid height
 */
void cuda_apply_velocity_bc(double* d_ux, double* d_uy, const bool* d_cylinder, int Nx, int Ny);

/**
 * Collision step with equilibrium relaxation on GPU
 * @param d_F Device pointer to distributions (input/output)
 * @param d_rho Device pointer to density
 * @param d_ux Device pointer to x-velocity
 * @param d_uy Device pointer to y-velocity
 * @param tau Relaxation time
 * @param Nx Grid width
 * @param Ny Grid height
 * @param NL Number of lattice directions
 */
void cuda_collision_step(double* d_F, const double* d_rho, const double* d_ux, const double* d_uy,
                         double tau, int Nx, int Ny, int NL);

/**
 * Compute vorticity field on GPU
 * @param d_ux Device pointer to x-velocity
 * @param d_uy Device pointer to y-velocity
 * @param d_curl Device pointer to vorticity (output)
 * @param Nx Grid width
 * @param Ny Grid height
 */
void cuda_compute_vorticity(const double* d_ux, const double* d_uy, double* d_curl, int Nx, int Ny);

/**
 * Copy data from device to host
 */
void cuda_copy_to_host(double* h_data, const double* d_data, size_t size);

/**
 * Copy data from host to device
 */
void cuda_copy_to_device(double* d_data, const double* h_data, size_t size);

/**
 * Free GPU memory
 */
void cuda_free(void* d_ptr);

/**
 * Get GPU memory info
 */
void cuda_print_memory_info();

/**
 * Synchronize GPU
 */
void cuda_synchronize();

#endif // LBM_KERNELS_H
