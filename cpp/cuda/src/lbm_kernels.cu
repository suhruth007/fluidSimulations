#include "lbm_kernels.h"
#include <curand_kernel.h>
#include <math.h>

// D2Q9 Lattice constants (device memory)
__constant__ int d_cxs[9] = {0, 0, 1, 1, 1, 0, -1, -1, -1};
__constant__ int d_cys[9] = {0, 1, 1, 0, -1, -1, -1, 0, 1};
__constant__ double d_weights[9] = {4.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/9.0, 1.0/36.0, 1.0/9.0, 1.0/36.0, 1.0/9.0, 1.0/36.0};

/**
 * Initialize particle distributions kernel
 */
__global__ void kernel_initialize_distributions(double* F, int Nx, int Ny, int NL, unsigned long seed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < Nx && y < Ny && i < NL) {
        // Linear index: F[y][x][i]
        int idx = y * Nx * NL + x * NL + i;
        
        // Initialize with 1.0 + small random perturbation
        curandState state;
        curand_init(seed + idx, 0, 0, &state);
        double random_val = 0.01 * (curand_normal_double(&state) - 0.5);
        
        F[idx] = 1.0 + random_val;
        
        // Higher density in direction 3
        if (i == 3) {
            F[idx] = 2.3;
        }
    }
}

/**
 * Create cylinder mask kernel
 */
__global__ void kernel_create_cylinder_mask(bool* cylinder, int Nx, int Ny, int centerX, int centerY, double radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < Nx && y < Ny) {
        int idx = y * Nx + x;
        double dx = (double)(x - centerX);
        double dy = (double)(y - centerY);
        double dist = sqrt(dx * dx + dy * dy);
        cylinder[idx] = (dist < radius);
    }
}

/**
 * Streaming phase kernel - move distributions to neighboring lattice nodes
 */
__global__ void kernel_streaming_phase(double* F, double* F_temp, const bool* cylinder, int Nx, int Ny, int NL) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < Nx && y < Ny && i < NL) {
        // Source position (with periodic boundary conditions)
        int x_src = (x - d_cxs[i] + Nx) % Nx;
        int y_src = (y - d_cys[i] + Ny) % Ny;
        
        int src_idx = y_src * Nx * NL + x_src * NL + i;
        int dst_idx = y * Nx * NL + x * NL + i;
        
        F[dst_idx] = F_temp[src_idx];
    }
}

/**
 * Edge boundary condition kernel - inlet/outlet
 */
__global__ void kernel_edge_boundary_conditions(double* F, int Nx, int Ny, int NL) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (y < Ny) {
        // Right edge (x = Nx-1)
        F[y * Nx * NL + (Nx-1) * NL + 6] = F[y * Nx * NL + (Nx-2) * NL + 6];
        F[y * Nx * NL + (Nx-1) * NL + 7] = F[y * Nx * NL + (Nx-2) * NL + 7];
        F[y * Nx * NL + (Nx-1) * NL + 8] = F[y * Nx * NL + (Nx-2) * NL + 8];
        
        // Left edge (x = 0)
        F[y * Nx * NL + 0 * NL + 2] = F[y * Nx * NL + 1 * NL + 2];
        F[y * Nx * NL + 0 * NL + 3] = F[y * Nx * NL + 1 * NL + 3];
        F[y * Nx * NL + 0 * NL + 4] = F[y * Nx * NL + 1 * NL + 4];
    }
}

/**
 * Cylinder bounce-back boundary condition kernel
 */
__global__ void kernel_cylinder_boundary_conditions(double* F, const bool* cylinder, int Nx, int Ny, int NL) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < Nx && y < Ny) {
        int idx = y * Nx + x;
        
        if (cylinder[idx]) {
            int base_idx = y * Nx * NL + x * NL;
            
            // Swap opposite directions
            // 1 <-> 5 (North <-> South)
            double temp = F[base_idx + 1];
            F[base_idx + 1] = F[base_idx + 5];
            F[base_idx + 5] = temp;
            
            // 2 <-> 6 (NE <-> SW)
            temp = F[base_idx + 2];
            F[base_idx + 2] = F[base_idx + 6];
            F[base_idx + 6] = temp;
            
            // 3 <-> 7 (East <-> West)
            temp = F[base_idx + 3];
            F[base_idx + 3] = F[base_idx + 7];
            F[base_idx + 7] = temp;
            
            // 4 <-> 8 (SE <-> NW)
            temp = F[base_idx + 4];
            F[base_idx + 4] = F[base_idx + 8];
            F[base_idx + 8] = temp;
        }
    }
}

/**
 * Compute macroscopic quantities kernel (density and velocity)
 */
__global__ void kernel_compute_macroscopic(const double* F, double* rho, double* ux, double* uy,
                                           int Nx, int Ny, int NL) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < Nx && y < Ny) {
        int base_idx = y * Nx * NL + x * NL;
        int grid_idx = y * Nx + x;
        
        double rho_local = 0.0;
        double ux_local = 0.0;
        double uy_local = 0.0;
        
        // Sum over all lattice directions
        for (int i = 0; i < NL; ++i) {
            double f = F[base_idx + i];
            rho_local += f;
            ux_local += f * d_cxs[i];
            uy_local += f * d_cys[i];
        }
        
        rho[grid_idx] = rho_local;
        ux[grid_idx] = ux_local / rho_local;
        uy[grid_idx] = uy_local / rho_local;
    }
}

/**
 * Apply velocity boundary conditions kernel (zero velocity in cylinder)
 */
__global__ void kernel_apply_velocity_bc(double* ux, double* uy, const bool* cylinder, int Nx, int Ny) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < Nx && y < Ny) {
        int idx = y * Nx + x;
        
        if (cylinder[idx]) {
            ux[idx] = 0.0;
            uy[idx] = 0.0;
        }
    }
}

/**
 * Collision step kernel (equilibrium relaxation)
 * This is the main computational kernel - heavily optimized for GPU
 */
__global__ void kernel_collision_step(double* F, const double* rho, const double* ux, const double* uy,
                                      double tau, int Nx, int Ny, int NL) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < Nx && y < Ny && i < NL) {
        int grid_idx = y * Nx + x;
        int f_idx = y * Nx * NL + x * NL + i;
        
        double cx = (double)d_cxs[i];
        double cy = (double)d_cys[i];
        double w = d_weights[i];
        
        double rho_val = rho[grid_idx];
        double ux_val = ux[grid_idx];
        double uy_val = uy[grid_idx];
        
        double cu = cx * ux_val + cy * uy_val;
        double u_sq = ux_val * ux_val + uy_val * uy_val;
        
        // Equilibrium distribution
        double Feq = rho_val * w * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq);
        
        // Collision: relax toward equilibrium
        F[f_idx] += -(1.0 / tau) * (F[f_idx] - Feq);
    }
}

/**
 * Compute vorticity kernel
 */
__global__ void kernel_compute_vorticity(const double* ux, const double* uy, double* curl, int Nx, int Ny) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Valid range: x in [1, Nx-2], y in [2, Ny-3]
    if (x >= 1 && x < Nx - 1 && y >= 2 && y < Ny - 2) {
        double dfydx = ux[(y + 2) * Nx + x] - ux[(y - 2) * Nx + x];
        double dfxdy = uy[y * Nx + (x + 1)] - uy[y * Nx + (x - 1)];
        
        int curl_idx = (y - 2) * (Nx - 2) + (x - 1);
        curl[curl_idx] = dfydx - dfxdy;
    }
}

// ==================== Wrapper Functions ====================

/**
 * Initialize distributions on GPU
 */
void cuda_initialize_distributions(double* d_F, int Nx, int Ny, int NL) {
    dim3 blockSize(8, 8, 2);
    dim3 gridSize((Nx + 7) / 8, (Ny + 7) / 8, (NL + 1) / 2);
    
    kernel_initialize_distributions<<<gridSize, blockSize>>>(d_F, Nx, Ny, NL, 12345);
    cudaCheckError();
}

/**
 * Create cylinder mask on GPU
 */
void cuda_create_cylinder_mask(bool* d_cylinder, int Nx, int Ny, int centerX, int centerY, double radius) {
    dim3 blockSize(16, 16);
    dim3 gridSize((Nx + 15) / 16, (Ny + 15) / 16);
    
    kernel_create_cylinder_mask<<<gridSize, blockSize>>>(d_cylinder, Nx, Ny, centerX, centerY, radius);
    cudaCheckError();
}

/**
 * Streaming phase with boundary conditions
 */
void cuda_streaming_phase(double* d_F, const bool* d_cylinder, int Nx, int Ny, int NL) {
    // Copy F to temporary buffer
    size_t F_size = (size_t)Nx * Ny * NL * sizeof(double);
    double* d_F_temp;
    cudaMalloc(&d_F_temp, F_size);
    cudaCheckError();
    
    cudaMemcpy(d_F_temp, d_F, F_size, cudaMemcpyDeviceToDevice);
    cudaCheckError();
    
    // Streaming kernel
    dim3 blockSize(8, 8, 2);
    dim3 gridSize((Nx + 7) / 8, (Ny + 7) / 8, (NL + 1) / 2);
    
    kernel_streaming_phase<<<gridSize, blockSize>>>(d_F, d_F_temp, d_cylinder, Nx, Ny, NL);
    cudaCheckError();
    
    // Edge boundary conditions
    dim3 edge_blockSize(256);
    dim3 edge_gridSize((Ny + 255) / 256);
    kernel_edge_boundary_conditions<<<edge_gridSize, edge_blockSize>>>(d_F, Nx, Ny, NL);
    cudaCheckError();
    
    // Cylinder boundary conditions
    dim3 cyl_blockSize(16, 16);
    dim3 cyl_gridSize((Nx + 15) / 16, (Ny + 15) / 16);
    kernel_cylinder_boundary_conditions<<<cyl_gridSize, cyl_blockSize>>>(d_F, d_cylinder, Nx, Ny, NL);
    cudaCheckError();
    
    cudaFree(d_F_temp);
    cudaCheckError();
}

/**
 * Compute macroscopic quantities
 */
void cuda_compute_macroscopic(const double* d_F, double* d_rho, double* d_ux, double* d_uy,
                               int Nx, int Ny, int NL) {
    dim3 blockSize(16, 16);
    dim3 gridSize((Nx + 15) / 16, (Ny + 15) / 16);
    
    kernel_compute_macroscopic<<<gridSize, blockSize>>>(d_F, d_rho, d_ux, d_uy, Nx, Ny, NL);
    cudaCheckError();
}

/**
 * Apply velocity boundary conditions
 */
void cuda_apply_velocity_bc(double* d_ux, double* d_uy, const bool* d_cylinder, int Nx, int Ny) {
    dim3 blockSize(16, 16);
    dim3 gridSize((Nx + 15) / 16, (Ny + 15) / 16);
    
    kernel_apply_velocity_bc<<<gridSize, blockSize>>>(d_ux, d_uy, d_cylinder, Nx, Ny);
    cudaCheckError();
}

/**
 * Collision step with equilibrium relaxation
 */
void cuda_collision_step(double* d_F, const double* d_rho, const double* d_ux, const double* d_uy,
                         double tau, int Nx, int Ny, int NL) {
    dim3 blockSize(8, 8, 2);
    dim3 gridSize((Nx + 7) / 8, (Ny + 7) / 8, (NL + 1) / 2);
    
    kernel_collision_step<<<gridSize, blockSize>>>(d_F, d_rho, d_ux, d_uy, tau, Nx, Ny, NL);
    cudaCheckError();
}

/**
 * Compute vorticity field
 */
void cuda_compute_vorticity(const double* d_ux, const double* d_uy, double* d_curl, int Nx, int Ny) {
    dim3 blockSize(16, 16);
    dim3 gridSize((Nx + 15) / 16, (Ny + 15) / 16);
    
    kernel_compute_vorticity<<<gridSize, blockSize>>>(d_ux, d_uy, d_curl, Nx, Ny);
    cudaCheckError();
}

/**
 * Copy from device to host
 */
void cuda_copy_to_host(double* h_data, const double* d_data, size_t size) {
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    cudaCheckError();
}

/**
 * Copy from host to device
 */
void cuda_copy_to_device(double* d_data, const double* h_data, size_t size) {
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    cudaCheckError();
}

/**
 * Free GPU memory
 */
void cuda_free(void* d_ptr) {
    cudaFree(d_ptr);
    cudaCheckError();
}

/**
 * Print GPU memory info
 */
void cuda_print_memory_info() {
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    
    printf("GPU Memory: %.2f MB / %.2f MB\n", 
           free_mem / (1024.0 * 1024.0), 
           total_mem / (1024.0 * 1024.0));
}

/**
 * Synchronize GPU
 */
void cuda_synchronize() {
    cudaDeviceSynchronize();
    cudaCheckError();
}
