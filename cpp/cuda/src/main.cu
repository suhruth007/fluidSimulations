#include <iostream>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>
#include "../include/lbm_kernels.h"

// Simulation parameters
const int Nx = 400;
const int Ny = 100;
const int NL = 9;
const int Nt = 30000;
const double tau = 0.53;
const int plotEvery = 25;

// Cylinder parameters
const int cylinderCenterX = Nx / 4;
const int cylinderCenterY = Ny / 2;
const double cylinderRadius = 13.0;

/**
 * Main LBM simulation on GPU
 */
void simulate_gpu() {
    std::cout << "=== Lattice Boltzmann Method - GPU CUDA Implementation ===" << std::endl;
    std::cout << "Domain: " << Nx << " x " << Ny << std::endl;
    std::cout << "Timesteps: " << Nt << std::endl;
    std::cout << "Relaxation time (tau): " << tau << std::endl;
    std::cout << std::endl;
    
    // Check CUDA availability
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "Error: No CUDA-capable GPU found!" << std::endl;
        exit(1);
    }
    
    // Print GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << std::endl;
    
    // Allocate GPU memory
    std::cout << "Allocating GPU memory..." << std::endl;
    double* d_F;
    double* d_rho;
    double* d_ux;
    double* d_uy;
    bool* d_cylinder;
    
    size_t F_size = (size_t)Nx * Ny * NL * sizeof(double);
    size_t grid2d_size = (size_t)Nx * Ny * sizeof(double);
    size_t bool_size = (size_t)Nx * Ny * sizeof(bool);
    
    cudaMalloc(&d_F, F_size);
    cudaMalloc(&d_rho, grid2d_size);
    cudaMalloc(&d_ux, grid2d_size);
    cudaMalloc(&d_uy, grid2d_size);
    cudaMalloc(&d_cylinder, bool_size);
    
    std::cout << "Total GPU memory allocated: " 
              << (F_size + grid2d_size + grid2d_size + grid2d_size + bool_size) / (1024.0 * 1024.0) 
              << " MB" << std::endl;
    std::cout << std::endl;
    
    // Initialize particle distributions on GPU
    std::cout << "Initializing particle distributions..." << std::endl;
    cuda_initialize_distributions(d_F, Nx, Ny, NL);
    cuda_synchronize();
    
    // Create cylinder mask on GPU
    std::cout << "Creating cylinder mask..." << std::endl;
    cuda_create_cylinder_mask(d_cylinder, Nx, Ny, cylinderCenterX, cylinderCenterY, cylinderRadius);
    cuda_synchronize();
    
    // Count cylinder nodes (optional - for information)
    std::cout << "Cylinder: Center (" << cylinderCenterX << ", " << cylinderCenterY 
              << "), Radius " << cylinderRadius << std::endl;
    std::cout << std::endl;
    
    // Start timing
    auto t_start = std::chrono::high_resolution_clock::now();
    
    std::cout << "Starting simulation..." << std::endl;
    std::cout << std::endl;
    
    // Main simulation loop
    for (int t = 0; t < Nt; ++t) {
        if (t % 5000 == 0) {
            std::cout << "Iteration " << t << "/" << Nt << std::endl;
        }
        
        // Step 1: Streaming phase with boundary conditions
        cuda_streaming_phase(d_F, d_cylinder, Nx, Ny, NL);
        
        // Step 2: Compute macroscopic quantities (density, velocity)
        cuda_compute_macroscopic(d_F, d_rho, d_ux, d_uy, Nx, Ny, NL);
        
        // Step 3: Apply velocity boundary conditions
        cuda_apply_velocity_bc(d_ux, d_uy, d_cylinder, Nx, Ny);
        
        // Step 4: Collision step
        cuda_collision_step(d_F, d_rho, d_ux, d_uy, tau, Nx, Ny, NL);
        
        // Visualization: compute vorticity (optional)
        if (t % plotEvery == 0) {
            double* d_curl;
            size_t curl_size = (size_t)(Nx - 2) * (Ny - 4) * sizeof(double);
            cudaMalloc(&d_curl, curl_size);
            
            cuda_compute_vorticity(d_ux, d_uy, d_curl, Nx, Ny);
            
            // To save to file, copy to host (optional):
            // double* h_curl = new double[(Nx - 2) * (Ny - 4)];
            // cuda_copy_to_host(h_curl, d_curl, curl_size);
            // ... save h_curl to file ...
            // delete[] h_curl;
            
            cudaFree(d_curl);
        }
    }
    
    // Synchronize to ensure all kernels complete
    cuda_synchronize();
    
    auto t_end = std::chrono::high_resolution_clock::now();
    
    // Calculate timing
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start);
    double total_time_seconds = duration.count() / 1000.0;
    double avg_time_ms = duration.count() / static_cast<double>(Nt);
    
    std::cout << std::endl;
    std::cout << "=== Simulation Complete ===" << std::endl;
    std::cout << "Total runtime: " << std::fixed << std::setprecision(2) << total_time_seconds << " seconds" << std::endl;
    std::cout << "Average per iteration: " << std::fixed << std::setprecision(3) << avg_time_ms << " ms" << std::endl;
    std::cout << std::endl;
    
    // Performance summary
    double flops = (double)Nt * Nx * Ny * NL * 10.0;  // Rough estimate
    double gflops = flops / (total_time_seconds * 1e9);
    std::cout << "Estimated performance: " << std::fixed << std::setprecision(2) << gflops << " GFLOPS" << std::endl;
    std::cout << std::endl;
    
    // Free GPU memory
    cudaFree(d_F);
    cudaFree(d_rho);
    cudaFree(d_ux);
    cudaFree(d_uy);
    cudaFree(d_cylinder);
}

int main() {
    try {
        simulate_gpu();
        std::cout << "Simulation completed successfully!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
