#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <omp.h>

// Constants
const int Nx = 400;
const int Ny = 100;
const int NL = 9;
const int Nt = 30000;
const double tau = 0.53;
const int plotEvery = 25;

// D2Q9 Lattice velocities and weights
const int cxs[NL] = {0, 0, 1, 1, 1, 0, -1, -1, -1};
const int cys[NL] = {0, 1, 1, 0, -1, -1, -1, 0, 1};
const double weights[NL] = {4.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/9.0, 1.0/36.0, 1.0/9.0, 1.0/36.0, 1.0/9.0, 1.0/36.0};

// 3D array for particle distributions F[y][x][i]
using Array3D = std::vector<std::vector<std::vector<double>>>;
using Array2D = std::vector<std::vector<double>>;
using Array2DBool = std::vector<std::vector<bool>>;

/**
 * Initialize 3D array with given dimensions and initial value
 */
Array3D createArray3D(int Ny, int Nx, int NL, double initValue = 0.0) {
    return Array3D(Ny, std::vector<std::vector<double>>(Nx, std::vector<double>(NL, initValue)));
}

/**
 * Initialize 2D array with given dimensions
 */
Array2D createArray2D(int Ny, int Nx, double initValue = 0.0) {
    return Array2D(Ny, std::vector<double>(Nx, initValue));
}

/**
 * Initialize 2D boolean array
 */
Array2DBool createArray2DBool(int Ny, int Nx, bool initValue = false) {
    return Array2DBool(Ny, std::vector<bool>(Nx, initValue));
}

/**
 * Compute distance between two points
 */
inline double distance(int x1, int y1, int x2, int y2) {
    double dx = static_cast<double>(x2 - x1);
    double dy = static_cast<double>(y2 - y1);
    return std::sqrt(dx * dx + dy * dy);
}

/**
 * Create cylinder mask for boundary condition
 */
Array2DBool createCylinderMask(int Nx, int Ny, int centerX, int centerY, double radius) {
    Array2DBool cylinder = createArray2DBool(Ny, Nx, false);
    
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < Ny; ++y) {
        for (int x = 0; x < Nx; ++x) {
            if (distance(centerX, centerY, x, y) < radius) {
                cylinder[y][x] = true;
            }
        }
    }
    
    return cylinder;
}

/**
 * Get indices of cylinder nodes
 */
std::vector<std::pair<int, int>> getCylinderIndices(const Array2DBool& cylinder) {
    std::vector<std::pair<int, int>> indices;
    
    for (int y = 0; y < Ny; ++y) {
        for (int x = 0; x < Nx; ++x) {
            if (cylinder[y][x]) {
                indices.push_back({x, y});
            }
        }
    }
    
    return indices;
}

/**
 * Perform streaming (advection) phase with boundary conditions
 */
void streamingPhase(Array3D& F, const Array2DBool& cylinder, 
                    const std::vector<std::pair<int, int>>& cylinderIndices) {
    Array3D F_temp = F;
    
    // Streaming: advect distributions to neighboring nodes
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < NL; ++i) {
        for (int y = 0; y < Ny; ++y) {
            for (int x = 0; x < Nx; ++x) {
                int x_src = (x - cxs[i] + Nx) % Nx;  // Periodic boundary in x
                int y_src = (y - cys[i] + Ny) % Ny;  // Periodic boundary in y
                F[y][x][i] = F_temp[y_src][x_src][i];
            }
        }
    }
    
    // Edge boundary conditions (inlet/outlet)
    for (int y = 0; y < Ny; ++y) {
        F[y][Nx-1][6] = F[y][Nx-2][6];
        F[y][Nx-1][7] = F[y][Nx-2][7];
        F[y][Nx-1][8] = F[y][Nx-2][8];
        
        F[y][0][2] = F[y][1][2];
        F[y][0][3] = F[y][1][3];
        F[y][0][4] = F[y][1][4];
    }
    
    // Cylinder bounce-back boundary condition
    for (const auto& [x, y] : cylinderIndices) {
        std::swap(F[y][x][1], F[y][x][5]);  // North ↔ South
        std::swap(F[y][x][2], F[y][x][6]);  // NE ↔ SW
        std::swap(F[y][x][3], F[y][x][7]);  // East ↔ West
        std::swap(F[y][x][4], F[y][x][8]);  // SE ↔ NW
    }
}

/**
 * Compute macroscopic quantities (density and velocity)
 */
void computeMacro(const Array3D& F, Array2D& rho, Array2D& ux, Array2D& uy) {
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < Ny; ++y) {
        for (int x = 0; x < Nx; ++x) {
            double rho_local = 0.0;
            double ux_local = 0.0;
            double uy_local = 0.0;
            
            for (int i = 0; i < NL; ++i) {
                rho_local += F[y][x][i];
                ux_local += F[y][x][i] * cxs[i];
                uy_local += F[y][x][i] * cys[i];
            }
            
            rho[y][x] = rho_local;
            ux[y][x] = ux_local / rho_local;
            uy[y][x] = uy_local / rho_local;
        }
    }
}

/**
 * Apply boundary conditions to velocity field
 */
void applyVelocityBC(Array2D& ux, Array2D& uy, const Array2DBool& cylinder) {
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < Ny; ++y) {
        for (int x = 0; x < Nx; ++x) {
            if (cylinder[y][x]) {
                ux[y][x] = 0.0;
                uy[y][x] = 0.0;
            }
        }
    }
}

/**
 * Collision step with equilibrium relaxation
 */
void collisionStep(Array3D& F, const Array2D& rho, const Array2D& ux, const Array2D& uy) {
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < NL; ++i) {
        for (int y = 0; y < Ny; ++y) {
            for (int x = 0; x < Nx; ++x) {
                double cx = static_cast<double>(cxs[i]);
                double cy = static_cast<double>(cys[i]);
                double w = weights[i];
                
                double cu = cx * ux[y][x] + cy * uy[y][x];
                double u_sq = ux[y][x] * ux[y][x] + uy[y][x] * uy[y][x];
                
                double Feq = rho[y][x] * w * (
                    1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq
                );
                
                F[y][x][i] += -(1.0 / tau) * (F[y][x][i] - Feq);
            }
        }
    }
}

/**
 * Compute vorticity field for visualization
 */
Array2D computeVorticity(const Array2D& ux, const Array2D& uy) {
    Array2D curl = createArray2D(Ny - 4, Nx - 4);
    
    #pragma omp parallel for collapse(2)
    for (int y = 2; y < Ny - 2; ++y) {
        for (int x = 1; x < Nx - 1; ++x) {
            double dfydx = ux[y + 2][x] - ux[y - 2][x];
            double dfxdy = uy[y][x + 1] - uy[y][x - 1];
            curl[y - 2][x - 1] = dfydx - dfxdy;
        }
    }
    
    return curl;
}

/**
 * Save vorticity field to file (for post-processing visualization)
 */
void saveVorticityToFile(const Array2D& curl, int timeStep) {
    std::string filename = "vorticity_t" + std::to_string(timeStep) + ".txt";
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    
    for (const auto& row : curl) {
        for (double val : row) {
            file << std::setw(10) << std::setprecision(6) << val << " ";
        }
        file << "\n";
    }
    
    file.close();
}

/**
 * Main LBM simulation
 */
void simulate() {
    std::cout << "=== Lattice Boltzmann Method (D2Q9) ===" << std::endl;
    std::cout << "Domain: " << Nx << " x " << Ny << std::endl;
    std::cout << "Timesteps: " << Nt << std::endl;
    std::cout << "Relaxation time (tau): " << tau << std::endl;
    std::cout << std::endl;
    
    // Initialize particle distributions
    Array3D F = createArray3D(Ny, Nx, NL, 1.0);
    
    // Add small random perturbations
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dis(0.0, 0.01);
    
    #pragma omp parallel for collapse(3)
    for (int y = 0; y < Ny; ++y) {
        for (int x = 0; x < Nx; ++x) {
            for (int i = 0; i < NL; ++i) {
                F[y][x][i] += dis(gen);
            }
        }
    }
    
    // Set higher density in one direction
    for (int y = 0; y < Ny; ++y) {
        for (int x = 0; x < Nx; ++x) {
            F[y][x][3] = 2.3;
        }
    }
    
    // Create cylinder mask
    int cylinderCenterX = Nx / 4;
    int cylinderCenterY = Ny / 2;
    double cylinderRadius = 13.0;
    
    Array2DBool cylinder = createCylinderMask(Nx, Ny, cylinderCenterX, cylinderCenterY, cylinderRadius);
    std::vector<std::pair<int, int>> cylinderIndices = getCylinderIndices(cylinder);
    
    std::cout << "Cylinder: Center (" << cylinderCenterX << ", " << cylinderCenterY 
              << "), Radius " << cylinderRadius << std::endl;
    std::cout << "Cylinder nodes: " << cylinderIndices.size() << std::endl;
    std::cout << std::endl;
    
    // Arrays for macroscopic quantities
    Array2D rho = createArray2D(Ny, Nx);
    Array2D ux = createArray2D(Ny, Nx);
    Array2D uy = createArray2D(Ny, Nx);
    
    // Starting time
    auto t_start = std::chrono::high_resolution_clock::now();
    
    // Main simulation loop
    for (int t = 0; t < Nt; ++t) {
        if (t % 5000 == 0) {
            std::cout << "Iteration " << t << "/" << Nt << std::endl;
        }
        
        // Step 1: Streaming
        streamingPhase(F, cylinder, cylinderIndices);
        
        // Step 2: Compute macroscopic quantities
        computeMacro(F, rho, ux, uy);
        
        // Step 3: Apply boundary conditions
        applyVelocityBC(ux, uy, cylinder);
        
        // Step 4: Collision
        collisionStep(F, rho, ux, uy);
        
        // Visualization output (save vorticity to file every plotEvery steps)
        if (t % plotEvery == 0) {
            Array2D curl = computeVorticity(ux, uy);
            // Uncomment to save to disk (warning: creates many files)
            // saveVorticityToFile(curl, t);
        }
    }
    
    auto t_end = std::chrono::high_resolution_clock::now();
    
    // Calculate and print timing statistics
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start);
    double total_time_seconds = duration.count() / 1000.0;
    double avg_time_ms = duration.count() / static_cast<double>(Nt);
    
    std::cout << "\n=== Simulation Complete ===" << std::endl;
    std::cout << "Total runtime: " << std::fixed << std::setprecision(2) << total_time_seconds << " seconds" << std::endl;
    std::cout << "Average per iteration: " << std::fixed << std::setprecision(2) << avg_time_ms << " ms" << std::endl;
    std::cout << std::endl;
}

int main() {
    simulate();
    return 0;
}
