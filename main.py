import numpy as np
from matplotlib import pyplot
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numba
import time
import tkinter as tk
from tkinter import ttk, messagebox
import threading
try:
    from scipy.fft import fft
except ImportError:
    # Fallback if scipy not available
    from numpy.fft import fft
import json
from pathlib import Path

plotEvery = 25
STOP_SIMULATION = False

# Color scheme definitions
COLOR_SCHEMES = {
    'Red-Blue (bwr)': 'bwr',
    'Hot': 'hot',
    'Cool': 'cool',
    'RdYlBu': 'RdYlBu',
    'Viridis': 'viridis',
    'Plasma': 'plasma',
    'Twilight': 'twilight',
}

@numba.jit(nopython=True)
def distance(x1, y1, x2, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

def run_simulation(Nt, colormap, update_callback, completion_callback):
    """Run the LBM simulation with Phase 1 metrics tracking."""
    global STOP_SIMULATION
    STOP_SIMULATION = False
    
    # Space
    Nx = 400 
    Ny = 100

    tau = .53

    #lattice speeds and weights 
    NL = 9 # Number of Lattices
    
    cxs = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
    cys = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
    weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])

    # initial conditions
    F = np.ones((Ny, Nx, NL)) + .01 * np.random.randn(Ny, Nx, NL)
    F[:, :, 3] = 2.3

    # Vectorized cylinder mask creation
    y_coords = np.arange(Ny)[:, np.newaxis]
    x_coords = np.arange(Nx)[np.newaxis, :]
    distances = np.sqrt((x_coords - Nx // 4) ** 2 + (y_coords - Ny // 2) ** 2)
    cylinder = distances < 13 

    # Pre-compute cylinder indices for faster boundary condition application
    cylinder_indices = np.where(cylinder)
    
    # PHASE 1: Initialize metrics tracker
    metrics = MetricsTracker()
    U_ref = 0.1
    D = 26
    
    # Monitor point for vorticity (wake location)
    monitor_x = Nx // 4 + 40
    monitor_y = Ny // 2

    # main loop
    t_start = time.perf_counter()
    for t in range(Nt):
        if STOP_SIMULATION:
            break
            
        if t % 5000 == 0:
            print(f"Iteration {t}/{Nt}")

        # Streaming with boundary conditions - vectorized numpy operations
        for i in range(NL):
            F[:, :, i] = np.roll(F[:, :, i], cxs[i], axis=1)
            F[:, :, i] = np.roll(F[:, :, i], cys[i], axis=0)
        
        # Edge boundary conditions
        F[:, -1, 6] = F[:, -2, 6]
        F[:, -1, 7] = F[:, -2, 7]
        F[:, -1, 8] = F[:, -2, 8]
        
        F[:, 0, 2] = F[:, 1, 2]
        F[:, 0, 3] = F[:, 1, 3]
        F[:, 0, 4] = F[:, 1, 4]
        
        # Cylinder bounce-back boundary condition
        for idx in range(len(cylinder_indices[0])):
            y, x = cylinder_indices[0][idx], cylinder_indices[1][idx]
            # Swap opposite directions
            F[y, x, 1], F[y, x, 5] = F[y, x, 5], F[y, x, 1]
            F[y, x, 2], F[y, x, 6] = F[y, x, 6], F[y, x, 2]
            F[y, x, 3], F[y, x, 7] = F[y, x, 7], F[y, x, 3]
            F[y, x, 4], F[y, x, 8] = F[y, x, 8], F[y, x, 4]
        
        # Compute macroscopic variables
        rho = np.sum(F, 2)
        ux = np.sum(F * cxs, 2) / rho
        uy = np.sum(F * cys, 2) / rho
        
        # Apply boundary conditions to velocity
        ux[cylinder] = 0
        uy[cylinder] = 0
        
        # Collision step (JIT compiled)
        _collision_step(F, rho, ux, uy, cxs, cys, weights, tau)

        # PHASE 1: Compute metrics every 50 iterations
        if t % 50 == 0:
            # Compute drag and lift
            cd, fx = compute_drag_coefficient(F, cylinder_indices, rho, ux, uy, cxs, cys, weights, D, U_ref)
            cl, fy = compute_lift_coefficient(F, cylinder_indices, rho, ux, uy, cxs, cys, weights, D, U_ref)
            
            # Compute vorticity at monitor point
            vorticity_point = ux[monitor_y+1, monitor_x] - ux[monitor_y-1, monitor_x] - \
                             (uy[monitor_y, monitor_x+1] - uy[monitor_y, monitor_x-1])
            
            # Compute kinetic energy and enstrophy
            ke = 0.5 * np.sum(rho * (ux**2 + uy**2)) / np.sum(rho)
            enstrophy = 0.5 * np.sum((uy[1:, :-1] - uy[:-1, :-1])**2 + (ux[:-1, 1:] - ux[:-1, :-1])**2) / np.sum(rho)
            
            metrics.record(t, cd, cl, vorticity_point, ke, enstrophy)

        # Update GUI every plotEvery iterations
        if (t % plotEvery == 0):
            dfydx = ux[2:, 1:-1] - ux[0:-2, 1:-1]
            dfxdy = uy[1:-1, 2:] - uy[1:-1, 0:-2]
            curl = dfydx - dfxdy

            update_callback(curl, colormap, t, Nt)
    
    t_end = time.perf_counter()
    elapsed = t_end - t_start
    print(f"\nTotal runtime: {elapsed:.2f} seconds")
    print(f"Average per iteration: {elapsed / max(t, 1) * 1000:.2f} ms")
    
    # PHASE 1: Compute and display Strouhal number with quality metric
    st, f_dom, quality = metrics.compute_strouhal(dt=1.0, U_ref=U_ref, D=D)
    stats = metrics.get_statistics()
    
    print(f"\n=== PHASE 1 METRICS ===")
    print(f"Drag Coefficient (Cd):  {stats['cd_mean']:.4f} ± {stats['cd_std']:.4f}")
    print(f"Lift Coefficient (Cl):  {stats['cl_mean']:.4f} ± {stats['cl_std']:.4f}")
    print(f"Strouhal Number (St):   {st:.4f}")
    print(f"Dominant Frequency:     {f_dom:.4f} Hz")
    print(f"Signal Quality (FFT):    {quality:.1%}")
    print(f"Mean Kinetic Energy:    {np.mean(metrics.kinetic_energy):.6f}")
    
    # Validation summary
    cd_error = abs(stats['cd_mean']-1.465)/1.465*100
    st_valid = 0.16 <= st <= 0.20
    
    print(f"\n=== BENCHMARK COMPARISON (Re=40) ===")
    print(f"Cd: Expected ≈ 1.465  | Your result: {stats['cd_mean']:.4f} | Error: {cd_error:.2f}%")
    print(f"St: Expected ≈ 0.17   | Your result: {st:.4f} | Valid: {'✓' if st_valid else '✗'}")
    
    if cd_error < 2 and st_valid and quality > 0.3:
        print(f"\n✅ PHASE 1 VALIDATION SUCCESSFUL!")
    
    # Save metrics
    metrics.save_metrics('phase1_metrics.json')
    print(f"\nMetrics saved to phase1_metrics.json")
    
    completion_callback(elapsed, t, metrics)




@numba.jit(nopython=True)
def _collision_step(F, rho, ux, uy, cxs, cys, weights, tau):
    """Collision step with equilibrium calculation (Numba JIT compiled)."""
    Ny, Nx, NL = F.shape
    
    for i in range(NL):
        cx, cy, w = cxs[i], cys[i], weights[i]
        
        for y in range(Ny):
            for x in range(Nx):
                cu = cx * ux[y, x] + cy * uy[y, x]
                Feq = rho[y, x] * w * (
                    1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * (ux[y, x] * ux[y, x] + uy[y, x] * uy[y, x])
                )
                F[y, x, i] += -(1.0 / tau) * (F[y, x, i] - Feq)


# ============================================================================
# PHASE 1: CORE PHYSICAL METRICS
# ============================================================================

def compute_equilibrium(rho, ux, uy, cxs, cys, weights):
    """Compute equilibrium distribution f_eq."""
    Ny, Nx = rho.shape
    NL = len(weights)
    f_eq = np.zeros((Ny, Nx, NL))
    
    for i in range(NL):
        cx, cy, w = cxs[i], cys[i], weights[i]
        cu = cx * ux + cy * uy
        u_sq = ux**2 + uy**2
        f_eq[:, :, i] = rho * w * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*u_sq)
    
    return f_eq


def compute_drag_coefficient(F, cylinder_indices, rho, ux, uy, cxs, cys, weights, D=26, U_ref=0.1):
    """
    PHASE 1: Compute drag coefficient using momentum exchange method.
    
    This is the preferred method for LBM - exploits natural structure of non-equilibrium stress.
    C_d = F_drag / (0.5 * rho * U_ref^2 * A)
    
    Args:
        F: Distribution function [Ny, Nx, 9]
        cylinder_indices: Indices of cylinder cells
        rho, ux, uy: Macroscopic variables
        cxs, cys: Lattice velocity components
        weights: Lattice weights
        D: Cylinder diameter in lattice units
        U_ref: Reference velocity (inlet velocity)
    
    Returns:
        Cd: Drag coefficient (dimensionless)
    """
    cs_squared = 1.0 / 3.0  # Speed of sound squared for D2Q9
    
    # Compute equilibrium and non-equilibrium distributions
    f_eq = compute_equilibrium(rho, ux, uy, cxs, cys, weights)
    f_neq = F - f_eq
    
    # Sum non-equilibrium stress on cylinder nodes (momentum exchange method)
    F_x = 0.0
    F_y = 0.0
    
    for idx in range(len(cylinder_indices[0])):
        y, x = cylinder_indices[0][idx], cylinder_indices[1][idx]
        for i in range(9):
            F_x += f_neq[y, x, i] * cxs[i]
            F_y += f_neq[y, x, i] * cys[i]
    
    F_x *= cs_squared
    
    # Compute dimensionless drag coefficient
    rho_ref = 1.0
    denom = 0.5 * rho_ref * U_ref**2 * D
    
    if denom != 0:
        Cd = F_x / denom
    else:
        Cd = 0.0
    
    return Cd, F_x


def compute_lift_coefficient(F, cylinder_indices, rho, ux, uy, cxs, cys, weights, D=26, U_ref=0.1):
    """
    PHASE 1: Compute lift coefficient from transverse force.
    
    Args:
        Same as compute_drag_coefficient
    
    Returns:
        Cl: Lift coefficient (dimensionless)
    """
    cs_squared = 1.0 / 3.0
    
    f_eq = compute_equilibrium(rho, ux, uy, cxs, cys, weights)
    f_neq = F - f_eq
    
    # Sum transverse force
    F_y = 0.0
    
    for idx in range(len(cylinder_indices[0])):
        y, x = cylinder_indices[0][idx], cylinder_indices[1][idx]
        for i in range(9):
            F_y += f_neq[y, x, i] * cys[i]
    
    F_y *= cs_squared
    
    # Compute lift coefficient
    rho_ref = 1.0
    denom = 0.5 * rho_ref * U_ref**2 * D
    
    if denom != 0:
        Cl = F_y / denom
    else:
        Cl = 0.0
    
    return Cl, F_y


def compute_strouhal_number(vorticity_history, dt=0.1, U_ref=0.1, D=26):
    """
    PHASE 1: Compute Strouhal number from vorticity time series using enhanced FFT.
    
    St = f * D / U_ref, where f is the dominant frequency in the wake
    
    Args:
        vorticity_history: List of vorticity values at cylinder location
        dt: Time step (in physical units)
        U_ref: Reference velocity
        D: Cylinder diameter
    
    Returns:
        St: Strouhal number (dimensionless)
        f_dominant: Dominant frequency (Hz)
        quality: Signal quality (0-1, higher = better resolution)
    """
    if len(vorticity_history) < 100:
        return 0.0, 0.0, 0.0
    
    # Use up to 4000 points for better frequency resolution
    n_points = min(len(vorticity_history), 4000)
    data = np.array(vorticity_history[-n_points:])
    
    # Remove mean and apply Hanning window (reduces spectral leakage)
    data = data - np.mean(data)
    window = np.hanning(len(data))
    data = data * window
    
    # Apply FFT
    N = len(data)
    fft_vals = np.abs(fft(data))
    freqs = np.fft.fftfreq(N, dt)
    
    # Find dominant frequency (exclude DC and look in reasonable shedding range)
    # For cylinder at Re~40, St~0.17, so f = St*U/D = 0.17*0.1/26 ~ 0.0006 Hz
    # But we're in lattice units, so adjust search range
    valid_idx = np.where((freqs > 0.0001) & (freqs < 0.1/dt))[0]
    
    if len(valid_idx) == 0:
        return 0.0, 0.0, 0.0
    
    # Find peak in valid range
    dominant_idx = valid_idx[np.argmax(fft_vals[valid_idx])]
    f_dominant = np.abs(freqs[dominant_idx])
    
    # Compute signal quality (peak vs background energy ratio)
    peak_energy = fft_vals[dominant_idx]
    background_energy = np.mean(fft_vals[valid_idx])
    quality = min(peak_energy / (background_energy + 1e-10) / 10.0, 1.0)
    
    # Compute Strouhal number
    St = f_dominant * D / U_ref
    
    return St, f_dominant, quality



def export_to_vtk(filename, ux, uy, rho, vorticity, cylinder, Nx, Ny):
    """
    PHASE 1: Export simulation data to VTK format for ParaView visualization.
    
    Creates a VTK rectilinear grid with velocity, density, and vorticity fields.
    """
    try:
        import vtk
        from vtk.util import numpy_support
    except ImportError:
        print("VTK not available. Install with: pip install vtk")
        return False
    
    # Create VTK rectilinear grid
    grid = vtk.vtkRectilinearGrid()
    grid.SetDimensions(Nx + 1, Ny + 1, 2)
    
    # Set coordinates
    x_coords = vtk.vtkFloatArray()
    for i in range(Nx + 1):
        x_coords.InsertNextValue(float(i))
    grid.SetXCoordinates(x_coords)
    
    y_coords = vtk.vtkFloatArray()
    for i in range(Ny + 1):
        y_coords.InsertNextValue(float(i))
    grid.SetYCoordinates(y_coords)
    
    z_coords = vtk.vtkFloatArray()
    for i in range(2):
        z_coords.InsertNextValue(float(i))
    grid.SetZCoordinates(z_coords)
    
    # Add velocity field
    velocity = np.zeros((Ny, Nx, 3))
    velocity[:, :, 0] = ux
    velocity[:, :, 1] = uy
    velocity_vtk = numpy_support.numpy_to_vtk(velocity.reshape(-1, 3))
    velocity_vtk.SetNumberOfComponents(3)
    velocity_vtk.SetName("Velocity")
    grid.GetPointData().AddArray(velocity_vtk)
    
    # Add density field
    rho_flat = rho.flatten()
    rho_vtk = numpy_support.numpy_to_vtk(rho_flat)
    rho_vtk.SetName("Density")
    grid.GetPointData().AddArray(rho_vtk)
    
    # Add vorticity field
    vort_flat = vorticity.flatten()
    vort_vtk = numpy_support.numpy_to_vtk(vort_flat)
    vort_vtk.SetName("Vorticity")
    grid.GetPointData().AddArray(vort_vtk)
    
    # Add cylinder mask
    cylinder_flat = cylinder.astype(np.float32).flatten()
    cyl_vtk = numpy_support.numpy_to_vtk(cylinder_flat)
    cyl_vtk.SetName("Cylinder")
    grid.GetPointData().AddArray(cyl_vtk)
    
    # Write to file
    writer = vtk.vtkRectilinearGridWriter()
    writer.SetFileName(filename)
    writer.SetInputData(grid)
    writer.Write()
    
    return True


def compute_pressure_coefficient(rho, U_ref=0.1, rho_ref=1.0):
    """
    PHASE 1: Compute pressure coefficient field.
    Cp = (p - p_ref) / (0.5 * rho * U_ref^2)
    """
    cs_squared = 1.0 / 3.0
    p = rho * cs_squared  # Pressure = rho * cs^2
    p_ref = rho_ref * cs_squared
    
    denom = 0.5 * rho_ref * U_ref**2
    Cp = (p - p_ref) / denom if denom != 0 else np.zeros_like(p)
    
    return Cp


class MetricsTracker:
    """Track Phase 1 metrics over simulation time."""
    
    def __init__(self):
        self.time_steps = []
        self.cd_values = []
        self.cl_values = []
        self.vorticity_signal = []
        self.st_values = []
        self.kinetic_energy = []
        self.enstrophy = []
    
    def record(self, t, cd, cl, vort_point, ke, enst):
        """Record metrics at time step t."""
        self.time_steps.append(t)
        self.cd_values.append(cd)
        self.cl_values.append(cl)
        self.vorticity_signal.append(vort_point)
        self.kinetic_energy.append(ke)
        self.enstrophy.append(enst)
    
    def compute_strouhal(self, dt=0.1, U_ref=0.1, D=26):
        """Compute Strouhal number from recorded data with quality metric."""
        st, f, quality = compute_strouhal_number(self.vorticity_signal, dt, U_ref, D)
        return st, f, quality
    
    def save_metrics(self, filename='metrics.json'):
        """Save metrics to JSON file."""
        data = {
            'time_steps': self.time_steps,
            'cd': self.cd_values,
            'cl': self.cl_values,
            'ke': self.kinetic_energy,
            'enstrophy': self.enstrophy,
        }
        with open(filename, 'w') as f:
            json.dump(data, f)
    
    def get_statistics(self):
        """Get mean values after settling time."""
        settle_idx = len(self.cd_values) // 2  # Skip first half as settling time
        
        if settle_idx >= len(self.cd_values):
            return {
                'cd_mean': 0, 'cd_std': 0,
                'cl_mean': 0, 'cl_std': 0,
            }
        
        cd_settled = np.array(self.cd_values[settle_idx:])
        cl_settled = np.array(self.cl_values[settle_idx:])
        
        return {
            'cd_mean': np.mean(cd_settled),
            'cd_std': np.std(cd_settled),
            'cl_mean': np.mean(cl_settled),
            'cl_std': np.std(cl_settled),
        }
    
    def check_convergence(self, tolerance=0.02):
        """Check if simulation has converged (Cd within tolerance)."""
        if len(self.cd_values) < 100:
            return False, "Not enough data"
        
        # Compare second half to first half
        split_idx = len(self.cd_values) // 2
        cd_first = np.array(self.cd_values[:split_idx])
        cd_second = np.array(self.cd_values[split_idx:])
        
        first_mean = np.mean(cd_first)
        second_mean = np.mean(cd_second)
        second_std = np.std(cd_second)
        
        # Check if std is within tolerance
        is_converged = second_std < tolerance
        
        return is_converged, {
            'first_half_mean': first_mean,
            'second_half_mean': second_mean,
            'second_half_std': second_std,
            'change': abs(second_mean - first_mean) / max(abs(first_mean), 1e-10),
        }



class LBMSimulatorGUI:
    """GUI for Lattice Boltzmann Method Simulation."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("LBM Fluid Dynamics Simulator")
        self.root.geometry("900x700")
        
        self.sim_thread = None
        self.is_running = False
        
        self.setup_ui()
    
    def setup_ui(self):
        """Create the GUI layout."""
        # Top Control Frame
        control_frame = ttk.LabelFrame(self.root, text="Simulation Controls", padding=10)
        control_frame.pack(fill="x", padx=10, pady=10)
        
        # Iterations Input
        iterations_frame = ttk.Frame(control_frame)
        iterations_frame.pack(fill="x", pady=5)
        ttk.Label(iterations_frame, text="Number of Iterations:", width=20).pack(side="left")
        self.iterations_var = tk.StringVar(value="30000")
        iterations_spin = ttk.Spinbox(
            iterations_frame,
            from_=100,
            to=1000000,
            textvariable=self.iterations_var,
            width=15
        )
        iterations_spin.pack(side="left", padx=5)
        ttk.Label(iterations_frame, text="(100 - 1,000,000)", foreground="gray").pack(side="left")
        
        # Color Scheme Selection
        colormap_frame = ttk.Frame(control_frame)
        colormap_frame.pack(fill="x", pady=5)
        ttk.Label(colormap_frame, text="Color Scheme:", width=20).pack(side="left")
        self.colormap_var = tk.StringVar(value="Red-Blue (bwr)")
        colormap_combo = ttk.Combobox(
            colormap_frame,
            textvariable=self.colormap_var,
            values=list(COLOR_SCHEMES.keys()),
            state="readonly",
            width=20
        )
        colormap_combo.pack(side="left", padx=5)
        
        # Button Frame
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill="x", pady=5)
        
        self.start_button = ttk.Button(button_frame, text="Start Simulation", command=self.start_simulation)
        self.start_button.pack(side="left", padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop Simulation", command=self.stop_simulation, state="disabled")
        self.stop_button.pack(side="left", padx=5)
        
        ttk.Button(button_frame, text="Clear Plot", command=self.clear_plot).pack(side="left", padx=5)
        
        # Status Frame
        status_frame = ttk.LabelFrame(self.root, text="Status", padding=10)
        status_frame.pack(fill="x", padx=10, pady=5)
        
        self.status_label = ttk.Label(status_frame, text="Ready", foreground="green")
        self.status_label.pack(anchor="w")
        
        self.progress_label = ttk.Label(status_frame, text="")
        self.progress_label.pack(anchor="w")
        
        # Canvas Frame for matplotlib
        canvas_frame = ttk.LabelFrame(self.root, text="Vorticity Visualization", padding=5)
        canvas_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.fig = Figure(figsize=(8, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Vorticity Field (curl of velocity)")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        self.im = None
    
    def update_plot(self, data, colormap, iteration, total_iterations):
        """Update the plot with new vorticity data."""
        self.ax.clear()
        self.im = self.ax.imshow(data, cmap=COLOR_SCHEMES[colormap])
        self.ax.set_title(f"Vorticity Field (Iteration {iteration}/{total_iterations})")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        
        if self.im:
            if not hasattr(self, 'cbar') or self.cbar is None:
                self.cbar = self.fig.colorbar(self.im, ax=self.ax)
            else:
                self.cbar.update_normal(self.im)
        
        self.canvas.draw()
        self.progress_label.config(text=f"Progress: {iteration}/{total_iterations} iterations")
        self.root.update()
    
    def simulation_complete(self, elapsed_time, total_iterations, metrics=None):
        """Called when simulation completes."""
        self.is_running = False
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.status_label.config(text=f"Complete! (Total time: {elapsed_time:.2f}s, Avg: {elapsed_time/max(total_iterations,1)*1000:.2f}ms/iter)", foreground="blue")
        
        # Prepare detailed message with Phase 1 metrics
        msg = f"Simulation completed!\n\nTotal time: {elapsed_time:.2f} seconds\nAverage per iteration: {elapsed_time/max(total_iterations,1)*1000:.2f} ms"
        
        if metrics:
            stats = metrics.get_statistics()
            st, f = metrics.compute_strouhal()
            msg += f"\n\n=== PHASE 1 METRICS ===\n"
            msg += f"Drag Coefficient: {stats['cd_mean']:.4f} ± {stats['cd_std']:.4f}\n"
            msg += f"Lift Coefficient: {stats['cl_mean']:.4f} ± {stats['cl_std']:.4f}\n"
            msg += f"Strouhal Number: {st:.4f}\n"
            msg += f"\nBenchmark (Re=40): Expected Cd ≈ 1.465"
        
        messagebox.showinfo("Simulation Complete", msg)

    
    def start_simulation(self):
        """Start the simulation in a separate thread."""
        try:
            iterations = int(self.iterations_var.get())
            if iterations < 100 or iterations > 1000000:
                messagebox.showerror("Invalid Input", "Iterations must be between 100 and 1,000,000")
                return
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number of iterations")
            return
        
        self.is_running = True
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.status_label.config(text="Running...", foreground="orange")
        self.progress_label.config(text="Starting simulation...")
        
        colormap = self.colormap_var.get()
        
        # Run simulation in separate thread to keep GUI responsive
        self.sim_thread = threading.Thread(
            target=run_simulation,
            args=(iterations, colormap, self.update_plot, self.simulation_complete),
            daemon=True
        )
        self.sim_thread.start()
    
    def stop_simulation(self):
        """Stop the running simulation."""
        global STOP_SIMULATION
        STOP_SIMULATION = True
        self.status_label.config(text="Stopping...", foreground="red")
        self.stop_button.config(state="disabled")
    
    def clear_plot(self):
        """Clear the plot."""
        self.ax.clear()
        self.ax.set_title("Vorticity Field (cleared)")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.canvas.draw()
        self.progress_label.config(text="")


def main():
    """Launch the GUI application."""
    root = tk.Tk()
    app = LBMSimulatorGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()