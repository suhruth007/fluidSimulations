import numpy as np
from matplotlib import pyplot
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numba
import time
import tkinter as tk
from tkinter import ttk, messagebox
import threading

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
    """Run the LBM simulation with callbacks for GUI updates."""
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
    
    completion_callback(elapsed, t)


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
    
    def simulation_complete(self, elapsed_time, total_iterations):
        """Called when simulation completes."""
        self.is_running = False
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.status_label.config(text=f"Complete! (Total time: {elapsed_time:.2f}s, Avg: {elapsed_time/max(total_iterations,1)*1000:.2f}ms/iter)", foreground="blue")
        messagebox.showinfo("Simulation Complete", 
            f"Simulation completed!\n\nTotal time: {elapsed_time:.2f} seconds\nAverage per iteration: {elapsed_time/max(total_iterations,1)*1000:.2f} ms")
    
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