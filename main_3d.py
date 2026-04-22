"""
3D LBM GUI Application

A Tkinter-based GUI for 3D lattice Boltzmann simulations with CAD file support.
Allows users to import STL/OBJ files, visualize voxelization, and run 3D simulations.

Usage:
    python main_3d.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from pathlib import Path
import threading
import json


class LBM3DGUI:
    """3D LBM Simulator GUI"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("3D LBM Simulator - CAD Import")
        self.root.geometry("1200x800")
        
        self.mesh = None
        self.voxel_grid = None
        self.simulator = None
        self.running = False
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create GUI layout"""
        
        # Top menu bar
        menu = tk.Menu(self.root)
        self.root.config(menu=menu)
        
        file_menu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Import STL/OBJ", command=self.import_file)
        file_menu.add_command(label="Create Test Cylinder", command=self.create_cylinder)
        file_menu.add_command(label="Create Test Sphere", command=self.create_sphere)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Main layout: left panel (controls) + right panel (info/results)
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # LEFT PANEL: Controls
        left_frame = ttk.LabelFrame(main_frame, text="Simulation Controls", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # File info
        ttk.Label(left_frame, text="1. Load CAD File:").pack(anchor=tk.W, pady=(0, 10))
        file_button_frame = ttk.Frame(left_frame)
        file_button_frame.pack(fill=tk.X, pady=(0, 15))
        ttk.Button(file_button_frame, text="Import File", command=self.import_file).pack(side=tk.LEFT)
        ttk.Button(file_button_frame, text="Create Cylinder", command=self.create_cylinder).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_button_frame, text="Create Sphere", command=self.create_sphere).pack(side=tk.LEFT)
        
        self.file_label = ttk.Label(left_frame, text="No file loaded", foreground="red")
        self.file_label.pack(anchor=tk.W, pady=(0, 15))
        
        # Mesh parameters
        ttk.Label(left_frame, text="2. Mesh Parameters:").pack(anchor=tk.W, pady=(0, 10))
        
        param_frame = ttk.Frame(left_frame)
        param_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(param_frame, text="Voxel Resolution:").pack(side=tk.LEFT)
        self.resolution_var = tk.DoubleVar(value=0.1)
        ttk.Scale(param_frame, from_=0.01, to=0.5, variable=self.resolution_var,
                  orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(param_frame, text="(smaller = higher detail)").pack(side=tk.LEFT)
        
        ttk.Button(left_frame, text="Voxelize Mesh", command=self.voxelize).pack(fill=tk.X, pady=(0, 15))
        
        self.voxel_label = ttk.Label(left_frame, text="No mesh voxelized", foreground="red")
        self.voxel_label.pack(anchor=tk.W, pady=(0, 15))
        
        # Simulation parameters
        ttk.Label(left_frame, text="3. Simulation Parameters:").pack(anchor=tk.W, pady=(0, 10))
        
        # Reynolds number
        ttk.Label(left_frame, text="Reynolds Number:").pack(anchor=tk.W)
        self.reynolds_var = tk.IntVar(value=100)
        ttk.Scale(left_frame, from_=10, to=500, variable=self.reynolds_var,
                  orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(0, 5))
        ttk.Label(left_frame, text=f"Re = {self.reynolds_var.get()}").pack(anchor=tk.W, pady=(0, 10))
        
        # Inlet velocity
        ttk.Label(left_frame, text="Inlet Velocity:").pack(anchor=tk.W)
        self.velocity_var = tk.DoubleVar(value=0.1)
        ttk.Scale(left_frame, from_=0.01, to=0.3, variable=self.velocity_var,
                  orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(0, 5))
        ttk.Label(left_frame, text=f"U_inlet = {self.velocity_var.get():.3f}").pack(anchor=tk.W, pady=(0, 10))
        
        # Number of steps
        ttk.Label(left_frame, text="Simulation Steps:").pack(anchor=tk.W)
        self.steps_var = tk.IntVar(value=1000)
        ttk.Spinbox(left_frame, from_=100, to=100000, textvariable=self.steps_var).pack(fill=tk.X, pady=(0, 15))
        
        # Run button
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill=tk.X, pady=(0, 15))
        self.run_button = ttk.Button(button_frame, text="▶ Run Simulation", command=self.run_simulation)
        self.run_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.pause_button = ttk.Button(button_frame, text="⏸ Pause", command=self.pause_simulation, state=tk.DISABLED)
        self.pause_button.pack(side=tk.LEFT, padx=5)
        
        # Progress
        self.progress = ttk.Progressbar(left_frame, mode='determinate', maximum=100)
        self.progress.pack(fill=tk.X, pady=(0, 10))
        
        self.progress_label = ttk.Label(left_frame, text="Ready")
        self.progress_label.pack(anchor=tk.W, pady=(0, 15))
        
        # RIGHT PANEL: Information & Results
        right_frame = ttk.LabelFrame(main_frame, text="Information & Results", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Info display
        self.info_text = tk.Text(right_frame, height=30, width=50, state=tk.DISABLED)
        self.info_text.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(right_frame, orient=tk.VERTICAL, command=self.info_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.info_text['yscrollcommand'] = scrollbar.set
        
        self.display_welcome()
    
    def display_welcome(self):
        """Display welcome message"""
        welcome = """
╔════════════════════════════════════════════════════════════════════════════╗
║                   3D LBM SIMULATOR WITH CAD SUPPORT                        ║
║                         Phase 4 Implementation                             ║
╚════════════════════════════════════════════════════════════════════════════╝

QUICK START:
1. Load a CAD file:
   - Click "Import File" to load STL/OBJ
   - Or create a test shape (Cylinder/Sphere)

2. Configure voxelization:
   - Adjust "Voxel Resolution" (smaller = more detail, slower)
   - Click "Voxelize Mesh" to convert to 3D grid

3. Set simulation parameters:
   - Reynolds number (flow intensity)
   - Inlet velocity
   - Number of simulation steps

4. Run simulation:
   - Click "▶ Run Simulation"
   - Watch progress in real-time
   - Results display below

FEATURES:
✓ D3Q27 Lattice (27-velocity 3D lattice)
✓ STL/OBJ file import
✓ Automatic voxelization (ray-casting)
✓ BGK collision operator
✓ Interactive parameter control
✓ Real-time statistics

PHYSICS:
• Drag force calculation
• Pressure field computation
• Velocity field tracking
• Vorticity analysis (planned)

NEXT STEPS:
1. Load a mesh file
2. Set voxel resolution (0.05-0.2 recommended)
3. Configure flow conditions
4. Run for 1000-10000 steps
5. Analyze results

For 3D visualization, see PHASE4_3D_LBM_GUIDE.md
        """
        self.update_info(welcome)
    
    def update_info(self, text):
        """Update info display"""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, text)
        self.info_text.config(state=tk.DISABLED)
    
    def import_file(self):
        """Import CAD file"""
        filepath = filedialog.askopenfilename(
            title="Select CAD File",
            filetypes=[("STL files", "*.stl"), ("OBJ files", "*.obj"), ("All files", "*.*")]
        )
        
        if not filepath:
            return
        
        try:
            from mesh_loader import MeshLoader
            
            self.file_label.config(text=f"Loading {Path(filepath).name}...", foreground="blue")
            self.root.update()
            
            loader = MeshLoader(filepath)
            self.mesh = loader.load()
            
            info = f"""
MESH LOADED: {self.mesh.name}
{'=' * 60}

Vertices: {self.mesh.num_vertices:,}
Faces: {self.mesh.num_faces:,}

Bounds:
  Min: {self.mesh.bounds[0]}
  Max: {self.mesh.bounds[1]}

Validation:
"""
            for msg in self.mesh.validate():
                info += f"  {msg}\n"
            
            self.file_label.config(text=f"✓ {Path(filepath).name}", foreground="green")
            self.update_info(info)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{e}")
            self.file_label.config(text="Error loading file", foreground="red")
    
    def create_cylinder(self):
        """Create test cylinder"""
        try:
            from mesh_loader import create_simple_cylinder
            
            self.mesh = create_simple_cylinder(radius=1.0, height=2.0, resolution=32)
            
            info = f"""
TEST MESH CREATED: Cylinder
{'=' * 60}

Vertices: {self.mesh.num_vertices:,}
Faces: {self.mesh.num_faces:,}

Bounds:
  Min: {self.mesh.bounds[0]}
  Max: {self.mesh.bounds[1]}

Validation:
"""
            for msg in self.mesh.validate():
                info += f"  {msg}\n"
            
            self.file_label.config(text="✓ Test Cylinder", foreground="green")
            self.update_info(info)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create cylinder:\n{e}")
    
    def create_sphere(self):
        """Create test sphere"""
        try:
            from mesh_loader import create_simple_sphere
            
            self.mesh = create_simple_sphere(radius=1.0, resolution=16)
            
            info = f"""
TEST MESH CREATED: Sphere
{'=' * 60}

Vertices: {self.mesh.num_vertices:,}
Faces: {self.mesh.num_faces:,}

Bounds:
  Min: {self.mesh.bounds[0]}
  Max: {self.mesh.bounds[1]}

Validation:
"""
            for msg in self.mesh.validate():
                info += f"  {msg}\n"
            
            self.file_label.config(text="✓ Test Sphere", foreground="green")
            self.update_info(info)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create sphere:\n{e}")
    
    def voxelize(self):
        """Voxelize mesh"""
        if self.mesh is None:
            messagebox.showwarning("Warning", "Please load a mesh first")
            return
        
        try:
            from voxelizer import Voxelizer
            
            self.voxel_label.config(text="Voxelizing...", foreground="blue")
            self.root.update()
            
            resolution = self.resolution_var.get()
            voxelizer = Voxelizer(resolution=resolution)
            self.voxel_grid = voxelizer.voxelize(self.mesh)
            
            self.voxel_label.config(text=f"✓ Voxelized ({self.voxel_grid.shape[0]}×{self.voxel_grid.shape[1]}×{self.voxel_grid.shape[2]})", 
                                   foreground="green")
            
            info = f"""
MESH VOXELIZED
{'=' * 60}

Grid Resolution: {resolution}

Grid Dimensions:
  {self.voxel_grid.shape[0]} × {self.voxel_grid.shape[1]} × {self.voxel_grid.shape[2]}
  = {self.voxel_grid.num_voxels:,} total voxels

Solid Voxels: {self.voxel_grid.num_solid:,} ({100*self.voxel_grid.num_solid/self.voxel_grid.num_voxels:.1f}%)
Fluid Voxels: {self.voxel_grid.num_fluid:,} ({100*self.voxel_grid.num_fluid/self.voxel_grid.num_voxels:.1f}%)

Physical Bounds:
  Min: {self.voxel_grid.bounds[0]}
  Max: {self.voxel_grid.bounds[1]}

Memory Estimate:
  Base lattice: {248*self.voxel_grid.num_voxels/1e9:.2f} GB (27 distributions)
  GPU memory (f32): ~{250*self.voxel_grid.num_voxels/1e9:.2f} GB

Ready for simulation!
"""
            self.update_info(info)
            
        except Exception as e:
            messagebox.showerror("Error", f"Voxelization failed:\n{e}")
            self.voxel_label.config(text="Voxelization failed", foreground="red")
    
    def run_simulation(self):
        """Run 3D LBM simulation in background thread"""
        if self.voxel_grid is None:
            messagebox.showwarning("Warning", "Please voxelize a mesh first")
            return
        
        self.running = True
        self.run_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.NORMAL)
        
        # Run in background thread
        thread = threading.Thread(target=self._run_simulation_thread)
        thread.start()
    
    def _run_simulation_thread(self):
        """Background simulation thread"""
        try:
            from lbm_3d import LBM3D
            
            reynolds = self.reynolds_var.get()
            velocity = self.velocity_var.get()
            num_steps = self.steps_var.get()
            
            # Create simulator
            self.simulator = LBM3D(self.voxel_grid, reynolds=reynolds, inlet_velocity=velocity)
            
            info = f"""
3D LBM SIMULATION RUNNING
{'=' * 60}

Parameters:
  Reynolds: {reynolds}
  Inlet velocity: {velocity}
  Grid: {self.voxel_grid.shape[0]}×{self.voxel_grid.shape[1]}×{self.voxel_grid.shape[2]}

Simulation progress:
"""
            self.update_info(info)
            
            # Run steps
            stats_history = []
            for step in range(num_steps):
                if not self.running:
                    break
                
                self.simulator.step()
                stats = self.simulator.get_statistics()
                stats_history.append(stats)
                
                # Update progress every 100 steps
                if (step + 1) % max(1, num_steps // 10) == 0:
                    progress = 100 * (step + 1) / num_steps
                    self.progress['value'] = progress
                    self.progress_label.config(
                        text=f"Step {step+1}/{num_steps} ({progress:.0f}%) - "
                             f"v_max={stats['max_velocity']:.4f}, "
                             f"Cd={stats['drag']:.4f}, "
                             f"t={stats['time_per_step']*1000:.2f}ms"
                    )
                    self.root.update()
            
            # Display results
            if stats_history:
                result_info = f"""
3D LBM SIMULATION COMPLETE
{'=' * 60}

Final Statistics:
  Steps completed: {len(stats_history)}
  Max velocity: {stats_history[-1]['max_velocity']:.6f}
  Min pressure: {stats_history[-1]['min_pressure']:.6f}
  Max pressure: {stats_history[-1]['max_pressure']:.6f}
  Drag force: {stats_history[-1]['drag']:.6f}

Performance:
  Time per step: {1000*np.mean([s['time_per_step'] for s in stats_history]):.2f} ms
  Simulated time: {len(stats_history) * 1e-3:.2f} (lattice units)

Velocity History:
  Initial: {stats_history[0]['max_velocity']:.6f}
  Final: {stats_history[-1]['max_velocity']:.6f}
  Change: {100*(stats_history[-1]['max_velocity']-stats_history[0]['max_velocity'])/max(stats_history[0]['max_velocity'], 1e-10):.1f}%

Simulation completed successfully!

NEXT STEPS:
1. Export results to VTK (for ParaView)
2. Generate pressure/velocity slices
3. Analyze vortex structures
4. Compare with experimental data
"""
                self.update_info(result_info)
        
        except Exception as e:
            messagebox.showerror("Simulation Error", str(e))
        
        finally:
            self.running = False
            self.progress['value'] = 0
            self.progress_label.config(text="Ready")
            self.run_button.config(state=tk.NORMAL)
            self.pause_button.config(state=tk.DISABLED)
    
    def pause_simulation(self):
        """Pause simulation"""
        self.running = False
        self.pause_button.config(state=tk.DISABLED)
        self.run_button.config(state=tk.NORMAL)


def main():
    """Launch GUI"""
    root = tk.Tk()
    app = LBM3DGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
