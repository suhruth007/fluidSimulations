"""
PHASE 4: 3D LBM with CAD Import - QUICK START GUIDE

Get started with 3D aerodynamic simulations in minutes!
"""

# ============================================================================
# INSTALLATION
# ============================================================================

"""
Step 1: Install dependencies
"""
pip install numpy scipy scikit-image numba


# ============================================================================
# TRY IT IN 5 MINUTES
# ============================================================================

# 1. CREATE TEST GEOMETRY (no CAD file needed)
# ────────────────────────────────────────────────────────────────────────

from mesh_loader import create_simple_cylinder, create_simple_sphere

# Create a test cylinder
mesh = create_simple_cylinder(radius=1.0, height=2.0, resolution=32)
print(f"Cylinder: {mesh.num_vertices} vertices, {mesh.num_faces} faces")

# Or create a sphere
mesh = create_simple_sphere(radius=1.0, resolution=16)
print(f"Sphere: {mesh.num_vertices} vertices, {mesh.num_faces} faces")


# 2. VOXELIZE THE MESH (convert to 3D grid)
# ────────────────────────────────────────────────────────────────────────

from voxelizer import Voxelizer

voxelizer = Voxelizer(resolution=0.1)  # 1 voxel = 0.1 mesh units
voxel_grid = voxelizer.voxelize(mesh)

print(f"Grid: {voxel_grid.shape[0]}×{voxel_grid.shape[1]}×{voxel_grid.shape[2]}")
print(f"Solid voxels: {voxel_grid.num_solid}")
print(f"Fluid voxels: {voxel_grid.num_fluid}")


# 3. CREATE SIMULATOR
# ────────────────────────────────────────────────────────────────────────

from lbm_3d import LBM3D

sim = LBM3D(
    voxel_grid=voxel_grid,
    reynolds=100,           # Flow intensity
    inlet_velocity=0.1,     # Inlet velocity
)


# 4. RUN SIMULATION (10 steps)
# ────────────────────────────────────────────────────────────────────────

print("\nRunning simulation...")
for step in range(10):
    sim.step()
    stats = sim.get_statistics()
    print(f"Step {step:2d}: "
          f"v_max={stats['max_velocity']:.4f}, "
          f"Cd={stats['drag']:.4f}, "
          f"t={stats['time_per_step']*1000:.1f}ms")


# 5. GET RESULTS
# ────────────────────────────────────────────────────────────────────────

pressure = sim.get_pressure()
velocity = sim.get_velocity_magnitude()
drag = sim.get_drag()

print(f"\nResults:")
print(f"  Pressure range: [{pressure.min():.6f}, {pressure.max():.6f}]")
print(f"  Velocity range: [{velocity.min():.6f}, {velocity.max():.6f}]")
print(f"  Total drag: {drag:.6f}")


# ============================================================================
# IMPORT YOUR OWN CAD FILE
# ============================================================================

"""
To import your own STL or OBJ file:
"""

from mesh_loader import MeshLoader

# Load from file
loader = MeshLoader('path/to/your_model.stl')
mesh = loader.load()

print(f"Loaded: {mesh.name}")
print(f"  Vertices: {mesh.num_vertices}")
print(f"  Faces: {mesh.num_faces}")

# Validate mesh
for msg in mesh.validate():
    print(f"  {msg}")

# Continue with voxelization and simulation...


# ============================================================================
# LAUNCH GUI
# ============================================================================

"""
For interactive 3D LBM with GUI:
"""

# python main_3d.py

# Then:
# 1. File → Import STL/OBJ (or create test shape)
# 2. Set voxel resolution
# 3. Click "Voxelize Mesh"
# 4. Set simulation parameters
# 5. Click "Run Simulation"


# ============================================================================
# ADVANCED EXAMPLES
# ============================================================================

# EXAMPLE 1: Parameter Study
# ────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("EXAMPLE 1: Parameter Study (drag vs Reynolds)")
print("="*60)

mesh = create_simple_cylinder(radius=1.0, height=2.0)
voxelizer = Voxelizer(resolution=0.15)
voxel_grid = voxelizer.voxelize(mesh)

results = []
for reynolds in [50, 100, 150, 200]:
    sim = LBM3D(voxel_grid, reynolds=reynolds, inlet_velocity=0.1)
    
    # Run 100 steps
    for step in range(100):
        sim.step()
    
    stats = sim.get_statistics()
    results.append({
        'reynolds': reynolds,
        'drag': stats['drag'],
        'max_velocity': stats['max_velocity'],
    })
    
    print(f"Re={reynolds:3d}: Cd={stats['drag']:.4f}, v_max={stats['max_velocity']:.4f}")

print("\nParameter study complete!")


# EXAMPLE 2: Convergence Analysis
# ────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("EXAMPLE 2: Convergence Analysis")
print("="*60)

mesh = create_simple_cylinder(radius=1.0, height=2.0)
voxel_grid = Voxelizer(resolution=0.12).voxelize(mesh)
sim = LBM3D(voxel_grid, reynolds=100, inlet_velocity=0.1)

# Track convergence
drag_history = []
for step in range(500):
    sim.step()
    if step % 50 == 0:
        drag = sim.get_drag()
        drag_history.append(drag)
        print(f"Step {step:4d}: Cd={drag:.6f}")

# Check convergence
if len(drag_history) > 2:
    convergence = abs(drag_history[-1] - drag_history[-2]) / max(drag_history[-2], 1e-10)
    print(f"\nConvergence rate: {convergence*100:.6f}%")
    print(f"Final Cd: {drag_history[-1]:.6f}")


# EXAMPLE 3: Create Multiple Geometries
# ────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("EXAMPLE 3: Test Different Geometries")
print("="*60)

geometries = [
    ("Cylinder", create_simple_cylinder(radius=1.0, height=2.0, resolution=24)),
    ("Sphere", create_simple_sphere(radius=1.0, resolution=12)),
]

for name, mesh in geometries:
    print(f"\n{name}:")
    print(f"  Vertices: {mesh.num_vertices}")
    print(f"  Faces: {mesh.num_faces}")
    
    # Voxelize
    voxel_grid = Voxelizer(resolution=0.15).voxelize(mesh)
    print(f"  Voxels: {voxel_grid.num_voxels:,}")
    print(f"  Solid: {voxel_grid.num_solid} ({100*voxel_grid.num_solid/voxel_grid.num_voxels:.1f}%)")


# ============================================================================
# PERFORMANCE TIPS
# ============================================================================

"""
1. RESOLUTION: Finer grids = more accurate but slower
   - 0.2: Fast (50³ grid), coarse results
   - 0.1: Balanced (100³ grid), good accuracy
   - 0.05: Slow (200³ grid), high accuracy
   
2. SIMULATION STEPS: Run more steps for convergence
   - 100 steps: Quick test
   - 1000 steps: Reasonable accuracy
   - 10000 steps: High accuracy (slow)
   
3. MEMORY: Check available RAM
   - 100³ grid: 256 MB
   - 200³ grid: 2 GB
   - 300³ grid: 6.7 GB
   
4. GPU ACCELERATION: Use CuPy when available (coming in Phase 4.2)
   - Expected: 10-100× speedup
"""


# ============================================================================
# TROUBLESHOOTING
# ============================================================================

"""
Q: "mesh_loader not found"
A: Make sure you're in the fluidSimulations directory:
   cd path/to/fluidSimulations

Q: "Voxelization is slow"
A: Increase resolution (use 0.2 or 0.3 instead of 0.1)
   Or reduce mesh complexity

Q: "Out of memory"
A: Use coarser resolution or reduce grid size

Q: "Strange simulation results"
A: Check mesh validity: mesh.validate()
   Ensure inlet velocity is reasonable (0.05-0.2)

Q: "Simulation diverging"
A: Lower inlet velocity
   Increase resolution (finer grid)
   Check mesh has no holes or intersections
"""


# ============================================================================
# NEXT STEPS
# ============================================================================

"""
1. ✅ Try the examples above
2. 📚 Read PHASE4_3D_LBM_GUIDE.md for details
3. 🚀 Launch GUI: python main_3d.py
4. 📊 Generate results for your CAD files
5. 📈 Compare with experimental data

Coming Next (Phase 4.2-4.3):
- GPU acceleration (10-100× speedup)
- VTK 3D visualization
- Benchmark validation
- Advanced physics (turbulence, heat transfer)
"""

print("\n" + "="*60)
print("Phase 4 Quick Start Complete!")
print("="*60)
print("\nNext: Try 'python main_3d.py' for interactive GUI")
print("Or run this file for standalone examples")
