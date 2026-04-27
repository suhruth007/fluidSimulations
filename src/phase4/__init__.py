"""
Phase 4: 3D Lattice Boltzmann Method with CAD Import

Components:
- lbm_3d.py: D3Q27 3D LBM solver
- mesh_loader.py: STL/OBJ CAD file loading
- voxelizer.py: Mesh to voxel grid conversion
- main_3d.py: Interactive 3D LBM GUI

Workflow: CAD File → Mesh → Voxels → LBM Simulation → Results
"""

__version__ = "0.1.0"

# Import main classes for easier access
try:
    from .lbm_3d import LBM3D
    from .mesh_loader import Mesh, MeshLoader
    from .voxelizer import Voxelizer, VoxelGrid
except ImportError:
    pass
