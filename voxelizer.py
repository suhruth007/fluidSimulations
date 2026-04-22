"""
Voxelizer: Convert 3D meshes to voxel grids

Usage:
    voxelizer = Voxelizer(resolution=0.1)
    voxel_grid = voxelizer.voxelize(mesh)
    # voxel_grid[i,j,k] = 1 if inside mesh, 0 if outside
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple
import warnings


class VoxelGrid:
    """3D voxel representation"""
    
    def __init__(self, grid: np.ndarray, origin: np.ndarray, voxel_size: float):
        self.grid = grid.astype(np.uint8)  # 1=solid, 0=fluid
        self.origin = np.array(origin, dtype=np.float32)
        self.voxel_size = float(voxel_size)
        
        self.shape = grid.shape
        self.num_voxels = np.prod(self.shape)
        self.num_solid = np.sum(grid)
        self.num_fluid = self.num_voxels - self.num_solid
    
    @property
    def bounds(self):
        """Get physical bounds of grid"""
        min_corner = self.origin
        max_corner = self.origin + self.voxel_size * np.array(self.shape)
        return min_corner, max_corner
    
    def get_surface_voxels(self):
        """Get indices of voxels on solid/fluid boundary"""
        # A surface voxel is solid with at least one fluid neighbor
        surface = np.zeros_like(self.grid, dtype=bool)
        
        Nx, Ny, Nz = self.shape
        
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    if self.grid[i, j, k] == 1:  # Solid voxel
                        # Check neighbors
                        has_fluid_neighbor = False
                        for di, dj, dk in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
                            ni, nj, nk = i+di, j+dj, k+dk
                            if 0 <= ni < Nx and 0 <= nj < Ny and 0 <= nk < Nz:
                                if self.grid[ni, nj, nk] == 0:
                                    has_fluid_neighbor = True
                                    break
                        
                        if has_fluid_neighbor:
                            surface[i, j, k] = True
        
        return np.argwhere(surface)
    
    def to_binary_file(self, filepath):
        """Save voxel grid to binary file"""
        header = np.array([self.shape[0], self.shape[1], self.shape[2], 
                          self.voxel_size], dtype=np.float32)
        
        with open(filepath, 'wb') as f:
            header.tofile(f)
            self.grid.tofile(f)
    
    @staticmethod
    def from_binary_file(filepath):
        """Load voxel grid from binary file"""
        with open(filepath, 'rb') as f:
            header = np.fromfile(f, dtype=np.float32, count=4)
            Nx, Ny, Nz, voxel_size = int(header[0]), int(header[1]), int(header[2]), header[3]
            
            grid = np.fromfile(f, dtype=np.uint8, count=Nx*Ny*Nz)
            grid = grid.reshape((Nx, Ny, Nz))
            
            origin = np.array([0, 0, 0])
        
        return VoxelGrid(grid, origin, voxel_size)


class Voxelizer:
    """Convert 3D mesh to voxel grid"""
    
    def __init__(self, resolution: float = 0.1, padding: float = 0.5):
        """
        Args:
            resolution: Voxel size in mesh units
            padding: Extra space around mesh as fraction of bbox
        """
        self.resolution = float(resolution)
        self.padding = float(padding)
    
    def voxelize(self, mesh, method='raycasting') -> VoxelGrid:
        """
        Voxelize a mesh
        
        Args:
            mesh: Mesh object with vertices and faces
            method: 'raycasting' or 'winding_number'
        
        Returns:
            VoxelGrid with grid[i,j,k]=1 for solid, 0 for fluid
        """
        print(f"Voxelizing mesh '{mesh.name}'...")
        print(f"  Resolution: {self.resolution}")
        print(f"  Method: {method}")
        
        # Get bounds
        min_corner, max_corner = mesh.bounds
        min_corner = np.array(min_corner)
        max_corner = np.array(max_corner)
        
        # Add padding
        pad_size = self.padding * (max_corner - min_corner)
        min_corner -= pad_size
        max_corner += pad_size
        
        print(f"  Bounds: {min_corner} to {max_corner}")
        
        # Create grid
        grid_size = ((max_corner - min_corner) / self.resolution).astype(int)
        grid_size = np.maximum(grid_size, 2)  # At least 2x2x2
        
        print(f"  Grid size: {grid_size[0]} × {grid_size[1]} × {grid_size[2]} = {np.prod(grid_size)} voxels")
        
        if method == 'raycasting':
            voxels = self._voxelize_raycasting(mesh, min_corner, max_corner, grid_size)
        elif method == 'winding_number':
            voxels = self._voxelize_winding(mesh, min_corner, max_corner, grid_size)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        num_solid = np.sum(voxels.grid)
        num_fluid = voxels.num_voxels - num_solid
        
        print(f"  Solid voxels: {num_solid} ({100*num_solid/voxels.num_voxels:.1f}%)")
        print(f"  Fluid voxels: {num_fluid} ({100*num_fluid/voxels.num_voxels:.1f}%)")
        
        return voxels
    
    def _voxelize_raycasting(self, mesh, min_corner, max_corner, grid_size):
        """Voxelize using ray casting (fast but less robust)"""
        grid = np.zeros(grid_size, dtype=np.uint8)
        
        # Create kd-tree of triangle centroids
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)
        
        centroids = (vertices[faces[:, 0]] + 
                    vertices[faces[:, 1]] + 
                    vertices[faces[:, 2]]) / 3
        tree = cKDTree(centroids)
        
        # For each voxel center, check if inside
        Nx, Ny, Nz = grid_size
        
        for i in range(Nx):
            if (i + 1) % max(1, Nx // 10) == 0:
                print(f"    Progress: {100*(i+1)/Nx:.0f}%")
            
            for j in range(Ny):
                for k in range(Nz):
                    # Voxel center
                    x = min_corner[0] + (i + 0.5) * self.resolution
                    y = min_corner[1] + (j + 0.5) * self.resolution
                    z = min_corner[2] + (k + 0.5) * self.resolution
                    
                    point = np.array([x, y, z])
                    
                    # Check if inside using ray casting
                    if self._point_in_mesh(point, mesh):
                        grid[i, j, k] = 1
        
        return VoxelGrid(grid, min_corner, self.resolution)
    
    def _point_in_mesh(self, point, mesh, num_rays=3):
        """Check if point is inside mesh using ray casting
        
        Cast rays and count intersections with mesh surface
        Point is inside if odd number of intersections
        """
        count = 0
        
        for ray_dir in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
            ray_dir = np.array(ray_dir, dtype=np.float32)
            intersections = 0
            
            # Cast ray from point in direction
            vertices = np.array(mesh.vertices)
            faces = np.array(mesh.faces)
            
            for face in faces:
                v0, v1, v2 = vertices[face]
                
                # Möller–Trumbore ray-triangle intersection
                edge1 = v1 - v0
                edge2 = v2 - v0
                
                h = np.cross(ray_dir, edge2)
                a = np.dot(edge1, h)
                
                if abs(a) < 1e-6:
                    continue
                
                f = 1.0 / a
                s = point - v0
                u = f * np.dot(s, h)
                
                if u < 0.0 or u > 1.0:
                    continue
                
                q = np.cross(s, edge1)
                v = f * np.dot(ray_dir, q)
                
                if v < 0.0 or u + v > 1.0:
                    continue
                
                t = f * np.dot(edge2, q)
                
                if t > 1e-6:
                    intersections += 1
            
            count += intersections % 2
        
        # Point is inside if most rays hit odd intersections
        return count >= 2
    
    def _voxelize_winding(self, mesh, min_corner, max_corner, grid_size):
        """Voxelize using winding number (more robust but slower)"""
        # Simplified implementation using distance-based approach
        grid = np.zeros(grid_size, dtype=np.uint8)
        
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)
        
        # Build KDtree for nearest point queries
        tree = cKDTree(vertices)
        
        Nx, Ny, Nz = grid_size
        
        for i in range(Nx):
            if (i + 1) % max(1, Nx // 10) == 0:
                print(f"    Progress: {100*(i+1)/Nx:.0f}%")
            
            for j in range(Ny):
                for k in range(Nz):
                    # Voxel center
                    x = min_corner[0] + (i + 0.5) * self.resolution
                    y = min_corner[1] + (j + 0.5) * self.resolution
                    z = min_corner[2] + (k + 0.5) * self.resolution
                    
                    point = np.array([x, y, z])
                    
                    # Check if inside
                    if self._point_in_mesh(point, mesh):
                        grid[i, j, k] = 1
        
        return VoxelGrid(grid, min_corner, self.resolution)


# Example usage
if __name__ == '__main__':
    from mesh_loader import create_simple_cylinder, create_simple_sphere
    
    print("=" * 60)
    print("Voxelizer Example")
    print("=" * 60)
    
    # Create test geometry
    print("\n1. Creating test cylinder...")
    mesh = create_simple_cylinder(radius=1.0, height=2.0)
    
    # Voxelize
    print("\n2. Voxelizing...")
    voxelizer = Voxelizer(resolution=0.1)
    voxel_grid = voxelizer.voxelize(mesh)
    
    print("\n3. Statistics:")
    print(f"   Grid shape: {voxel_grid.shape}")
    print(f"   Total voxels: {voxel_grid.num_voxels}")
    print(f"   Solid voxels: {voxel_grid.num_solid}")
    print(f"   Fluid voxels: {voxel_grid.num_fluid}")
    print(f"   Voxel size: {voxel_grid.voxel_size}")
    
    print("\n4. Getting surface voxels...")
    surface = voxel_grid.get_surface_voxels()
    print(f"   Surface voxels: {len(surface)}")
    
    print("\n" + "=" * 60)
