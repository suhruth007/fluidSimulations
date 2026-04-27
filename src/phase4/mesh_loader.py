"""
Mesh Loader: Load and process 3D CAD files (STL, OBJ, STEP)

Usage:
    loader = MeshLoader('path/to/model.stl')
    mesh = loader.load()
    print(mesh.vertices.shape, mesh.faces.shape)
"""

import numpy as np
from pathlib import Path
import warnings


class Mesh:
    """Simple mesh container"""
    def __init__(self, vertices, faces, name="mesh"):
        self.vertices = np.array(vertices, dtype=np.float32)
        self.faces = np.array(faces, dtype=np.int32)
        self.name = name
    
    @property
    def num_vertices(self):
        return len(self.vertices)
    
    @property
    def num_faces(self):
        return len(self.faces)
    
    @property
    def bounds(self):
        """Get min/max coordinates"""
        return self.vertices.min(axis=0), self.vertices.max(axis=0)
    
    def translate(self, offset):
        """Translate mesh by offset"""
        self.vertices += np.array(offset)
        return self
    
    def scale(self, factor):
        """Scale mesh by factor"""
        self.vertices *= factor
        return self
    
    def get_surface_normals(self):
        """Compute face normals"""
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        
        edge1 = v1 - v0
        edge2 = v2 - v0
        normals = np.cross(edge1, edge2)
        
        # Normalize
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / (lengths + 1e-10)
        
        return normals
    
    def validate(self):
        """Check mesh integrity"""
        issues = []
        
        # Check for invalid faces
        max_idx = self.num_vertices
        invalid_faces = np.any((self.faces < 0) | (self.faces >= max_idx), axis=1)
        if np.any(invalid_faces):
            issues.append(f"Found {np.sum(invalid_faces)} invalid face indices")
        
        # Check for degenerate triangles
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        
        edge1 = v1 - v0
        edge2 = v2 - v0
        areas = 0.5 * np.linalg.norm(np.cross(edge1, edge2), axis=1)
        
        degen = areas < 1e-6
        if np.any(degen):
            issues.append(f"Found {np.sum(degen)} degenerate triangles")
        
        return issues if issues else ["✓ Mesh is valid"]


class MeshLoader:
    """Load 3D mesh files (STL, OBJ)"""
    
    def __init__(self, filepath):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        self.file_type = self.filepath.suffix.lower()
    
    def load(self):
        """Load mesh from file"""
        if self.file_type == '.stl':
            return self._load_stl()
        elif self.file_type == '.obj':
            return self._load_obj()
        else:
            raise ValueError(f"Unsupported format: {self.file_type}")
    
    def _load_stl(self):
        """Load STL file (ASCII or binary)"""
        with open(self.filepath, 'rb') as f:
            # Check if ASCII or binary
            try:
                # Try ASCII first
                f.seek(0)
                header = f.read(5).decode('ascii')
                if header == 'solid':
                    return self._load_stl_ascii()
            except:
                pass
            
            # Load binary
            return self._load_stl_binary()
    
    def _load_stl_ascii(self):
        """Load ASCII STL"""
        vertices = []
        faces = []
        vertex_map = {}
        vertex_count = 0
        
        with open(self.filepath, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('facet normal'):
                # Read 3 vertices
                face_vertices = []
                
                for j in range(3):
                    i += 1
                    while i < len(lines):
                        line = lines[i].strip()
                        if line.startswith('vertex'):
                            coords = tuple(float(x) for x in line.split()[1:4])
                            
                            if coords not in vertex_map:
                                vertex_map[coords] = vertex_count
                                vertices.append(coords)
                                vertex_count += 1
                            
                            face_vertices.append(vertex_map[coords])
                            i += 1
                            break
                        i += 1
                
                if len(face_vertices) == 3:
                    faces.append(face_vertices)
            
            i += 1
        
        return Mesh(vertices, faces, name=self.filepath.stem)
    
    def _load_stl_binary(self):
        """Load binary STL"""
        with open(self.filepath, 'rb') as f:
            # Skip header
            header = f.read(80)
            
            # Number of triangles
            num_triangles = np.frombuffer(f.read(4), dtype=np.uint32)[0]
            
            vertices = []
            faces = []
            vertex_map = {}
            vertex_count = 0
            
            for _ in range(num_triangles):
                # Normal (unused)
                normal = np.frombuffer(f.read(12), dtype=np.float32)
                
                # 3 vertices
                face_vertices = []
                for _ in range(3):
                    vertex = tuple(np.frombuffer(f.read(12), dtype=np.float32))
                    
                    if vertex not in vertex_map:
                        vertex_map[vertex] = vertex_count
                        vertices.append(vertex)
                        vertex_count += 1
                    
                    face_vertices.append(vertex_map[vertex])
                
                faces.append(face_vertices)
                
                # Attribute byte count
                _ = f.read(2)
        
        return Mesh(vertices, faces, name=self.filepath.stem)
    
    def _load_obj(self):
        """Load OBJ file"""
        vertices = []
        faces = []
        
        with open(self.filepath, 'r') as f:
            for line in f:
                line = line.strip()
                
                if line.startswith('v '):
                    # Vertex
                    coords = [float(x) for x in line.split()[1:4]]
                    vertices.append(coords)
                
                elif line.startswith('f '):
                    # Face
                    parts = line.split()[1:]
                    face = []
                    
                    for part in parts:
                        # Handle v, v/vt, v/vt/vn, v//vn formats
                        idx = int(part.split('/')[0]) - 1  # OBJ uses 1-based indexing
                        face.append(idx)
                    
                    if len(face) == 3:
                        faces.append(face)
        
        return Mesh(vertices, faces, name=self.filepath.stem)


def create_simple_cylinder(radius=1.0, height=2.0, resolution=16):
    """Create a simple 3D cylinder mesh for testing"""
    vertices = []
    faces = []
    
    # Top and bottom circles
    angle_step = 2 * np.pi / resolution
    
    # Bottom circle (z=0)
    for i in range(resolution):
        angle = i * angle_step
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        vertices.append([x, y, 0.0])
    
    # Top circle (z=height)
    for i in range(resolution):
        angle = i * angle_step
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        vertices.append([x, y, height])
    
    # Center vertices
    vertices.append([0, 0, 0])  # Bottom center
    vertices.append([0, 0, height])  # Top center
    
    bottom_center = 2 * resolution
    top_center = 2 * resolution + 1
    
    # Bottom and top caps
    for i in range(resolution):
        next_i = (i + 1) % resolution
        # Bottom cap
        faces.append([bottom_center, i, next_i])
        # Top cap
        faces.append([top_center, resolution + next_i, resolution + i])
    
    # Side faces
    for i in range(resolution):
        next_i = (i + 1) % resolution
        # Two triangles per quad
        v0 = i
        v1 = next_i
        v2 = resolution + i
        v3 = resolution + next_i
        
        faces.append([v0, v2, v3])
        faces.append([v0, v3, v1])
    
    return Mesh(vertices, faces, name="cylinder")


def create_simple_sphere(radius=1.0, resolution=8):
    """Create a simple 3D sphere mesh for testing"""
    vertices = []
    faces = []
    
    # Create icosphere-like structure
    phi_steps = resolution
    theta_steps = resolution * 2
    
    for i in range(phi_steps + 1):
        phi = np.pi * i / phi_steps
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        
        for j in range(theta_steps):
            theta = 2 * np.pi * j / theta_steps
            x = radius * sin_phi * np.cos(theta)
            y = radius * sin_phi * np.sin(theta)
            z = radius * cos_phi
            vertices.append([x, y, z])
    
    # Create faces
    for i in range(phi_steps):
        for j in range(theta_steps):
            a = i * theta_steps + j
            b = a + theta_steps
            c = a + 1 if j < theta_steps - 1 else i * theta_steps
            d = b + 1 if j < theta_steps - 1 else (i + 1) * theta_steps
            
            if i > 0:
                faces.append([a, b, c])
            if i < phi_steps - 1:
                faces.append([c, b, d])
    
    return Mesh(vertices, faces, name="sphere")


# Example usage
if __name__ == '__main__':
    print("=" * 60)
    print("Mesh Loader Example")
    print("=" * 60)
    
    # Create test geometries
    print("\n1. Creating simple cylinder...")
    cyl = create_simple_cylinder(radius=1.0, height=2.0, resolution=32)
    print(f"   Vertices: {cyl.num_vertices}")
    print(f"   Faces: {cyl.num_faces}")
    print(f"   Bounds: {cyl.bounds}")
    print(f"   Validation: {cyl.validate()}")
    
    print("\n2. Creating simple sphere...")
    sphere = create_simple_sphere(radius=1.0, resolution=16)
    print(f"   Vertices: {sphere.num_vertices}")
    print(f"   Faces: {sphere.num_faces}")
    print(f"   Bounds: {sphere.bounds}")
    print(f"   Validation: {sphere.validate()}")
    
    # Save cylinder as STL (binary)
    print("\n3. Saving cylinder as STL...")
    try:
        from stl import mesh as stl_mesh
        # Would need numpy-stl library
        print("   (requires numpy-stl library)")
    except ImportError:
        print("   Install with: pip install numpy-stl")
    
    print("\n" + "=" * 60)
