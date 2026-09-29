"""Reading meshes and point clouds in the same vertex order as the descriptors.

A Diff3F `.pt` file has one row per vertex (or point), in the order the geometry
was loaded when the descriptor was computed. OBJ files are therefore read with
the upstream `MeshContainer`, which keeps the order of the `v` records.
"""

from pathlib import Path

import numpy as np
import torch
import trimesh

from dataloaders.mesh_container import MeshContainer


def _as_points(points, path):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected Nx3 coordinates in {path}, got shape {points.shape}.")
    if len(points) == 0:
        raise ValueError(f"{path} contains no vertices or points.")
    return points


def load_vertices(path):
    """Vertex or point positions as float32 [N, 3]; rows match the descriptor rows."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        points = np.load(path)
    elif suffix in (".xyz", ".txt"):
        points = np.loadtxt(path, dtype=np.float32)
    elif suffix == ".obj":
        points = MeshContainer().load_from_file(str(path)).vert
    else:
        loaded = trimesh.load(path, process=False, maintain_order=True)
        if isinstance(loaded, trimesh.Scene):
            parts = [geom for geom in loaded.geometry.values() if hasattr(geom, "vertices")]
            if not parts:
                raise ValueError(f"Could not read vertices or points from {path}.")
            points = np.concatenate([np.asarray(geom.vertices) for geom in parts], axis=0)
        elif hasattr(loaded, "vertices"):
            points = loaded.vertices
        else:
            raise ValueError(f"Could not read vertices or points from {path}.")
    return _as_points(points, path)


def load_point_tensor(path):
    return torch.from_numpy(load_vertices(path))


def load_mesh(path):
    """Vertices and triangle faces of a mesh, keeping the descriptor vertex order."""
    path = Path(path)
    if path.suffix.lower() == ".obj":
        mesh = MeshContainer().load_from_file(str(path))
        return np.asarray(mesh.vert, dtype=np.float64), np.asarray(mesh.face, dtype=np.int64)
    loaded = trimesh.load(path, force="mesh", process=False, maintain_order=True)
    return np.asarray(loaded.vertices), np.asarray(loaded.faces)


def load_geometry(path):
    """(vertices, faces) for a mesh, (points, None) for a point cloud."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".obj":
        vertices, faces = load_mesh(path)
        return vertices, faces if len(faces) else None
    if suffix in (".npy", ".xyz", ".txt"):
        return load_vertices(path), None

    loaded = trimesh.load(path, process=False, maintain_order=True)
    if isinstance(loaded, trimesh.Scene) or len(getattr(loaded, "faces", ())) == 0:
        return load_vertices(path), None
    return np.asarray(loaded.vertices), np.asarray(loaded.faces, dtype=np.int64)


def load_colored_points(path):
    """Points and RGBA colors from a colored PLY. Uncolored input gets light grey."""
    loaded = trimesh.load(path, process=False, maintain_order=True)
    if not hasattr(loaded, "vertices"):
        raise ValueError(f"Could not read point positions from {path}.")
    points = _as_points(loaded.vertices, path)

    colors = getattr(loaded, "colors", None)
    if colors is None or len(colors) != len(points):
        colors = np.full((len(points), 4), 210, dtype=np.uint8)
        colors[:, 3] = 255
    else:
        colors = np.asarray(colors, dtype=np.uint8)
        if colors.shape[1] == 3:
            colors = np.column_stack((colors, np.full(len(colors), 255, dtype=np.uint8)))
    return points, colors


def bbox_diagonal(vertices):
    return max(float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))), 1e-6)


def check_rows(name, geometry_rows, feature_rows):
    if geometry_rows != feature_rows:
        raise ValueError(
            f"{name}: geometry has {geometry_rows} vertices/points but the descriptor has {feature_rows} rows. "
            "Use the exact geometry file the .pt was computed from."
        )
