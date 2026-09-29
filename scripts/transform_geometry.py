"""Rotate, scale or translate a mesh or point cloud and save a copy.

A transformed shape has new vertex positions, so compute a new descriptor for it.
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh


def parse_args():
    parser = argparse.ArgumentParser(description="Apply simple transforms to a mesh or point cloud.")
    parser.add_argument("--input", required=True, help="Input .obj/.ply/.npy/.xyz geometry.")
    parser.add_argument("--out", required=True, help="Output transformed geometry path.")
    parser.add_argument(
        "--kind",
        choices=("auto", "mesh", "pointcloud"),
        default="auto",
        help="Geometry type. auto treats .npy/.xyz as point clouds and meshes with faces as meshes.",
    )
    parser.add_argument(
        "--rotate",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("X_DEG", "Y_DEG", "Z_DEG"),
        help="Euler rotation in degrees around the X, Y, and Z axes.",
    )
    parser.add_argument(
        "--order",
        choices=("xyz", "xzy", "yxz", "yzx", "zxy", "zyx"),
        default="xyz",
        help="Order used to apply the Euler rotations.",
    )
    parser.add_argument(
        "--pivot",
        choices=("center", "origin"),
        default="center",
        help="Rotate around the geometry center or around coordinate origin.",
    )
    parser.add_argument("--scale", type=float, default=1.0, help="Uniform scale applied after rotation.")
    parser.add_argument(
        "--translate",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("DX", "DY", "DZ"),
        help="Translation applied after rotation and scale.",
    )
    return parser.parse_args()


def rotation_matrix(axis, degrees):
    radians = np.deg2rad(degrees)
    c = np.cos(radians)
    s = np.sin(radians)
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)
    if axis == "y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def compose_rotation(angles, order):
    angle_by_axis = dict(zip(("x", "y", "z"), angles))
    matrix = np.eye(3, dtype=np.float64)
    for axis in order:
        matrix = rotation_matrix(axis, angle_by_axis[axis]) @ matrix
    return matrix


def transform_points(points, rotation, pivot, scale, translate):
    points = np.asarray(points, dtype=np.float64)
    if pivot == "center":
        center = (points.min(axis=0) + points.max(axis=0)) / 2.0
    else:
        center = np.zeros(3, dtype=np.float64)

    transformed = (points - center) @ rotation.T
    transformed = transformed * scale + center + np.asarray(translate, dtype=np.float64)
    return transformed, center


def load_geometry(path, kind):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        points = np.load(path)
        return "pointcloud", points, None
    if suffix in (".xyz", ".txt"):
        points = np.loadtxt(path, dtype=np.float64)
        return "pointcloud", points, None

    force = "mesh" if kind == "mesh" else None
    loaded = trimesh.load(path, process=False, maintain_order=True, force=force)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))

    has_faces = hasattr(loaded, "faces") and len(loaded.faces) > 0
    resolved_kind = "mesh" if has_faces else "pointcloud"
    if kind != "auto" and resolved_kind != kind:
        raise ValueError(f"Expected {kind}, but {path} looks like {resolved_kind}.")

    points = np.asarray(loaded.vertices, dtype=np.float64)
    return resolved_kind, points, loaded


def save_geometry(path, kind, points, loaded):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    if kind == "pointcloud":
        if suffix == ".npy":
            np.save(path, points.astype(np.float32))
        elif suffix in (".xyz", ".txt"):
            np.savetxt(path, points, fmt="%.6f")
        else:
            colors = getattr(loaded, "colors", None) if loaded is not None else None
            trimesh.PointCloud(points, colors=colors).export(path)
        return

    mesh = loaded.copy()
    mesh.vertices = points
    mesh.export(path)


def main():
    args = parse_args()
    input_path = Path(args.input)
    out_path = Path(args.out)
    kind, points, loaded = load_geometry(input_path, args.kind)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected Nx3 coordinates, got shape {points.shape}.")
    if len(points) == 0:
        raise ValueError(f"{input_path} contains no vertices/points.")

    rotation = compose_rotation(args.rotate, args.order)
    transformed, center = transform_points(points, rotation, args.pivot, args.scale, args.translate)
    save_geometry(out_path, kind, transformed, loaded)

    face_count = len(loaded.faces) if kind == "mesh" and loaded is not None and hasattr(loaded, "faces") else 0
    print(f"Loaded {kind}: {len(points)} points/vertices, {face_count} faces")
    print(f"Rotation degrees xyz: {tuple(float(v) for v in args.rotate)}")
    print(f"Rotation order: {args.order}")
    print(f"Pivot {args.pivot}: {tuple(float(v) for v in center)}")
    print(f"Scale: {args.scale}")
    print(f"Translate xyz: {tuple(float(v) for v in args.translate)}")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
