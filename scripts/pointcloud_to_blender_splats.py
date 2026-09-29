"""Turn a coloured point cloud into a mesh of tiny coloured octahedra.

Blender does not show vertex colours of a point cloud without faces; the splat
mesh imports with its colours visible.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bugnist_tools.geometry import load_colored_points


OCTAHEDRON_OFFSETS = np.array(
    [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float32,
)

OCTAHEDRON_FACES = np.array(
    [
        [0, 2, 4],
        [2, 1, 4],
        [1, 3, 4],
        [3, 0, 4],
        [2, 0, 5],
        [1, 2, 5],
        [3, 1, 5],
        [0, 3, 5],
    ],
    dtype=np.int64,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a colored point-cloud PLY into a small colored splat mesh for Blender."
    )
    parser.add_argument("--pointcloud", required=True, help="Input colored .ply point cloud.")
    parser.add_argument("--out", required=True, help="Output colored .ply mesh.")
    parser.add_argument(
        "--radius",
        type=float,
        help="Splat radius in geometry units. Default is bbox diagonal * --radius-scale.",
    )
    parser.add_argument(
        "--radius-scale",
        type=float,
        default=0.003,
        help="Automatic radius as fraction of bbox diagonal when --radius is omitted.",
    )
    parser.add_argument("--max-points", type=int, help="Optional random subset for a lighter Blender file.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def subset(points, colors, max_points, seed):
    if max_points is None or len(points) <= max_points:
        return points, colors
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(points), size=max_points, replace=False)
    return points[indices], colors[indices]


def automatic_radius(points, radius_scale):
    diagonal = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
    return max(float(diagonal) * radius_scale, 1e-6)


def make_splat_mesh(points, colors, radius):
    vertex_chunks = []
    face_chunks = []
    color_chunks = []
    for idx, (point, color) in enumerate(zip(points, colors)):
        start = idx * len(OCTAHEDRON_OFFSETS)
        vertex_chunks.append(point + OCTAHEDRON_OFFSETS * radius)
        face_chunks.append(OCTAHEDRON_FACES + start)
        color_chunks.append(np.repeat(color[None, :], len(OCTAHEDRON_OFFSETS), axis=0))

    vertices = np.vstack(vertex_chunks).astype(np.float32)
    faces = np.vstack(face_chunks).astype(np.int64)
    vertex_colors = np.vstack(color_chunks).astype(np.uint8)
    return trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors, process=False)


def main():
    args = parse_args()
    pointcloud_path = Path(args.pointcloud)
    out_path = Path(args.out)
    points, colors = load_colored_points(pointcloud_path)
    points, colors = subset(points, colors, args.max_points, args.seed)
    radius = args.radius if args.radius is not None else automatic_radius(points, args.radius_scale)

    mesh = make_splat_mesh(points, colors, radius)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(out_path)

    print(f"Loaded points: {len(points)}")
    print(f"Splat radius: {radius:.6f}")
    print(f"Output vertices: {len(mesh.vertices)}")
    print(f"Output faces: {len(mesh.faces)}")
    print(f"Saved Blender-friendly splat mesh: {out_path}")


if __name__ == "__main__":
    main()
