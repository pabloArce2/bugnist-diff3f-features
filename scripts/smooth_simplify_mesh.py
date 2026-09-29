"""Smooth a marching-cubes mesh and optionally reduce its face count.

Taubin smoothing removes the voxel staircase without shrinking the shape much.
Decimation uses Open3D quadric simplification if it is installed, otherwise
vertex clustering (merge vertices on a grid, sized by binary search to reach
--target-faces). The result has a new vertex order: compute a new descriptor.
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh
from trimesh import smoothing


def parse_args():
    parser = argparse.ArgumentParser(description="Smooth and optionally simplify a triangle mesh.")
    parser.add_argument("--input", required=True, help="Input .obj/.ply mesh.")
    parser.add_argument("--out", required=True, help="Output .obj/.ply mesh.")
    parser.add_argument(
        "--smooth-method",
        choices=("none", "taubin", "laplacian"),
        default="taubin",
        help="Smoothing method. Taubin usually preserves volume better than plain Laplacian.",
    )
    parser.add_argument("--smooth-iterations", type=int, default=10)
    parser.add_argument("--smooth-lambda", type=float, default=0.5)
    parser.add_argument("--smooth-nu", type=float, default=0.5, help="Taubin smoothing nu parameter.")
    parser.add_argument(
        "--target-faces",
        type=int,
        help="Desired face count after decimation/simplification. Omit to smooth without reducing faces.",
    )
    parser.add_argument(
        "--decimate-method",
        choices=("auto", "quadric", "cluster", "none"),
        default="auto",
        help="quadric uses Open3D if installed; cluster is dependency-free but less shape-preserving.",
    )
    parser.add_argument(
        "--cluster-iterations",
        type=int,
        default=14,
        help="Binary-search iterations used by cluster decimation.",
    )
    parser.add_argument(
        "--final-smooth-iterations",
        type=int,
        default=0,
        help="Optional extra Taubin smoothing after decimation.",
    )
    parser.add_argument("--merge-vertices", action="store_true", help="Merge duplicate vertices before processing.")
    return parser.parse_args()


def load_mesh(path):
    loaded = trimesh.load(path, force="mesh", process=False, maintain_order=True)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if not hasattr(loaded, "faces") or len(loaded.faces) == 0:
        raise ValueError(f"{path} does not contain triangle faces.")
    return loaded


def cleanup_mesh(mesh):
    mesh.remove_unreferenced_vertices()
    if hasattr(mesh, "nondegenerate_faces"):
        mesh.update_faces(mesh.nondegenerate_faces())
    else:
        mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()
    return mesh


def apply_smoothing(mesh, method, iterations, lamb, nu):
    if method == "none" or iterations <= 0:
        return mesh
    if method == "taubin":
        result = smoothing.filter_taubin(mesh, lamb=lamb, nu=nu, iterations=iterations)
    else:
        result = smoothing.filter_laplacian(mesh, lamb=lamb, iterations=iterations)
    return mesh if result is None else result


def try_quadric_decimation(mesh, target_faces):
    try:
        simplified = mesh.simplify_quadric_decimation(int(target_faces))
    except Exception as exc:
        raise RuntimeError(
            "Quadric decimation is unavailable in this environment. "
            "Install Open3D, or use --decimate-method cluster."
        ) from exc
    return cleanup_mesh(simplified)


def cluster_mesh(mesh, voxel_size):
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    mins = vertices.min(axis=0)
    keys = np.floor((vertices - mins) / max(float(voxel_size), 1e-12)).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    vertex_count = int(inverse.max()) + 1

    new_vertices = np.zeros((vertex_count, 3), dtype=np.float64)
    np.add.at(new_vertices, inverse, vertices)
    counts = np.bincount(inverse, minlength=vertex_count).astype(np.float64)
    new_vertices = new_vertices / counts[:, None]

    new_faces = inverse[faces]
    nondegenerate = (
        (new_faces[:, 0] != new_faces[:, 1])
        & (new_faces[:, 1] != new_faces[:, 2])
        & (new_faces[:, 2] != new_faces[:, 0])
    )
    new_faces = new_faces[nondegenerate]
    if len(new_faces) == 0:
        raise ValueError("Cluster decimation collapsed all faces. Use a smaller target reduction.")

    sorted_faces = np.sort(new_faces, axis=1)
    _, unique_indices = np.unique(sorted_faces, axis=0, return_index=True)
    new_faces = new_faces[np.sort(unique_indices)]

    clustered = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
    return cleanup_mesh(clustered)


def cluster_decimation(mesh, target_faces, iterations):
    if target_faces >= len(mesh.faces):
        return mesh.copy()

    bounds = mesh.bounds
    diagonal = np.linalg.norm(bounds[1] - bounds[0])
    if diagonal <= 0:
        raise ValueError("Mesh bounding box is degenerate.")

    low = 0.0
    high = diagonal / 64.0
    best = None
    best_error = float("inf")

    for _ in range(12):
        candidate = cluster_mesh(mesh, high)
        if len(candidate.faces) <= target_faces:
            best = candidate
            best_error = abs(len(candidate.faces) - target_faces)
            break
        high *= 1.6

    if best is None:
        best = candidate
        best_error = abs(len(candidate.faces) - target_faces)

    for _ in range(max(int(iterations), 1)):
        mid = (low + high) / 2.0
        candidate = cluster_mesh(mesh, mid)
        error = abs(len(candidate.faces) - target_faces)
        if len(candidate.faces) <= target_faces and error < best_error:
            best = candidate
            best_error = error
        if len(candidate.faces) > target_faces:
            low = mid
        else:
            high = mid

    return best


def decimate_mesh(mesh, target_faces, method, cluster_iterations):
    if target_faces is None or method == "none":
        return mesh
    if target_faces < 4:
        raise ValueError("--target-faces must be at least 4.")
    if target_faces >= len(mesh.faces):
        print(f"Target faces {target_faces} >= current faces {len(mesh.faces)}; skipping decimation.")
        return mesh

    if method in ("auto", "quadric"):
        try:
            return try_quadric_decimation(mesh, target_faces)
        except RuntimeError as exc:
            if method == "quadric":
                raise
            print(str(exc))
            print("Falling back to vertex-cluster decimation.")

    return cluster_decimation(mesh, target_faces, cluster_iterations)


def main():
    args = parse_args()
    input_path = Path(args.input)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mesh = load_mesh(input_path)
    if args.merge_vertices:
        mesh.merge_vertices()
    mesh = cleanup_mesh(mesh)
    original_vertices = len(mesh.vertices)
    original_faces = len(mesh.faces)

    mesh = apply_smoothing(
        mesh,
        method=args.smooth_method,
        iterations=args.smooth_iterations,
        lamb=args.smooth_lambda,
        nu=args.smooth_nu,
    )
    mesh = cleanup_mesh(mesh)

    after_smooth_vertices = len(mesh.vertices)
    after_smooth_faces = len(mesh.faces)

    mesh = decimate_mesh(
        mesh,
        target_faces=args.target_faces,
        method=args.decimate_method,
        cluster_iterations=args.cluster_iterations,
    )
    mesh = cleanup_mesh(mesh)

    if args.final_smooth_iterations > 0:
        mesh = apply_smoothing(
            mesh,
            method="taubin",
            iterations=args.final_smooth_iterations,
            lamb=args.smooth_lambda,
            nu=args.smooth_nu,
        )
        mesh = cleanup_mesh(mesh)

    mesh.export(out_path)

    print(f"Saved {out_path}")
    print(f"Original vertices/faces: {original_vertices} / {original_faces}")
    print(f"After smoothing vertices/faces: {after_smooth_vertices} / {after_smooth_faces}")
    print(f"Final vertices/faces: {len(mesh.vertices)} / {len(mesh.faces)}")


if __name__ == "__main__":
    main()
