"""Segment one insect in a BugNIST CT volume and sample a point cloud from it.

Uses the same crop/threshold/clean-up steps as bugnist_tif_to_mesh.py. Points
are written in scan voxel units, ordered (X, Y, Z).

Sampling methods:
  mesh-surface    marching-cubes surface, then uniform samples on the triangles
  surface-voxels  jittered centres of the voxels on the mask boundary
  volume-voxels   jittered centres of all mask voxels (includes the interior)
"""

import argparse
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage as ndi
from skimage import measure
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bugnist_tools.ct import add_segmentation_args, crop_from_args, load_volume, segment_from_args


def parse_args():
    parser = argparse.ArgumentParser(description="Convert a BugNIST TIFF volume to a point cloud.")
    parser.add_argument("--tif", required=True, help="Input .tif/.tiff volume.")
    parser.add_argument("--out", required=True, help="Output .ply/.xyz point cloud path.")
    parser.add_argument("--npy", help="Also save the raw Nx3 points to this .npy path.")
    add_segmentation_args(parser)
    parser.add_argument("--num-points", type=int, default=20000, help="Number of points to sample.")
    parser.add_argument(
        "--method",
        choices=("surface-voxels", "mesh-surface", "volume-voxels"),
        default="mesh-surface",
        help="How points are drawn from the segmented volume.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--center", action="store_true", help="Center points around the origin.")
    return parser.parse_args()


def sample_rows(points, count, rng):
    if len(points) == 0:
        raise ValueError("No points to sample.")
    replace = len(points) < count
    return points[rng.choice(len(points), size=count, replace=replace)]


def points_from_surface_voxels(mask, count, rng):
    surface = mask ^ ndi.binary_erosion(mask)
    points_zyx = np.argwhere(surface).astype(np.float32)
    points_zyx += rng.random(points_zyx.shape, dtype=np.float32) - 0.5
    return sample_rows(points_zyx, count, rng)


def points_from_volume_voxels(mask, count, rng):
    points_zyx = np.argwhere(mask).astype(np.float32)
    points_zyx += rng.random(points_zyx.shape, dtype=np.float32) - 0.5
    return sample_rows(points_zyx, count, rng)


def points_from_mesh_surface(mask, count, seed):
    verts_zyx, faces, _, _ = measure.marching_cubes(mask.astype(np.float32), level=0.5)
    mesh = trimesh.Trimesh(vertices=verts_zyx, faces=faces, process=True)
    points_zyx, _ = trimesh.sample.sample_surface(mesh, count, seed=seed)
    return points_zyx.astype(np.float32)


def write_xyz(path, points_xyz):
    with open(path, "w", encoding="utf-8") as file:
        for x, y, z in points_xyz:
            file.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    volume, offset = crop_from_args(load_volume(args.tif), args, offset_dtype=np.float32)
    mask, threshold, threshold_label = segment_from_args(volume, args)

    if args.method == "surface-voxels":
        points_zyx = points_from_surface_voxels(mask, args.num_points, rng)
    elif args.method == "volume-voxels":
        points_zyx = points_from_volume_voxels(mask, args.num_points, rng)
    else:
        points_zyx = points_from_mesh_surface(mask, args.num_points, args.seed)

    points_zyx = (points_zyx + offset) * args.downsample
    points_xyz = points_zyx[:, [2, 1, 0]]
    if args.center:
        points_xyz = points_xyz - points_xyz.mean(axis=0, keepdims=True)

    if out_path.suffix.lower() == ".xyz":
        write_xyz(out_path, points_xyz)
    else:
        trimesh.PointCloud(points_xyz).export(out_path)
    print(f"Saved {out_path}")

    if args.npy:
        npy_path = Path(args.npy)
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy_path, points_xyz)
        print(f"Saved {npy_path}")

    print(f"Volume shape after crop/downsample: {volume.shape}")
    print(f"Volume offset before downsample: {tuple(int(v) for v in (offset * args.downsample))}")
    print(f"Segmentation threshold: {threshold:.3f} ({threshold_label})")
    print(f"Segmented voxels: {int(mask.sum())}")
    print(f"Point count: {len(points_xyz)}")


if __name__ == "__main__":
    main()
