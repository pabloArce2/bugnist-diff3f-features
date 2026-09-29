"""Sample points uniformly on the surface of existing meshes."""

import argparse
from pathlib import Path

import numpy as np
import trimesh


def parse_args():
    parser = argparse.ArgumentParser(description="Sample point clouds from mesh surfaces.")
    parser.add_argument("--mesh", nargs="+", required=True)
    parser.add_argument("--outdir", default="pointclouds/from_mesh")
    parser.add_argument("--num-points", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--center", action="store_true")
    parser.add_argument("--npy", action="store_true", help="Also write Nx3 .npy files.")
    return parser.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for mesh_path in args.mesh:
        mesh_path = Path(mesh_path)
        mesh = trimesh.load(mesh_path, force="mesh")
        points, _ = trimesh.sample.sample_surface(mesh, args.num_points, seed=args.seed)
        if args.center:
            points = points - points.mean(axis=0, keepdims=True)

        ply_path = outdir / f"{mesh_path.stem}_{args.num_points // 1000}k.ply"
        trimesh.PointCloud(points).export(ply_path)
        print(f"Saved {ply_path}")

        if args.npy:
            npy_path = ply_path.with_suffix(".npy")
            np.save(npy_path, points)
            print(f"Saved {npy_path}")


if __name__ == "__main__":
    main()
