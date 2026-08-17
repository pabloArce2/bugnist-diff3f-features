import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh


def parse_args():
    parser = argparse.ArgumentParser(description="Save quick PNG previews of point clouds.")
    parser.add_argument("--pointcloud", nargs="+", required=True)
    parser.add_argument("--outdir", default="previews/pointclouds")
    parser.add_argument("--max-points", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_points(path):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path)
    loaded = trimesh.load(path)
    if hasattr(loaded, "vertices"):
        return np.asarray(loaded.vertices)
    raise ValueError(f"Could not load point positions from {path}")


def set_axes_equal(ax, points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = (mins + maxs) / 2
    radius = (maxs - mins).max() / 2
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def preview(path, outdir, max_points, rng):
    points = load_points(path)
    if len(points) > max_points:
        points = points[rng.choice(len(points), size=max_points, replace=False)]

    fig = plt.figure(figsize=(7, 7), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.35, c="#5d6670", alpha=0.8)
    set_axes_equal(ax, points)
    ax.view_init(elev=24, azim=38)
    ax.set_axis_off()
    fig.tight_layout(pad=0)

    out_path = outdir / f"{Path(path).stem}.png"
    fig.savefig(out_path, transparent=False, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for path in args.pointcloud:
        preview(path, outdir, args.max_points, rng)


if __name__ == "__main__":
    main()
