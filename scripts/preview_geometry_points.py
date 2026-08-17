import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image
import trimesh


def parse_args():
    parser = argparse.ArgumentParser(description="Render a quick point-based PNG preview of a mesh or point cloud.")
    parser.add_argument("--input", required=True, help="Input .obj/.ply/.npy geometry file.")
    parser.add_argument("--out", required=True, help="Output .png path.")
    parser.add_argument("--kind", choices=("mesh", "pointcloud"), default="mesh")
    parser.add_argument("--max-points", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--elev", type=float, default=24.0)
    parser.add_argument("--azim", type=float, default=38.0)
    return parser.parse_args()


def rotate_points(points, elev_deg, azim_deg):
    elev = math.radians(elev_deg)
    azim = math.radians(azim_deg)
    ca = math.cos(azim)
    sa = math.sin(azim)
    ce = math.cos(elev)
    se = math.sin(elev)

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    x1 = ca * x - sa * y
    y1 = sa * x + ca * y
    z1 = z
    return np.column_stack((x1, ce * y1 - se * z1, se * y1 + ce * z1))


def load_points(path, kind):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path)

    loaded = trimesh.load(path, force="mesh" if kind == "mesh" else None)
    if hasattr(loaded, "vertices"):
        return np.asarray(loaded.vertices)
    raise ValueError(f"Could not read points from {path}")


def subset_points(points, max_points, seed):
    points = np.asarray(points, dtype=np.float32)
    if len(points) > max_points:
        rng = np.random.default_rng(seed)
        points = points[rng.choice(len(points), size=max_points, replace=False)]
    return points


def render_points(points, out_path, size, elev, azim):
    if len(points) == 0:
        raise ValueError("Cannot preview an empty point set.")

    points = points - points.mean(axis=0, keepdims=True)
    points = rotate_points(points, elev, azim)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    span = max(float((maxs[:2] - mins[:2]).max()), 1e-6)
    scale = size * 0.82 / span

    xy = (points[:, :2] - (mins[:2] + maxs[:2]) / 2) * scale + size / 2
    x = np.rint(xy[:, 0]).astype(np.int32)
    y = np.rint(size - xy[:, 1]).astype(np.int32)
    z = points[:, 2]

    valid = (x >= 0) & (x < size) & (y >= 0) & (y < size)
    x = x[valid]
    y = y[valid]
    z = z[valid]
    z_norm = (z - z.min()) / max(float(z.max() - z.min()), 1e-6)
    order = np.argsort(z_norm)
    shade = (70 + 150 * z_norm).astype(np.uint8)

    image = np.full((size, size, 3), 248, dtype=np.uint8)
    image[y[order], x[order], 0] = shade[order]
    image[y[order], x[order], 1] = np.minimum(shade[order].astype(np.int16) + 4, 255).astype(np.uint8)
    image[y[order], x[order], 2] = np.minimum(shade[order].astype(np.int16) + 10, 255).astype(np.uint8)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(out_path)


def main():
    args = parse_args()
    points = load_points(args.input, args.kind)
    points = subset_points(points, args.max_points, args.seed)
    render_points(points, args.out, args.size, args.elev, args.azim)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
