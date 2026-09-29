"""Save a PNG preview of meshes and point clouds.

The default renderer uses matplotlib: meshes are drawn as a shaded surface,
point clouds as a scatter plot. --renderer points draws the vertices as
depth-shaded dots instead, which is much faster for dense CT meshes.
Each input is written to <outdir>/<stem>.png.
"""

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bugnist_tools.geometry import load_geometry
from bugnist_tools.preview import BACKGROUND, project_to_pixels


def parse_args():
    parser = argparse.ArgumentParser(description="Save PNG previews of meshes and point clouds.")
    parser.add_argument("--input", nargs="+", required=True, help="Mesh or point-cloud files (.obj, .ply, .npy, .xyz).")
    parser.add_argument("--outdir", default="previews/geometry")
    parser.add_argument("--renderer", choices=("matplotlib", "points"), default="matplotlib")
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Subsample to this many points (default 12000 for matplotlib scatter plots, 30000 for --renderer points).",
    )
    parser.add_argument("--size", type=int, default=1000, help="Image size for --renderer points.")
    parser.add_argument("--elev", type=float, default=24.0)
    parser.add_argument("--azim", type=float, default=38.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def subsample(points, max_points, rng):
    if len(points) > max_points:
        points = points[rng.choice(len(points), size=max_points, replace=False)]
    return points


def set_axes_equal(ax, points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = (mins + maxs) / 2
    radius = (maxs - mins).max() / 2
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def matplotlib_preview(vertices, faces, out_path, elev, azim):
    fig = plt.figure(figsize=(7, 7), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    if faces is not None:
        ax.plot_trisurf(
            vertices[:, 0],
            vertices[:, 1],
            faces,
            vertices[:, 2],
            color=(0.72, 0.74, 0.76),
            linewidth=0,
            antialiased=False,
            shade=True,
        )
    else:
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=0.35, c="#5d6670", alpha=0.8)
    set_axes_equal(ax, vertices)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(out_path, transparent=False, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def points_preview(points, out_path, size, elev, azim):
    """Grey dots, darker further away."""
    x, y, z, _ = project_to_pixels(np.asarray(points, dtype=np.float32), size, elev, azim)
    z_norm = (z - z.min()) / max(float(z.max() - z.min()), 1e-6)
    order = np.argsort(z_norm)
    shade = (70 + 150 * z_norm).astype(np.uint8)

    image = np.full((size, size, 3), BACKGROUND, dtype=np.uint8)
    image[y[order], x[order], 0] = shade[order]
    image[y[order], x[order], 1] = np.minimum(shade[order].astype(np.int16) + 4, 255).astype(np.uint8)
    image[y[order], x[order], 2] = np.minimum(shade[order].astype(np.int16) + 10, 255).astype(np.uint8)
    Image.fromarray(image).save(out_path)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    max_points = args.max_points or (30000 if args.renderer == "points" else 12000)
    for path in args.input:
        rng = np.random.default_rng(args.seed)
        vertices, faces = load_geometry(path)
        out_path = outdir / f"{Path(path).stem}.png"
        if args.renderer == "points":
            points_preview(subsample(vertices, max_points, rng), out_path, args.size, args.elev, args.azim)
        else:
            if faces is None:
                vertices = subsample(vertices, max_points, rng)
            matplotlib_preview(vertices, faces, out_path, args.elev, args.azim)

        kind = f"{len(vertices)} vertices, {len(faces)} faces" if faces is not None else f"{len(vertices)} points"
        print(f"Saved {out_path} ({kind})")


if __name__ == "__main__":
    main()
