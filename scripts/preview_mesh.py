import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import trimesh


def parse_args():
    parser = argparse.ArgumentParser(description="Save a quick PNG preview of mesh files.")
    parser.add_argument("--mesh", nargs="+", required=True)
    parser.add_argument("--outdir", default="previews")
    return parser.parse_args()


def set_axes_equal(ax, vertices):
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    centers = (mins + maxs) / 2
    radius = (maxs - mins).max() / 2
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def preview(mesh_path, outdir):
    mesh = trimesh.load(mesh_path, force="mesh")
    vertices = mesh.vertices
    faces = mesh.faces

    fig = plt.figure(figsize=(7, 7), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
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
    set_axes_equal(ax, vertices)
    ax.view_init(elev=24, azim=38)
    ax.set_axis_off()
    fig.tight_layout(pad=0)

    out_path = outdir / f"{Path(mesh_path).stem}.png"
    fig.savefig(out_path, transparent=False, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved {out_path}")
    print(f"{mesh_path}: {len(vertices)} vertices, {len(faces)} faces")


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for mesh_path in args.mesh:
        preview(mesh_path, outdir)


if __name__ == "__main__":
    main()
