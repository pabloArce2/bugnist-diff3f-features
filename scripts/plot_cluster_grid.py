"""One image per shape with K panels, each showing a single shared k-means cluster.

The cluster is drawn in its colour on top of the rest of the shape in grey, so
clusters hidden behind others in the combined map can be inspected one by one.
Uses the same shared PCA and k-means as visualize_feature_comparison.py.
"""

import argparse
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bugnist_tools.features import CLUSTER_SPACES, SharedPCA, cluster_palette, fit_shared_kmeans, load_features, predict_clusters
from bugnist_tools.geometry import check_rows, load_geometry
from bugnist_tools.preview import rotate_points, thicken
from bugnist_tools.util import safe_name


def parse_args():
    parser = argparse.ArgumentParser(description="Render one panel per shared k-means cluster for every item.")
    parser.add_argument(
        "--item",
        nargs=3,
        action="append",
        metavar=("NAME", "GEOMETRY", "FEATURES"),
        required=True,
        help="Label, mesh or point cloud, and its .pt descriptor. One grid image is written per item.",
    )
    parser.add_argument("--kmeans", type=int, required=True, metavar="K")
    parser.add_argument(
        "--cluster-on",
        choices=CLUSTER_SPACES,
        default="features",
        help="Cluster the full descriptor (default) or its 3-D shared-PCA projection.",
    )
    parser.add_argument("--outdir", default="visualizations/pca_kmeans_cluster_grid")
    parser.add_argument("--fit-sample-per-item", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--panel-size", type=int, default=420, help="Panel width and height in pixels.")
    parser.add_argument("--elev", type=float, default=24.0)
    parser.add_argument("--azim", type=float, default=38.0)
    parser.add_argument("--grey", type=int, nargs=3, default=(222, 222, 222), metavar=("R", "G", "B"))
    return parser.parse_args()


def render_isolated_panel(vertices, labels, cluster_id, palette, grey, size, elev, azim, dot_radius=1):
    """All points in grey, then the points of one cluster in colour on top."""
    rotated = rotate_points(vertices - vertices.mean(axis=0, keepdims=True), elev, azim)
    mins = rotated.min(axis=0)
    maxs = rotated.max(axis=0)
    span = max(float((maxs[:2] - mins[:2]).max()), 1e-6)
    scale = size * 0.86 / span
    center_xy = (mins[:2] + maxs[:2]) / 2

    def to_pixels(points):
        xy = (points[:, :2] - center_xy) * scale + size / 2
        x = np.rint(xy[:, 0]).astype(np.int32)
        y = np.rint(size - xy[:, 1]).astype(np.int32)
        valid = (x >= 0) & (x < size) & (y >= 0) & (y < size)
        return x[valid], y[valid], points[valid, 2]

    image = np.full((size, size, 3), 248, dtype=np.uint8)
    x, y, z = to_pixels(rotated)
    order = np.argsort(z)
    image[y[order], x[order]] = np.array(grey, dtype=np.uint8)

    x, y, z = to_pixels(rotated[labels == cluster_id])
    if dot_radius > 0 and len(x) > 0:
        x, y, z, _ = thicken(x, y, z, None, size, dot_radius)
    if len(x) > 0:
        order = np.argsort(z)
        image[y[order], x[order]] = np.array(palette[cluster_id], dtype=np.uint8)

    return Image.fromarray(image)


def compose_grid(panels, panel_labels, rows, cols, cell_size, title):
    gap = 8
    header_h = 20
    title_h = 30
    width = cols * (cell_size + gap) + gap
    height = title_h + rows * (cell_size + header_h + gap) + gap
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text((gap, 6), title, fill=(20, 20, 20), font=font)

    for idx, (panel, label) in enumerate(zip(panels, panel_labels)):
        r, c = divmod(idx, cols)
        x = gap + c * (cell_size + gap)
        y = title_h + gap + r * (cell_size + header_h + gap)
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        draw.text((x + max(0, (cell_size - text_w) // 2), y), label, fill=(20, 20, 20), font=font)
        canvas.paste(panel, (x, y + header_h))

    return canvas


def grid_shape(k):
    rows = max(1, int(math.floor(math.sqrt(k))))
    cols = math.ceil(k / rows)
    return rows, cols


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    normalize = not args.no_normalize
    k = args.kmeans
    if k < 2:
        raise ValueError("--kmeans must be at least 2.")

    names, vertex_sets, feature_sets = [], [], []
    for name, geometry_path, feature_path in args.item:
        vertices, _ = load_geometry(geometry_path)
        features = load_features(feature_path)
        check_rows(name, len(vertices), features.shape[0])
        names.append(name)
        vertex_sets.append(vertices)
        feature_sets.append(features)

    pca = SharedPCA(feature_sets, args.fit_sample_per_item, args.seed, normalize)
    kmeans = fit_shared_kmeans(feature_sets, pca, k, args.fit_sample_per_item, args.seed, args.cluster_on)
    palette = cluster_palette(k)
    rows, cols = grid_shape(k)

    manifest_lines = [
        "Per-cluster grid (one panel per cluster, the rest of the shape in grey)",
        f"kmeans_k: {k}",
        f"cluster_on: {args.cluster_on}",
        f"seed: {args.seed}",
        "",
    ]

    for name, vertices, features in zip(names, vertex_sets, feature_sets):
        labels = predict_clusters(features, pca, kmeans, args.cluster_on)
        counts = np.bincount(labels, minlength=k)
        total = max(int(counts.sum()), 1)

        panels = []
        panel_labels = []
        for cluster_id in range(k):
            panels.append(
                render_isolated_panel(
                    vertices, labels, cluster_id, palette, args.grey, args.panel_size, args.elev, args.azim
                )
            )
            pct = 100.0 * counts[cluster_id] / total
            panel_labels.append(f"cluster {cluster_id}  ({counts[cluster_id]}, {pct:.1f}%)")

        grid = compose_grid(
            panels,
            panel_labels,
            rows,
            cols,
            args.panel_size,
            title=f"{name}: shared K={k} clusters, isolated one at a time",
        )
        out_path = outdir / f"{safe_name(name)}_cluster_grid_k{k}.png"
        grid.save(out_path)
        print(f"Saved {out_path}")
        manifest_lines.append(f"name: {name}")
        manifest_lines.append(f"grid_png: {out_path}")
        manifest_lines.append("cluster_sizes: " + ", ".join(f"{i}={int(counts[i])}" for i in range(k)))
        manifest_lines.append("")

    manifest_path = outdir / f"cluster_grid_k{k}_manifest.txt"
    manifest_path.write_text("\n".join(manifest_lines), encoding="utf-8")
    print(f"Saved {manifest_path}")


if __name__ == "__main__":
    main()
