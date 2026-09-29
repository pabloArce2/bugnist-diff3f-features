"""Colour several shapes with one shared PCA basis, optionally with shared k-means clusters.

Because the basis (and the k-means model) is fitted on samples from every
--item, equal colours or cluster ids mean the same thing on all shapes.
Each item can be a mesh or a point cloud.

Outputs in --outdir, per item:
  <name>_shared_pca_features.ply            PCA colours
  <name>_shared_kmeans_k<K>_clusters.ply    cluster colours      (--kmeans)
  <name>_shared_kmeans_k<K>_labels.npy      cluster id per row   (--kmeans)
  *.png                                     previews             (--preview)
plus shared_kmeans_k<K>_legend.png and manifest.txt.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bugnist_tools.features import (
    CLUSTER_SPACES,
    SharedPCA,
    cluster_palette,
    cluster_sizes,
    fit_shared_kmeans,
    load_features,
    predict_clusters,
)
from bugnist_tools.geometry import check_rows, load_geometry
from bugnist_tools.preview import save_color_preview
from bugnist_tools.util import safe_name


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export several geometry/.pt pairs with comparable shared-PCA colors (and optional shared k-means)."
    )
    parser.add_argument(
        "--item",
        nargs=3,
        action="append",
        metavar=("NAME", "GEOMETRY", "FEATURES"),
        required=True,
        help="Label, mesh or point cloud, and its .pt descriptor. Repeat for every shape.",
    )
    parser.add_argument("--outdir", default="visualizations/feature_comparison")
    parser.add_argument("--fit-sample-per-item", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clip-percentiles", type=float, nargs=2, default=(1.0, 99.0), metavar=("LOW", "HIGH"))
    parser.add_argument("--no-normalize", action="store_true", help="Do not L2-normalise the rows before PCA.")
    parser.add_argument("--kmeans", type=int, default=None, metavar="K", help="Also fit one shared K-means with K clusters.")
    parser.add_argument(
        "--cluster-on",
        choices=CLUSTER_SPACES,
        default="features",
        help="Cluster the full descriptor (default) or its 3-D shared-PCA projection.",
    )
    parser.add_argument(
        "--cluster-sample-per-item",
        type=int,
        default=None,
        help="Rows per item used to fit K-means. Defaults to --fit-sample-per-item.",
    )
    parser.add_argument("--preview", action="store_true", help="Also save PNG previews.")
    parser.add_argument("--preview-max-points", type=int, default=50000)
    parser.add_argument("--preview-size", type=int, default=1200)
    parser.add_argument("--elev", type=float, default=24.0)
    parser.add_argument("--azim", type=float, default=38.0)
    return parser.parse_args()


def export_colored(path, vertices, faces, colors):
    rgba = np.column_stack((colors, np.full(len(colors), 255, dtype=np.uint8)))
    if faces is None:
        trimesh.PointCloud(vertices, colors=rgba).export(path)
    else:
        trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=rgba, process=False).export(path)


def save_cluster_legend(path, palette, sizes_per_item, k):
    row_h = 26
    image = Image.new("RGB", (460, row_h * k + 40), (250, 250, 250))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    header = "  vs  ".join(name for name, _ in sizes_per_item)
    draw.text((10, 6), f"cluster id  |  {header}", fill=(20, 20, 20), font=font)
    for i in range(k):
        y = 30 + i * row_h
        draw.rectangle([10, y, 32, y + row_h - 6], fill=tuple(int(c) for c in palette[i]), outline=(0, 0, 0))
        parts = [f"{name}: {sizes[i][1]} ({sizes[i][2]:.1f}%)" for name, sizes in sizes_per_item]
        draw.text((40, y + 4), f"{i}: " + "   ".join(parts), fill=(20, 20, 20), font=font)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    normalize = not args.no_normalize
    if args.kmeans is not None and args.kmeans < 2:
        raise ValueError("--kmeans must be at least 2.")

    items = [(name, Path(geometry), Path(features), load_features(features)) for name, geometry, features in args.item]
    feature_sets = [item[3] for item in items]
    pca = SharedPCA(feature_sets, args.fit_sample_per_item, args.seed, normalize, args.clip_percentiles)

    kmeans = None
    if args.kmeans:
        cluster_sample = args.cluster_sample_per_item or args.fit_sample_per_item
        kmeans = fit_shared_kmeans(feature_sets, pca, args.kmeans, cluster_sample, args.seed, args.cluster_on)
        palette = cluster_palette(args.kmeans)

    preview_options = (args.preview_max_points, args.preview_size, args.elev, args.azim, args.seed)
    manifest = [
        "Shared PCA feature comparison",
        f"items: {len(items)}",
        f"fit_sample_per_item: {args.fit_sample_per_item}",
        f"normalize: {normalize}",
        "",
    ]
    sizes_per_item = []
    for name, geometry_path, feature_path, features in items:
        vertices, faces = load_geometry(geometry_path)
        check_rows(name, len(vertices), len(features))
        stem = safe_name(name)

        colors = pca.colors(features)
        out_path = outdir / f"{stem}_shared_pca_features.ply"
        export_colored(out_path, vertices, faces, colors)
        manifest.append(f"name: {name}")
        if faces is None:
            manifest += [f"pointcloud: {geometry_path}", f"features: {feature_path}", f"colored_ply: {out_path}"]
            manifest += [f"points: {len(vertices)}"]
        else:
            manifest += [f"mesh: {geometry_path}", f"features: {feature_path}", f"colored_ply: {out_path}"]
            manifest += [f"vertices: {len(vertices)}", f"faces: {len(faces)}"]
        print(f"Saved {out_path}")

        if args.preview:
            preview_path = outdir / f"{stem}_shared_pca_features.png"
            save_color_preview(vertices, colors, preview_path, *preview_options)
            manifest.append(f"colored_preview: {preview_path}")
            print(f"Saved {preview_path}")

        if kmeans is not None:
            labels = predict_clusters(features, pca, kmeans, args.cluster_on)
            cluster_path = outdir / f"{stem}_shared_kmeans_k{args.kmeans}_clusters.ply"
            export_colored(cluster_path, vertices, faces, palette[labels])
            labels_path = outdir / f"{stem}_shared_kmeans_k{args.kmeans}_labels.npy"
            np.save(labels_path, labels)
            sizes = cluster_sizes(labels, args.kmeans)
            sizes_per_item.append((name, sizes))
            manifest += [f"cluster_ply: {cluster_path}", f"cluster_labels: {labels_path}"]
            print(f"Saved {cluster_path}")

            if args.preview:
                cluster_preview_path = outdir / f"{stem}_shared_kmeans_k{args.kmeans}_clusters.png"
                # Discrete colours get lost as single pixels, so draw each point as a 3x3 dot.
                save_color_preview(vertices, palette[labels], cluster_preview_path, *preview_options, dot_radius=1)
                manifest.append(f"cluster_preview: {cluster_preview_path}")
                print(f"Saved {cluster_preview_path}")
            manifest.append("cluster_sizes: " + ", ".join(f"{i}={count}({pct:.1f}%)" for i, count, pct in sizes))

        manifest.append("")

    if kmeans is not None:
        legend_path = outdir / f"shared_kmeans_k{args.kmeans}_legend.png"
        save_cluster_legend(legend_path, palette, sizes_per_item, args.kmeans)
        manifest.extend(
            [
                f"kmeans_k: {args.kmeans}",
                f"cluster_on: {args.cluster_on}",
                f"cluster_seed: {args.seed}",
                "cluster_palette_rgb: " + " ".join(f"{i}:{tuple(int(c) for c in palette[i])}" for i in range(args.kmeans)),
                f"cluster_legend: {legend_path}",
                "",
            ]
        )
        print(f"Saved {legend_path}")

    manifest_path = outdir / "manifest.txt"
    manifest_path.write_text("\n".join(manifest), encoding="utf-8")
    print(f"Saved {manifest_path}")


if __name__ == "__main__":
    main()
