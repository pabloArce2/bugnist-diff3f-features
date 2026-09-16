import argparse
import colorsys
import math
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans

from visualize_pointcloud_features import load_feature_tensor, load_points


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export several point-cloud/.pt feature pairs with comparable shared-PCA point colors."
    )
    parser.add_argument(
        "--item",
        nargs=3,
        action="append",
        metavar=("NAME", "POINTCLOUD", "FEATURES"),
        required=True,
        help="Comparison item: label, point cloud path, .pt feature tensor path.",
    )
    parser.add_argument("--outdir", default="visualizations/pointcloud_feature_comparison")
    parser.add_argument("--fit-sample-per-item", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clip-percentiles", type=float, nargs=2, default=(1.0, 99.0), metavar=("LOW", "HIGH"))
    parser.add_argument("--no-normalize", action="store_true", help="Skip L2-normalizing feature rows before PCA.")
    parser.add_argument(
        "--kmeans",
        type=int,
        default=None,
        metavar="K",
        help="If set, also fit one shared K-means clustering (K clusters) across all items and export a "
        "discrete cluster-colored PLY per item, alongside the continuous shared-PCA PLY.",
    )
    parser.add_argument(
        "--cluster-on",
        choices=("pca", "features"),
        default="features",
        help="Cluster on the full normalized 2048-D descriptor (default) -- the standard choice, since the "
        "3-D PCA projection is a lossy compression picked for RGB display, not for clustering. 'pca' clusters "
        "that same 3-D projection instead, which is cheaper and matches the visible RGB colors exactly, but "
        "can only ever see the structure PCA's top 3 components kept.",
    )
    parser.add_argument(
        "--cluster-sample-per-item",
        type=int,
        default=None,
        help="Rows per item sampled to fit the shared K-means. Defaults to --fit-sample-per-item.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Also render a flat PNG preview for the shared-PCA colors and, if --kmeans is set, the cluster colors.",
    )
    parser.add_argument("--preview-max-points", type=int, default=50000)
    parser.add_argument("--preview-size", type=int, default=1200)
    parser.add_argument("--elev", type=float, default=24.0)
    parser.add_argument("--azim", type=float, default=38.0)
    return parser.parse_args()


def safe_name(name):
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in name)


def sample_features(features, count, rng):
    if len(features) <= count:
        return features
    indices = torch.from_numpy(rng.choice(len(features), size=count, replace=False)).long()
    return features[indices]


def fit_shared_pca(feature_sets, sample_count, seed, normalize, clip_percentiles):
    rng = np.random.default_rng(seed)
    samples = []
    for features in feature_sets:
        if normalize:
            features = torch.nn.functional.normalize(features, dim=1)
        samples.append(sample_features(features, sample_count, rng))

    fit_features = torch.cat(samples, dim=0)
    mean = fit_features.mean(dim=0, keepdim=True)
    fit_centered = fit_features - mean
    _, _, basis = torch.pca_lowrank(fit_centered, q=3, center=False, niter=4)
    basis = basis[:, :3]
    projected_sample = (fit_centered @ basis).cpu().numpy()
    low, high = np.percentile(projected_sample, clip_percentiles, axis=0)
    high = np.maximum(high, low + 1e-6)
    return mean, basis, low, high


def colors_from_pca(features, mean, basis, low, high, normalize):
    if normalize:
        features = torch.nn.functional.normalize(features, dim=1)
    projected = ((features - mean) @ basis).cpu().numpy()
    colors = np.clip((projected - low) / (high - low), 0.0, 1.0)
    return (colors * 255).astype(np.uint8)


def fit_shared_kmeans(feature_sets, mean, basis, k, sample_count, seed, normalize, cluster_on):
    """Fit one K-means model shared across all items, so cluster ids are comparable between items.

    cluster_on="pca" clusters the same 3-D projection used for the RGB colors (a literal k-way split of
    the shared-PCA space). cluster_on="features" clusters the full normalized descriptor instead, which
    can separate structure that the 3-D PCA compression collapses together.
    """
    rng = np.random.default_rng(seed)
    samples = []
    for features in feature_sets:
        if normalize:
            features = torch.nn.functional.normalize(features, dim=1)
        samples.append(sample_features(features, sample_count, rng))
    fit_features = torch.cat(samples, dim=0)

    if cluster_on == "pca":
        fit_vectors = ((fit_features - mean) @ basis).cpu().numpy()
    else:
        fit_vectors = fit_features.cpu().numpy()

    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
    kmeans.fit(fit_vectors)
    return kmeans


def predict_cluster_labels(features, mean, basis, kmeans, normalize, cluster_on):
    if normalize:
        features = torch.nn.functional.normalize(features, dim=1)
    if cluster_on == "pca":
        vectors = ((features - mean) @ basis).cpu().numpy()
    else:
        vectors = features.cpu().numpy()
    return kmeans.predict(vectors)


def cluster_palette(k):
    colors = []
    for i in range(k):
        hue = i / k
        r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.90)
        colors.append((int(round(r * 255)), int(round(g * 255)), int(round(b * 255))))
    return np.array(colors, dtype=np.uint8)


def cluster_sizes(labels, k):
    counts = np.bincount(labels, minlength=k)
    total = max(int(counts.sum()), 1)
    return [(i, int(counts[i]), 100.0 * counts[i] / total) for i in range(k)]


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
    return np.column_stack((x1, ce * y1 - se * z, se * y1 + ce * z))


def save_color_preview(points, colors, out_path, max_points, size, elev, azim, seed, dot_radius=0):
    rng = np.random.default_rng(seed)
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    if len(points) > max_points:
        indices = rng.choice(len(points), size=max_points, replace=False)
        points = points[indices]
        colors = colors[indices]

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
    colors = colors[valid]

    if dot_radius > 0:
        offsets = [
            (dx, dy)
            for dx in range(-dot_radius, dot_radius + 1)
            for dy in range(-dot_radius, dot_radius + 1)
        ]
        x = np.concatenate([x + dx for dx, dy in offsets])
        y = np.concatenate([y + dy for dx, dy in offsets])
        z = np.tile(z, len(offsets))
        colors = np.tile(colors, (len(offsets), 1))
        valid = (x >= 0) & (x < size) & (y >= 0) & (y < size)
        x = x[valid]
        y = y[valid]
        z = z[valid]
        colors = colors[valid]

    image = np.full((size, size, 3), 248, dtype=np.uint8)
    order = np.argsort(z)
    image[y[order], x[order]] = colors[order]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(out_path)


def save_cluster_legend(path, palette, sizes_per_item, k):
    row_h = 26
    width = 460
    height = row_h * k + 40
    image = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    header = "  vs  ".join(name for name, _ in sizes_per_item)
    draw.text((10, 6), f"cluster id  |  {header}", fill=(20, 20, 20), font=font)

    for i in range(k):
        y = 30 + i * row_h
        color = tuple(int(c) for c in palette[i])
        draw.rectangle([10, y, 32, y + row_h - 6], fill=color, outline=(0, 0, 0))
        parts = [f"{name}: {count} ({pct:.1f}%)" for name, sizes in sizes_per_item for idx, count, pct in [sizes[i]]]
        draw.text((40, y + 4), f"{i}: " + "   ".join(parts), fill=(20, 20, 20), font=font)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def export_colored_pointcloud(name, points, colors, outdir):
    if len(points) != len(colors):
        raise ValueError(
            f"{name}: point cloud has {len(points)} points but {len(colors)} colors were computed. "
            "Use the exact point cloud that created the .pt file."
        )

    point_colors = np.column_stack((colors, np.full(len(colors), 255, dtype=np.uint8)))
    out_path = outdir / f"{safe_name(name)}_shared_pca_features.ply"
    trimesh.PointCloud(points, colors=point_colors).export(out_path)
    return out_path, len(points)


def export_cluster_pointcloud(name, points, labels, palette, outdir, k):
    if len(points) != len(labels):
        raise ValueError(f"{name}: point cloud has {len(points)} points but cluster labels have {len(labels)} rows.")

    colors = palette[labels]
    point_colors = np.column_stack((colors, np.full(len(colors), 255, dtype=np.uint8)))
    out_path = outdir / f"{safe_name(name)}_shared_kmeans_k{k}_clusters.ply"
    trimesh.PointCloud(points, colors=point_colors).export(out_path)
    return out_path, colors


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    normalize = not args.no_normalize

    if args.kmeans is not None and args.kmeans < 2:
        raise ValueError("--kmeans must be at least 2.")

    loaded_items = []
    feature_sets = []
    for name, pointcloud_path, feature_path in args.item:
        features = load_feature_tensor(feature_path)
        loaded_items.append((name, Path(pointcloud_path), Path(feature_path), features))
        feature_sets.append(features)

    mean, basis, low, high = fit_shared_pca(
        feature_sets,
        args.fit_sample_per_item,
        args.seed,
        normalize,
        args.clip_percentiles,
    )

    kmeans = None
    palette = None
    if args.kmeans:
        cluster_sample = args.cluster_sample_per_item or args.fit_sample_per_item
        kmeans = fit_shared_kmeans(
            feature_sets, mean, basis, args.kmeans, cluster_sample, args.seed, normalize, args.cluster_on
        )
        palette = cluster_palette(args.kmeans)

    manifest_lines = [
        "Shared PCA point-cloud feature comparison",
        f"items: {len(loaded_items)}",
        f"fit_sample_per_item: {args.fit_sample_per_item}",
        f"normalize: {normalize}",
        "",
    ]

    sizes_per_item = []
    for name, pointcloud_path, feature_path, features in loaded_items:
        points = load_points(pointcloud_path)

        colors = colors_from_pca(features, mean, basis, low, high, normalize)
        out_path, point_count = export_colored_pointcloud(name, points, colors, outdir)
        manifest_lines.extend(
            [
                f"name: {name}",
                f"pointcloud: {pointcloud_path}",
                f"features: {feature_path}",
                f"colored_ply: {out_path}",
                f"points: {point_count}",
            ]
        )
        print(f"Saved {out_path}")

        if args.preview:
            preview_path = outdir / f"{safe_name(name)}_shared_pca_features.png"
            save_color_preview(
                points, colors, preview_path, args.preview_max_points, args.preview_size, args.elev, args.azim, args.seed
            )
            manifest_lines.append(f"colored_preview: {preview_path}")
            print(f"Saved {preview_path}")

        if kmeans is not None:
            labels = predict_cluster_labels(features, mean, basis, kmeans, normalize, args.cluster_on)
            cluster_path, cluster_colors = export_cluster_pointcloud(name, points, labels, palette, outdir, args.kmeans)
            labels_path = outdir / f"{safe_name(name)}_shared_kmeans_k{args.kmeans}_labels.npy"
            np.save(labels_path, labels)
            sizes = cluster_sizes(labels, args.kmeans)
            sizes_per_item.append((name, sizes))

            manifest_lines.append(f"cluster_ply: {cluster_path}")
            manifest_lines.append(f"cluster_labels: {labels_path}")
            print(f"Saved {cluster_path}")

            if args.preview:
                cluster_preview_path = outdir / f"{safe_name(name)}_shared_kmeans_k{args.kmeans}_clusters.png"
                save_color_preview(
                    points,
                    cluster_colors,
                    cluster_preview_path,
                    args.preview_max_points,
                    args.preview_size,
                    args.elev,
                    args.azim,
                    args.seed,
                    dot_radius=1,
                )
                manifest_lines.append(f"cluster_preview: {cluster_preview_path}")
                print(f"Saved {cluster_preview_path}")

            manifest_lines.append(
                "cluster_sizes: " + ", ".join(f"{i}={count}({pct:.1f}%)" for i, count, pct in sizes)
            )

        manifest_lines.append("")

    if kmeans is not None:
        legend_path = outdir / f"shared_kmeans_k{args.kmeans}_legend.png"
        save_cluster_legend(legend_path, palette, sizes_per_item, args.kmeans)

        manifest_lines.append(f"kmeans_k: {args.kmeans}")
        manifest_lines.append(f"cluster_on: {args.cluster_on}")
        manifest_lines.append(f"cluster_seed: {args.seed}")
        manifest_lines.append(
            "cluster_palette_rgb: "
            + " ".join(f"{i}:{tuple(int(c) for c in palette[i])}" for i in range(args.kmeans))
        )
        manifest_lines.append(f"cluster_legend: {legend_path}")
        manifest_lines.append("")
        print(f"Saved {legend_path}")

    manifest_path = outdir / "manifest.txt"
    manifest_path.write_text("\n".join(manifest_lines), encoding="utf-8")
    print(f"Saved {manifest_path}")


if __name__ == "__main__":
    main()
