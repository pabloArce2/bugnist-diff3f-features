import argparse
from pathlib import Path

import numpy as np
import torch
import trimesh

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


def export_colored_pointcloud(name, pointcloud_path, features, colors, outdir):
    points = load_points(pointcloud_path)
    if len(points) != features.shape[0]:
        raise ValueError(
            f"{name}: point cloud has {len(points)} points but features have {features.shape[0]} rows. "
            "Use the exact point cloud that created the .pt file."
        )

    point_colors = np.column_stack((colors, np.full(len(colors), 255, dtype=np.uint8)))
    out_path = outdir / f"{safe_name(name)}_shared_pca_features.ply"
    trimesh.PointCloud(points, colors=point_colors).export(out_path)
    return out_path, len(points)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    normalize = not args.no_normalize

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

    manifest_lines = [
        "Shared PCA point-cloud feature comparison",
        f"items: {len(loaded_items)}",
        f"fit_sample_per_item: {args.fit_sample_per_item}",
        f"normalize: {normalize}",
        "",
    ]
    for name, pointcloud_path, feature_path, features in loaded_items:
        colors = colors_from_pca(features, mean, basis, low, high, normalize)
        out_path, point_count = export_colored_pointcloud(name, pointcloud_path, features, colors, outdir)
        manifest_lines.extend(
            [
                f"name: {name}",
                f"pointcloud: {pointcloud_path}",
                f"features: {feature_path}",
                f"colored_ply: {out_path}",
                f"points: {point_count}",
                "",
            ]
        )
        print(f"Saved {out_path}")

    manifest_path = outdir / "manifest.txt"
    manifest_path.write_text("\n".join(manifest_lines), encoding="utf-8")
    print(f"Saved {manifest_path}")


if __name__ == "__main__":
    main()
