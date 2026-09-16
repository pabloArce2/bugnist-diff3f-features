import argparse
import colorsys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Ellipse
from sklearn.cluster import KMeans

PROJECTIONS = [(0, 1, "PC1", "PC2"), (0, 2, "PC1", "PC3"), (1, 2, "PC2", "PC3")]
MARKERS = ["o", "^", "s", "D", "v", "P", "*"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scatter-plot the shared-PCA feature space colored by shared K-means cluster id, with "
        "cluster centroids marked. This is the 'feature space' view of the clustering: raw PCA points and "
        "K-means centroids, as opposed to the on-mesh region-map view from visualize_feature_comparison.py."
    )
    parser.add_argument(
        "--item",
        nargs=2,
        action="append",
        metavar=("NAME", "FEATURES"),
        required=True,
        help="Comparison item: label, .pt feature tensor path. No mesh needed; this plots feature space only.",
    )
    parser.add_argument("--kmeans", type=int, required=True, metavar="K", help="Number of shared clusters.")
    parser.add_argument(
        "--cluster-on",
        choices=("pca", "features"),
        default="features",
        help="Same meaning and same default as visualize_feature_comparison.py. Centroids are always shown "
        "projected into the 3-D PCA space for plotting, even when clustering happened in full descriptor "
        "space -- that projection is only how this script draws the result, not what K-means used to find it.",
    )
    parser.add_argument("--outdir", default="visualizations/pca_kmeans_scatter")
    parser.add_argument("--fit-sample-per-item", type=int, default=8000)
    parser.add_argument("--plot-sample-per-item", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument(
        "--ellipse-std", type=float, default=2.0, help="Std devs for the dashed cluster ellipse. 0 disables it."
    )
    parser.add_argument("--point-size", type=float, default=10.0)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--title", default=None)
    return parser.parse_args()


def load_features(path):
    loaded = torch.load(path, map_location="cpu")
    if isinstance(loaded, torch.Tensor):
        features = loaded
    elif isinstance(loaded, dict):
        for key in ("features", "feat", "x"):
            if key in loaded and isinstance(loaded[key], torch.Tensor):
                features = loaded[key]
                break
        else:
            raise ValueError(f"{path} is a dict but has no tensor under features/feat/x.")
    else:
        raise ValueError(f"Expected tensor or dict in {path}, got {type(loaded).__name__}.")

    if features.ndim != 2:
        raise ValueError(f"Expected [num_rows, feature_dim] in {path}, got {tuple(features.shape)}.")
    return torch.nan_to_num(features.float())


def sample_features(features, count, rng):
    if len(features) <= count:
        return features
    indices = torch.from_numpy(rng.choice(len(features), size=count, replace=False)).long()
    return features[indices]


def fit_shared_pca(feature_sets, sample_count, seed, normalize):
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
    return mean, basis[:, :3]


def fit_shared_kmeans(feature_sets, mean, basis, k, sample_count, seed, normalize, cluster_on):
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


def project_pca(features, mean, basis, normalize):
    if normalize:
        features = torch.nn.functional.normalize(features, dim=1)
    return ((features - mean) @ basis).cpu().numpy()


def cluster_palette(k):
    colors = []
    for i in range(k):
        hue = i / k
        colors.append(colorsys.hsv_to_rgb(hue, 0.75, 0.90))
    return colors


def covariance_ellipse(points_2d, n_std):
    if len(points_2d) < 3 or n_std <= 0:
        return None
    cov = np.cov(points_2d, rowvar=False)
    if not np.all(np.isfinite(cov)):
        return None
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 0, None)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(eigvals)
    return points_2d.mean(axis=0), width, height, angle


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    normalize = not args.no_normalize

    if args.kmeans < 2:
        raise ValueError("--kmeans must be at least 2.")

    names = []
    feature_sets = []
    for name, feature_path in args.item:
        feature_sets.append(load_features(feature_path))
        names.append(name)

    mean, basis = fit_shared_pca(feature_sets, args.fit_sample_per_item, args.seed, normalize)
    kmeans = fit_shared_kmeans(
        feature_sets, mean, basis, args.kmeans, args.fit_sample_per_item, args.seed, normalize, args.cluster_on
    )
    palette = cluster_palette(args.kmeans)

    if args.cluster_on == "pca":
        centers_3d = kmeans.cluster_centers_
    else:
        centers_features = torch.from_numpy(kmeans.cluster_centers_).float()
        centers_3d = ((centers_features - mean) @ basis).cpu().numpy()

    rng = np.random.default_rng(args.seed)
    plotted = []
    for i, (name, features) in enumerate(zip(names, feature_sets)):
        plot_features = sample_features(features, args.plot_sample_per_item, rng)
        projected = project_pca(plot_features, mean, basis, normalize)
        if args.cluster_on == "pca":
            vectors = projected
        else:
            feats = torch.nn.functional.normalize(plot_features, dim=1) if normalize else plot_features
            vectors = feats.cpu().numpy()
        labels = kmeans.predict(vectors)
        plotted.append((name, MARKERS[i % len(MARKERS)], projected, labels))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, (i, j, xi, yj) in zip(axes, PROJECTIONS):
        for name, marker, projected, labels in plotted:
            for cluster_id in range(args.kmeans):
                mask = labels == cluster_id
                if not np.any(mask):
                    continue
                ax.scatter(
                    projected[mask, i],
                    projected[mask, j],
                    s=args.point_size,
                    alpha=args.alpha,
                    color=palette[cluster_id],
                    marker=marker,
                    linewidths=0,
                )

        if args.ellipse_std > 0:
            for cluster_id in range(args.kmeans):
                cluster_points = [
                    projected[labels == cluster_id][:, [i, j]]
                    for _, _, projected, labels in plotted
                    if np.any(labels == cluster_id)
                ]
                if not cluster_points:
                    continue
                all_points = np.concatenate(cluster_points, axis=0)
                ellipse = covariance_ellipse(all_points, args.ellipse_std)
                if ellipse is None:
                    continue
                center, width, height, angle = ellipse
                ax.add_patch(
                    Ellipse(
                        center,
                        width,
                        height,
                        angle=angle,
                        fill=False,
                        linestyle="--",
                        edgecolor=palette[cluster_id],
                        linewidth=1.6,
                        zorder=4,
                    )
                )

        ax.scatter(centers_3d[:, i], centers_3d[:, j], marker="x", s=90, linewidths=2.2, color="black", zorder=5)
        ax.set_xlabel(xi)
        ax.set_ylabel(yj)
        ax.set_title(f"{xi} vs {yj}")
        ax.grid(alpha=0.25)

    cluster_handles = [
        plt.Line2D(
            [0], [0], marker="o", linestyle="", markerfacecolor=palette[c], markeredgewidth=0, markersize=8,
            label=f"cluster {c}",
        )
        for c in range(args.kmeans)
    ]
    item_handles = [
        plt.Line2D(
            [0], [0], marker=marker, linestyle="", markerfacecolor="grey", markeredgewidth=0, markersize=8,
            label=name,
        )
        for name, marker, _, _ in plotted
    ]
    centroid_handle = [plt.Line2D([0], [0], marker="x", linestyle="", color="black", markersize=9, label="centroid")]
    legend = fig.legend(
        handles=cluster_handles + item_handles + centroid_handle,
        loc="lower center",
        ncol=min(len(cluster_handles) + len(item_handles) + 1, 8),
        frameon=False,
        bbox_to_anchor=(0.5, -0.06),
    )

    title = args.title or f"Shared PCA + K={args.kmeans} K-Means (cluster_on={args.cluster_on})"
    fig.suptitle(title, y=1.03, fontsize=13)
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    out_path = outdir / f"pca_kmeans_scatter_k{args.kmeans}.png"
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight", bbox_extra_artists=(legend,))
    plt.close(fig)
    print(f"Saved {out_path}")

    manifest_path = outdir / f"pca_kmeans_scatter_k{args.kmeans}_manifest.txt"
    lines = [
        "Shared PCA + K-Means feature-space scatter",
        f"kmeans_k: {args.kmeans}",
        f"cluster_on: {args.cluster_on}",
        f"seed: {args.seed}",
        f"fit_sample_per_item: {args.fit_sample_per_item}",
        f"plot_sample_per_item: {args.plot_sample_per_item}",
        "",
    ]
    for name, feature_path in args.item:
        lines.append(f"name: {name}")
        lines.append(f"features: {feature_path}")
    lines.append("")
    lines.append(f"scatter_png: {out_path}")
    manifest_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {manifest_path}")


if __name__ == "__main__":
    main()
