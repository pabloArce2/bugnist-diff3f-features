"""Scatter plot of the shared-PCA feature space, coloured by shared k-means cluster.

This is the feature-space view of the clustering (points, centroids and one
dashed covariance ellipse per cluster) on the three pairwise planes PC1-PC2,
PC1-PC3 and PC2-PC3. Marker shape identifies the item. Cluster ids and colours
match visualize_feature_comparison.py for the same items, K and seed.
"""

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bugnist_tools.features import (
    CLUSTER_SPACES,
    SharedPCA,
    cluster_centers_pca,
    cluster_colors,
    fit_shared_kmeans,
    load_features,
    sample_rows,
)

PROJECTIONS = [(0, 1, "PC1", "PC2"), (0, 2, "PC1", "PC3"), (1, 2, "PC2", "PC3")]
MARKERS = ["o", "^", "s", "D", "v", "P", "*"]


def parse_args():
    parser = argparse.ArgumentParser(description="Scatter-plot the shared-PCA feature space colored by k-means cluster.")
    parser.add_argument(
        "--item",
        nargs=2,
        action="append",
        metavar=("NAME", "FEATURES"),
        required=True,
        help="Label and .pt descriptor. Repeat for every shape; no geometry is needed.",
    )
    parser.add_argument("--kmeans", type=int, required=True, metavar="K", help="Number of shared clusters.")
    parser.add_argument(
        "--cluster-on",
        choices=CLUSTER_SPACES,
        default="features",
        help="Cluster the full descriptor (default) or its 3-D shared-PCA projection. The plot is always in PCA space.",
    )
    parser.add_argument("--outdir", default="visualizations/pca_kmeans_scatter")
    parser.add_argument("--fit-sample-per-item", type=int, default=8000)
    parser.add_argument("--plot-sample-per-item", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument(
        "--ellipse-std", type=float, default=2.0, help="Size of the dashed cluster ellipse in standard deviations; 0 hides it."
    )
    parser.add_argument("--point-size", type=float, default=10.0)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--title", default=None)
    return parser.parse_args()


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

    names = [name for name, _ in args.item]
    feature_sets = [load_features(path) for _, path in args.item]

    pca = SharedPCA(feature_sets, args.fit_sample_per_item, args.seed, normalize)
    kmeans = fit_shared_kmeans(feature_sets, pca, args.kmeans, args.fit_sample_per_item, args.seed, args.cluster_on)
    palette = cluster_colors(args.kmeans)
    centers_3d = cluster_centers_pca(pca, kmeans, args.cluster_on)

    rng = np.random.default_rng(args.seed)
    plotted = []
    for i, (name, features) in enumerate(zip(names, feature_sets)):
        rows = pca.prepare(sample_rows(features, args.plot_sample_per_item, rng))
        projected = pca.project_prepared(rows)
        labels = kmeans.predict(projected if args.cluster_on == "pca" else rows.cpu().numpy())
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
