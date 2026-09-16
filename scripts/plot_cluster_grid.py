import argparse
import colorsys
import math
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans


def parse_args():
    parser = argparse.ArgumentParser(
        description="For each --item, render one grid image with K panels, each panel isolating a single "
        "shared K-means cluster (highlighted in its color) against the rest of the mesh (greyed out). "
        "This is the 'inspect one cluster at a time' view, as opposed to visualize_feature_comparison.py's "
        "single image with every cluster overlapping at once."
    )
    parser.add_argument(
        "--item",
        nargs=3,
        action="append",
        metavar=("NAME", "MESH", "FEATURES"),
        required=True,
        help="Comparison item: label, mesh path, .pt feature tensor path. One grid PNG is produced per item.",
    )
    parser.add_argument("--kmeans", type=int, required=True, metavar="K")
    parser.add_argument(
        "--cluster-on",
        choices=("pca", "features"),
        default="features",
        help="Same meaning and same default as visualize_feature_comparison.py: cluster the full 2048-D "
        "descriptor by default, or the 3-D shared-PCA projection with 'pca'.",
    )
    parser.add_argument("--outdir", default="visualizations/pca_kmeans_cluster_grid")
    parser.add_argument("--fit-sample-per-item", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--panel-size", type=int, default=420, help="Each cluster panel's square size in pixels.")
    parser.add_argument("--elev", type=float, default=24.0)
    parser.add_argument("--azim", type=float, default=38.0)
    parser.add_argument("--grey", type=int, nargs=3, default=(222, 222, 222), metavar=("R", "G", "B"))
    return parser.parse_args()


def safe_name(name):
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in name)


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
        raise ValueError(f"Expected [num_vertices, feature_dim] in {path}, got {tuple(features.shape)}.")
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


def render_isolated_panel(vertices, labels, cluster_id, palette, grey, size, elev, azim, dot_radius=1):
    """Render one panel: every vertex in `grey`, except `cluster_id`'s vertices painted in their palette
    color and drawn last, so they are always visible instead of being hidden behind non-highlighted points."""
    rotated = rotate_points(vertices - vertices.mean(axis=0, keepdims=True), elev, azim)
    mins = rotated.min(axis=0)
    maxs = rotated.max(axis=0)
    span = max(float((maxs[:2] - mins[:2]).max()), 1e-6)
    scale = size * 0.86 / span
    center_xy = (mins[:2] + maxs[:2]) / 2

    def to_pixels(rot_subset):
        xy = (rot_subset[:, :2] - center_xy) * scale + size / 2
        x = np.rint(xy[:, 0]).astype(np.int32)
        y = np.rint(size - xy[:, 1]).astype(np.int32)
        z = rot_subset[:, 2]
        valid = (x >= 0) & (x < size) & (y >= 0) & (y < size)
        return x[valid], y[valid], z[valid]

    image = np.full((size, size, 3), 248, dtype=np.uint8)

    x_all, y_all, z_all = to_pixels(rotated)
    grey_arr = np.array(grey, dtype=np.uint8)
    order = np.argsort(z_all)
    image[y_all[order], x_all[order]] = grey_arr

    mask = labels == cluster_id
    x_sub, y_sub, z_sub = to_pixels(rotated[mask])

    if dot_radius > 0 and len(x_sub) > 0:
        offsets = [
            (dx, dy) for dx in range(-dot_radius, dot_radius + 1) for dy in range(-dot_radius, dot_radius + 1)
        ]
        xs = np.concatenate([x_sub + dx for dx, dy in offsets])
        ys = np.concatenate([y_sub + dy for dx, dy in offsets])
        zs = np.tile(z_sub, len(offsets))
        valid = (xs >= 0) & (xs < size) & (ys >= 0) & (ys < size)
        xs, ys, zs = xs[valid], ys[valid], zs[valid]
    else:
        xs, ys, zs = x_sub, y_sub, z_sub

    if len(xs) > 0:
        color = np.array(palette[cluster_id], dtype=np.uint8)
        order2 = np.argsort(zs)
        image[ys[order2], xs[order2]] = color

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

    names = []
    meshes = []
    feature_sets = []
    for name, mesh_path, feature_path in args.item:
        mesh = trimesh.load(mesh_path, force="mesh", process=False, maintain_order=True)
        features = load_features(feature_path)
        if len(mesh.vertices) != features.shape[0]:
            raise ValueError(
                f"{name}: mesh has {len(mesh.vertices)} vertices but features have {features.shape[0]} rows."
            )
        names.append(name)
        meshes.append(mesh)
        feature_sets.append(features)

    mean, basis = fit_shared_pca(feature_sets, args.fit_sample_per_item, args.seed, normalize)
    kmeans = fit_shared_kmeans(
        feature_sets, mean, basis, k, args.fit_sample_per_item, args.seed, normalize, args.cluster_on
    )
    palette = cluster_palette(k)
    rows, cols = grid_shape(k)

    manifest_lines = [
        "Per-cluster isolation grid (one panel per cluster, rest of mesh greyed out)",
        f"kmeans_k: {k}",
        f"cluster_on: {args.cluster_on}",
        f"seed: {args.seed}",
        "",
    ]

    for name, mesh, features in zip(names, meshes, feature_sets):
        labels = predict_cluster_labels(features, mean, basis, kmeans, normalize, args.cluster_on)
        counts = np.bincount(labels, minlength=k)
        total = max(int(counts.sum()), 1)

        panels = []
        panel_labels = []
        for cluster_id in range(k):
            panels.append(
                render_isolated_panel(
                    mesh.vertices, labels, cluster_id, palette, args.grey, args.panel_size, args.elev, args.azim
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
