import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import trimesh


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize per-vertex .pt features as colored mesh/preview files.")
    parser.add_argument("--mesh", required=True, help="Mesh whose vertex order matches the feature tensor.")
    parser.add_argument("--features", required=True, help=".pt tensor with shape [num_vertices, feature_dim].")
    parser.add_argument("--out", required=True, help="Output colored .ply path.")
    parser.add_argument("--preview", help="Optional output .png point-render preview.")
    parser.add_argument("--fit-sample", type=int, default=12000, help="Rows used to fit PCA; lower is faster.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clip-percentiles", type=float, nargs=2, default=(1.0, 99.0), metavar=("LOW", "HIGH"))
    parser.add_argument("--no-normalize", action="store_true", help="Skip L2-normalizing feature rows before PCA.")
    parser.add_argument("--invert", action="store_true", help="Invert RGB colors.")
    parser.add_argument("--preview-max-points", type=int, default=50000)
    parser.add_argument("--preview-size", type=int, default=1200)
    parser.add_argument("--elev", type=float, default=24.0)
    parser.add_argument("--azim", type=float, default=38.0)
    return parser.parse_args()


def load_feature_tensor(path):
    loaded = torch.load(path, map_location="cpu")
    if isinstance(loaded, torch.Tensor):
        features = loaded
    elif isinstance(loaded, dict):
        for key in ("features", "feat", "x"):
            if key in loaded and isinstance(loaded[key], torch.Tensor):
                features = loaded[key]
                break
        else:
            raise ValueError("Feature .pt dict did not contain a tensor under features/feat/x.")
    else:
        raise ValueError(f"Expected a tensor or dict in {path}, got {type(loaded).__name__}.")

    if features.ndim != 2:
        raise ValueError(f"Expected feature tensor shape [num_vertices, feature_dim], got {tuple(features.shape)}.")
    return features.float()


def fit_pca_colors(features, fit_sample, seed, clip_percentiles, normalize=True, invert=False):
    features = torch.nan_to_num(features)
    if normalize:
        features = torch.nn.functional.normalize(features, dim=1)

    rng = np.random.default_rng(seed)
    if len(features) > fit_sample:
        fit_indices = torch.from_numpy(rng.choice(len(features), size=fit_sample, replace=False)).long()
        fit_features = features[fit_indices]
    else:
        fit_features = features

    mean = fit_features.mean(dim=0, keepdim=True)
    fit_centered = fit_features - mean
    _, _, basis = torch.pca_lowrank(fit_centered, q=3, center=False, niter=4)
    projected = (features - mean) @ basis[:, :3]
    projected = projected.cpu().numpy()

    low, high = np.percentile(projected, clip_percentiles, axis=0)
    span = np.maximum(high - low, 1e-6)
    colors = np.clip((projected - low) / span, 0.0, 1.0)
    if invert:
        colors = 1.0 - colors
    return (colors * 255).astype(np.uint8)


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


def save_color_preview(vertices, colors, out_path, max_points, size, elev, azim, seed):
    rng = np.random.default_rng(seed)
    vertices = np.asarray(vertices, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    if len(vertices) > max_points:
        indices = rng.choice(len(vertices), size=max_points, replace=False)
        vertices = vertices[indices]
        colors = colors[indices]

    points = vertices - vertices.mean(axis=0, keepdims=True)
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

    image = np.full((size, size, 3), 248, dtype=np.uint8)
    order = np.argsort(z)
    image[y[order], x[order]] = colors[order]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(out_path)


def main():
    args = parse_args()
    mesh_path = Path(args.mesh)
    feature_path = Path(args.features)
    out_path = Path(args.out)

    mesh = trimesh.load(mesh_path, force="mesh", process=False, maintain_order=True)
    features = load_feature_tensor(feature_path)
    if len(mesh.vertices) != features.shape[0]:
        raise ValueError(
            f"Mesh has {len(mesh.vertices)} vertices, but features have {features.shape[0]} rows. "
            "Use the exact mesh used to compute the .pt file."
        )

    colors = fit_pca_colors(
        features,
        fit_sample=args.fit_sample,
        seed=args.seed,
        clip_percentiles=args.clip_percentiles,
        normalize=not args.no_normalize,
        invert=args.invert,
    )
    vertex_colors = np.column_stack((colors, np.full(len(colors), 255, dtype=np.uint8)))
    colored_mesh = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces,
        vertex_colors=vertex_colors,
        process=False,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    colored_mesh.export(out_path)
    print(f"Saved colored mesh: {out_path}")

    if args.preview:
        save_color_preview(
            mesh.vertices,
            colors,
            args.preview,
            args.preview_max_points,
            args.preview_size,
            args.elev,
            args.azim,
            args.seed,
        )
        print(f"Saved preview: {args.preview}")


if __name__ == "__main__":
    main()
