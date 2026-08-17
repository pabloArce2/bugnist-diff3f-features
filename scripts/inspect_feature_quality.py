import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import torch
import trimesh


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect basic quality diagnostics for dense Diff3F features.")
    parser.add_argument("--geometry", required=True, help="Mesh or point-cloud geometry matching the feature rows.")
    parser.add_argument("--features", required=True, help=".pt feature tensor to inspect.")
    parser.add_argument("--compare-features", help="Optional second .pt tensor on the same geometry.")
    parser.add_argument("--sample-count", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", help="Optional path to write metrics as JSON.")
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
        raise ValueError(f"Expected [num_points, feature_dim], got {tuple(features.shape)} for {path}.")
    return torch.nan_to_num(features.float())


def load_geometry(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        points = np.load(path).astype(np.float32)
        return points, None
    if suffix in (".xyz", ".txt"):
        points = np.loadtxt(path, dtype=np.float32)
        return points, None

    loaded = trimesh.load(path, process=False, maintain_order=True)
    if isinstance(loaded, trimesh.Scene):
        geometries = [geom for geom in loaded.geometry.values() if hasattr(geom, "vertices")]
        if not geometries:
            raise ValueError(f"Could not read vertices from {path}.")
        points = np.concatenate([np.asarray(geom.vertices) for geom in geometries], axis=0).astype(np.float32)
        return points, None

    if not hasattr(loaded, "vertices"):
        raise ValueError(f"Could not read vertices from {path}.")

    points = np.asarray(loaded.vertices, dtype=np.float32)
    faces = np.asarray(loaded.faces, dtype=np.int64) if hasattr(loaded, "faces") and len(loaded.faces) else None
    return points, faces


def summarize(values, prefix):
    values = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p10": float(np.percentile(values, 10)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
    }


def mesh_edges(faces):
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = np.sort(edges, axis=1)
    return np.unique(edges, axis=0)


def pointcloud_neighbor_pairs(points):
    tree = cKDTree(points)
    _, indices = tree.query(points, k=2, workers=-1)
    pairs = np.column_stack((np.arange(len(points)), indices[:, 1]))
    return np.sort(pairs, axis=1)


def sampled_pair_cosines(features, pairs, sample_count, rng):
    if len(pairs) > sample_count:
        pairs = pairs[rng.choice(len(pairs), size=sample_count, replace=False)]
    left = torch.from_numpy(pairs[:, 0]).long()
    right = torch.from_numpy(pairs[:, 1]).long()
    return (features[left] * features[right]).sum(dim=1).numpy()


def random_pair_cosines(features, sample_count, rng):
    left = torch.from_numpy(rng.integers(0, len(features), size=sample_count)).long()
    right = torch.from_numpy(rng.integers(0, len(features), size=sample_count)).long()
    return (features[left] * features[right]).sum(dim=1).numpy()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    points, faces = load_geometry(args.geometry)
    features = load_features(args.features)

    if len(points) != len(features):
        raise ValueError(
            f"Geometry has {len(points)} points/vertices, but features have {len(features)} rows. "
            "Use the exact geometry that produced the features."
        )

    norms = features.norm(dim=1)
    normalized = torch.nn.functional.normalize(features, dim=1)
    metrics = {
        "geometry": args.geometry,
        "features": args.features,
        "points_or_vertices": int(len(points)),
        "feature_dim": int(features.shape[1]),
        "feature_dtype_after_load": str(features.dtype),
        "all_finite": bool(torch.isfinite(features).all()),
        "zero_norm_rows": int((norms == 0).sum()),
    }
    metrics.update(summarize(norms.numpy(), "feature_norm"))

    if faces is not None:
        pairs = mesh_edges(faces)
        pair_kind = "mesh_edge"
    else:
        pairs = pointcloud_neighbor_pairs(points)
        pair_kind = "nearest_neighbor"

    pair_sample_count = min(args.sample_count, len(pairs))
    local_cos = sampled_pair_cosines(normalized, pairs, args.sample_count, rng)
    random_cos = random_pair_cosines(normalized, pair_sample_count, rng)
    metrics["local_pair_kind"] = pair_kind
    metrics["local_pair_count"] = int(len(pairs))
    metrics["sampled_pair_count"] = int(pair_sample_count)
    metrics.update(summarize(local_cos, "local_cosine"))
    metrics.update(summarize(random_cos, "random_cosine"))
    metrics["local_minus_random_median"] = float(np.median(local_cos) - np.median(random_cos))

    if args.compare_features:
        comparison = load_features(args.compare_features)
        if comparison.shape != features.shape:
            raise ValueError(
                f"Comparison feature shape {tuple(comparison.shape)} does not match {tuple(features.shape)}."
            )
        comparison = torch.nn.functional.normalize(comparison, dim=1)
        same_row_cos = (normalized * comparison).sum(dim=1).numpy()
        metrics["compare_features"] = args.compare_features
        metrics.update(summarize(same_row_cos, "same_geometry_compare_cosine"))

    print(json.dumps(metrics, indent=2))
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
