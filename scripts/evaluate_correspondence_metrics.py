import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import trimesh
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate feature-correspondence diagnostics without ground truth.")
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--source-mesh", required=True, help="Source mesh or point cloud geometry.")
    parser.add_argument("--source-features", required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--target-mesh", required=True, help="Target mesh or point cloud geometry.")
    parser.add_argument("--target-features", required=True)
    parser.add_argument("--outdir", default="visualizations/correspondence_metrics")
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--sampling", choices=("farthest", "random"), default="farthest")
    parser.add_argument("--device", default=None)
    parser.add_argument("--query-chunk-size", type=int, default=16)
    parser.add_argument("--target-chunk-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def safe_name(name):
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in name)


def load_geometry_vertices(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        vertices = np.load(path)
    elif suffix in (".xyz", ".txt"):
        vertices = np.loadtxt(path, dtype=np.float32)
    else:
        loaded = trimesh.load(path, process=False, maintain_order=True)
        if isinstance(loaded, trimesh.Scene):
            geometries = [geom for geom in loaded.geometry.values() if hasattr(geom, "vertices")]
            if not geometries:
                raise ValueError(f"Could not load vertices or points from {path}.")
            vertices = np.concatenate([np.asarray(geom.vertices) for geom in geometries], axis=0)
        elif hasattr(loaded, "vertices"):
            vertices = np.asarray(loaded.vertices)
        else:
            raise ValueError(f"Could not load vertices or points from {path}.")

    vertices = np.asarray(vertices, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Expected Nx3 geometry coordinates in {path}, got shape {vertices.shape}.")
    return vertices


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
        raise ValueError(f"Expected feature tensor [num_vertices, feature_dim], got {tuple(features.shape)}.")
    return torch.nan_to_num(features.float())


def farthest_point_indices(vertices, count, seed):
    rng = np.random.default_rng(seed)
    count = min(count, len(vertices))
    center = vertices.mean(axis=0, keepdims=True)
    selected = np.empty(count, dtype=np.int64)
    selected[0] = int(np.argmax(np.sum((vertices - center) ** 2, axis=1)))
    min_dist2 = np.sum((vertices - vertices[selected[0]]) ** 2, axis=1)

    for i in range(1, count):
        if i % 10 == 0:
            top_count = min(64, len(min_dist2))
            candidates = np.argpartition(min_dist2, -top_count)[-top_count:]
            chosen = int(rng.choice(candidates))
        else:
            chosen = int(np.argmax(min_dist2))
        selected[i] = chosen
        dist2 = np.sum((vertices - vertices[chosen]) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, dist2)
    return selected


def sample_indices(vertices, count, sampling, seed):
    if sampling == "random":
        rng = np.random.default_rng(seed)
        return rng.choice(len(vertices), size=min(count, len(vertices)), replace=False)
    return farthest_point_indices(vertices, count, seed)


def normalize_features(features):
    return torch.nn.functional.normalize(features, dim=1)


def nearest_feature_indices(query_features, target_features, device, query_chunk_size, target_chunk_size):
    query_features = normalize_features(query_features)
    target_features = normalize_features(target_features)
    best_indices = torch.empty(len(query_features), dtype=torch.long)
    best_scores = torch.full((len(query_features),), -float("inf"), dtype=torch.float32)
    second_scores = torch.full((len(query_features),), -float("inf"), dtype=torch.float32)

    for q_start in tqdm(range(0, len(query_features), query_chunk_size), desc="NN query chunks"):
        q_stop = min(q_start + query_chunk_size, len(query_features))
        query = query_features[q_start:q_stop].to(device)
        chunk_best_scores = torch.full((q_stop - q_start,), -float("inf"), device=device)
        chunk_second_scores = torch.full((q_stop - q_start,), -float("inf"), device=device)
        chunk_best_indices = torch.zeros((q_stop - q_start,), dtype=torch.long, device=device)

        for t_start in range(0, len(target_features), target_chunk_size):
            t_stop = min(t_start + target_chunk_size, len(target_features))
            target = target_features[t_start:t_stop].to(device)
            similarity = query @ target.T
            values, local_indices = torch.topk(similarity, k=min(2, similarity.shape[1]), dim=1)
            candidate_scores = values[:, 0]
            candidate_indices = local_indices[:, 0] + t_start
            candidate_second = values[:, 1] if values.shape[1] > 1 else torch.full_like(candidate_scores, -float("inf"))

            update = candidate_scores > chunk_best_scores
            old_best = chunk_best_scores.clone()
            chunk_second_scores = torch.where(update, torch.maximum(old_best, candidate_second), chunk_second_scores)
            chunk_second_scores = torch.where(~update, torch.maximum(chunk_second_scores, candidate_scores), chunk_second_scores)
            chunk_best_scores = torch.where(update, candidate_scores, chunk_best_scores)
            chunk_best_indices = torch.where(update, candidate_indices, chunk_best_indices)

        best_indices[q_start:q_stop] = chunk_best_indices.cpu()
        best_scores[q_start:q_stop] = chunk_best_scores.cpu()
        second_scores[q_start:q_stop] = chunk_second_scores.cpu()

    return best_indices.numpy(), best_scores.numpy(), second_scores.numpy()


def bbox_diagonal(vertices):
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    return max(float(np.linalg.norm(maxs - mins)), 1e-6)


def upper_triangular_distances(points):
    diff = points[:, None, :] - points[None, :, :]
    distances = np.linalg.norm(diff, axis=2)
    return distances[np.triu_indices(len(points), k=1)]


def rankdata(values):
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    return pearson(rankdata(x), rankdata(y))


def entropy_normalized(indices):
    _, counts = np.unique(indices, return_counts=True)
    probs = counts / counts.sum()
    entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
    return entropy / max(float(np.log(len(indices))), 1e-12)


def metric_summary(values, prefix):
    values = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_p10": float(np.percentile(values, 10)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
    }


def evaluate(args):
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    source_vertices = load_geometry_vertices(args.source_mesh)
    target_vertices = load_geometry_vertices(args.target_mesh)
    source_features = load_features(args.source_features)
    target_features = load_features(args.target_features)

    if len(source_vertices) != len(source_features):
        raise ValueError("Source geometry point/vertex count does not match source feature rows.")
    if len(target_vertices) != len(target_features):
        raise ValueError("Target geometry point/vertex count does not match target feature rows.")

    source_indices = sample_indices(source_vertices, args.num_samples, args.sampling, args.seed)
    source_query = source_features[torch.from_numpy(source_indices)]
    target_indices, scores, second_scores = nearest_feature_indices(
        source_query,
        target_features,
        device,
        args.query_chunk_size,
        args.target_chunk_size,
    )

    reverse_source_indices, reverse_scores, _ = nearest_feature_indices(
        target_features[torch.from_numpy(target_indices)],
        source_features,
        device,
        args.query_chunk_size,
        args.target_chunk_size,
    )

    source_points = source_vertices[source_indices]
    target_points = target_vertices[target_indices]
    reverse_source_points = source_vertices[reverse_source_indices]

    source_diag = bbox_diagonal(source_vertices)
    target_diag = bbox_diagonal(target_vertices)
    cycle_distances = np.linalg.norm(source_points - reverse_source_points, axis=1) / source_diag
    source_pair_distances = upper_triangular_distances(source_points) / source_diag
    target_pair_distances = upper_triangular_distances(target_points) / target_diag
    distance_distortion = np.abs(source_pair_distances - target_pair_distances)
    margins = scores - second_scores

    unique_targets = len(np.unique(target_indices))
    target_counts = np.unique(target_indices, return_counts=True)[1]
    mutual = reverse_source_indices == source_indices

    metrics = {
        "source": args.source_name,
        "target": args.target_name,
        "num_samples": int(len(source_indices)),
        "source_vertices": int(len(source_vertices)),
        "target_vertices": int(len(target_vertices)),
        "unique_target_ratio": float(unique_targets / len(target_indices)),
        "max_target_reuse": int(target_counts.max()),
        "target_entropy_normalized": entropy_normalized(target_indices),
        "mutual_exact_ratio": float(mutual.mean()),
        "cycle_within_1pct_bbox": float((cycle_distances <= 0.01).mean()),
        "cycle_within_5pct_bbox": float((cycle_distances <= 0.05).mean()),
        "cycle_within_10pct_bbox": float((cycle_distances <= 0.10).mean()),
        "pairwise_distance_pearson": pearson(source_pair_distances, target_pair_distances),
        "pairwise_distance_spearman": spearman(source_pair_distances, target_pair_distances),
        "pairwise_distance_distortion_mean": float(np.mean(distance_distortion)),
        "pairwise_distance_distortion_median": float(np.median(distance_distortion)),
    }
    metrics.update(metric_summary(scores, "cosine"))
    metrics.update(metric_summary(margins, "nn_margin"))
    metrics.update(metric_summary(cycle_distances, "cycle_distance_bbox"))
    metrics.update(metric_summary(reverse_scores, "reverse_cosine"))

    rows = []
    for i, source_index in enumerate(source_indices):
        rows.append(
            {
                "match_id": i,
                "source_index": int(source_index),
                "target_index": int(target_indices[i]),
                "reverse_source_index": int(reverse_source_indices[i]),
                "cosine_score": float(scores[i]),
                "second_cosine_score": float(second_scores[i]),
                "nn_margin": float(margins[i]),
                "reverse_cosine_score": float(reverse_scores[i]),
                "mutual_exact": bool(mutual[i]),
                "cycle_distance_bbox": float(cycle_distances[i]),
            }
        )
    return metrics, pd.DataFrame(rows)


def main():
    args = parse_args()
    metrics, matches = evaluate(args)
    pair_name = f"{safe_name(args.source_name)}_to_{safe_name(args.target_name)}"
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    metrics_path = outdir / f"{pair_name}_metrics.json"
    csv_path = outdir / f"{pair_name}_diagnostic_matches.csv"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    matches.to_csv(csv_path, index=False)

    print(f"Saved {metrics_path}")
    print(f"Saved {csv_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
