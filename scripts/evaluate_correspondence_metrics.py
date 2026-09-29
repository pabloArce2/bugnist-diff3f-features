"""Label-free checks of how nearest-neighbour descriptor matching behaves between two shapes.

For --num-samples source vertices (farthest-point sampling by default):
  cosine, nn_margin           similarity of the best match and its lead over the runner-up
  cycle_within_Xpct_bbox      source -> target -> source ends within X% of the source bbox diagonal
  mutual_exact_ratio          the round trip returns to exactly the same vertex
  unique_target_ratio,
  max_target_reuse            how often several queries collapse onto the same target vertex
  pairwise_distance_spearman  whether distances between the queries survive in their matches
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bugnist_tools.features import load_features
from bugnist_tools.geometry import bbox_diagonal, check_rows, load_vertices
from bugnist_tools.matching import nearest_two, sample_indices
from bugnist_tools.util import pick_device, safe_name


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate feature-correspondence diagnostics without ground truth.")
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--source-mesh", required=True, help="Source mesh or point cloud.")
    parser.add_argument("--source-features", required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--target-mesh", required=True, help="Target mesh or point cloud.")
    parser.add_argument("--target-features", required=True)
    parser.add_argument("--outdir", default="visualizations/correspondence_metrics")
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--sampling", choices=("farthest", "random"), default="farthest")
    parser.add_argument("--device", default=None)
    parser.add_argument("--query-chunk-size", type=int, default=16)
    parser.add_argument("--target-chunk-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


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
    device = pick_device(args.device)
    print(f"Using device: {device}")

    source_vertices = load_vertices(args.source_mesh)
    target_vertices = load_vertices(args.target_mesh)
    source_features = load_features(args.source_features)
    target_features = load_features(args.target_features)
    check_rows(args.source_name, len(source_vertices), len(source_features))
    check_rows(args.target_name, len(target_vertices), len(target_features))

    source_indices = sample_indices(source_vertices, args.num_samples, args.sampling, args.seed)
    source_query = source_features[torch.from_numpy(source_indices)]
    target_indices, scores, second_scores = nearest_two(
        source_query,
        target_features,
        device,
        args.query_chunk_size,
        args.target_chunk_size,
    )

    reverse_source_indices, reverse_scores, _ = nearest_two(
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
