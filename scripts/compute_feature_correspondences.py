"""Match sampled source vertices to their most similar target vertices (cosine similarity).

Writes <source>_to_<target>_matches.csv and .npz. With --mutual-check each match
is also searched back from the target to see if it returns to the same vertex.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bugnist_tools.features import load_features
from bugnist_tools.geometry import check_rows, load_vertices
from bugnist_tools.matching import farthest_point_indices, nearest_neighbors
from bugnist_tools.util import pick_device, safe_name


def parse_args():
    parser = argparse.ArgumentParser(description="Compute sampled feature nearest-neighbor correspondences.")
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--source-mesh", required=True, help="Source mesh or point cloud.")
    parser.add_argument("--source-features", required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--target-mesh", required=True, help="Target mesh or point cloud.")
    parser.add_argument("--target-features", required=True)
    parser.add_argument("--outdir", default="visualizations/correspondences")
    parser.add_argument("--num-source-points", type=int, default=80)
    parser.add_argument("--sampling", choices=("farthest", "random"), default="farthest")
    parser.add_argument("--source-indices", help="Text file with one source vertex index per line, instead of sampling.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--query-chunk-size", type=int, default=8)
    parser.add_argument("--target-chunk-size", type=int, default=8192)
    parser.add_argument("--mutual-check", action="store_true", help="Also search back from each matched target vertex.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def source_indices_from_args(args, vertices):
    if args.source_indices:
        values = []
        for line in Path(args.source_indices).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                values.append(int(line))
        indices = np.array(values, dtype=np.int64)
    elif args.sampling == "random":
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(len(vertices), size=min(args.num_source_points, len(vertices)), replace=False)
    else:
        indices = farthest_point_indices(vertices, args.num_source_points, args.seed)

    if np.any(indices < 0) or np.any(indices >= len(vertices)):
        raise ValueError("Source indices contain values outside the source mesh vertex range.")
    return indices


def write_outputs(outdir, pair_name, rows, source_indices, target_indices, scores):
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"{pair_name}_matches.csv"
    npz_path = outdir / f"{pair_name}_matches.npz"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    np.savez(
        npz_path,
        source_indices=source_indices,
        target_indices=target_indices,
        scores=scores,
    )
    print(f"Saved {csv_path}")
    print(f"Saved {npz_path}")


def main():
    args = parse_args()
    device = pick_device(args.device)
    print(f"Using device: {device}")

    source_vertices = load_vertices(args.source_mesh)
    target_vertices = load_vertices(args.target_mesh)
    source_features = load_features(args.source_features)
    target_features = load_features(args.target_features)
    check_rows(args.source_name, len(source_vertices), len(source_features))
    check_rows(args.target_name, len(target_vertices), len(target_features))
    if source_features.shape[1] != target_features.shape[1]:
        raise ValueError("Source and target feature dimensions differ.")

    source_indices = source_indices_from_args(args, source_vertices)
    query_features = source_features[torch.from_numpy(source_indices)]
    target_indices, scores = nearest_neighbors(
        query_features,
        target_features,
        device,
        args.query_chunk_size,
        args.target_chunk_size,
    )

    mutual = np.full(len(source_indices), False)
    if args.mutual_check:
        reverse_indices, _ = nearest_neighbors(
            target_features[torch.from_numpy(target_indices)],
            source_features,
            device,
            args.query_chunk_size,
            args.target_chunk_size,
        )
        mutual = reverse_indices == source_indices

    rows = []
    for i, (source_index, target_index, score, is_mutual) in enumerate(
        zip(source_indices, target_indices, scores, mutual)
    ):
        source_xyz = source_vertices[source_index]
        target_xyz = target_vertices[target_index]
        rows.append(
            {
                "match_id": i,
                "source_name": args.source_name,
                "target_name": args.target_name,
                "source_index": int(source_index),
                "target_index": int(target_index),
                "cosine_score": float(score),
                "mutual": bool(is_mutual),
                "source_x": float(source_xyz[0]),
                "source_y": float(source_xyz[1]),
                "source_z": float(source_xyz[2]),
                "target_x": float(target_xyz[0]),
                "target_y": float(target_xyz[1]),
                "target_z": float(target_xyz[2]),
            }
        )

    pair_name = f"{safe_name(args.source_name)}_to_{safe_name(args.target_name)}"
    write_outputs(Path(args.outdir), pair_name, rows, source_indices, target_indices, scores)
    print(f"Mean cosine score: {float(np.mean(scores)):.4f}")
    print(f"Median cosine score: {float(np.median(scores)):.4f}")
    if args.mutual_check:
        print(f"Mutual matches: {int(mutual.sum())}/{len(mutual)}")


if __name__ == "__main__":
    main()
