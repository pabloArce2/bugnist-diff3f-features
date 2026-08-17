import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import trimesh
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Compute sampled feature nearest-neighbor correspondences.")
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--source-mesh", required=True, help="Source mesh or point cloud geometry.")
    parser.add_argument("--source-features", required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--target-mesh", required=True, help="Target mesh or point cloud geometry.")
    parser.add_argument("--target-features", required=True)
    parser.add_argument("--outdir", default="visualizations/correspondences")
    parser.add_argument("--num-source-points", type=int, default=80)
    parser.add_argument("--sampling", choices=("farthest", "random"), default="farthest")
    parser.add_argument("--source-indices", help="Optional text file with one source vertex index per line.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--query-chunk-size", type=int, default=8)
    parser.add_argument("--target-chunk-size", type=int, default=8192)
    parser.add_argument("--mutual-check", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
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
        raise ValueError(f"Expected [num_vertices, feature_dim], got {tuple(features.shape)} for {path}.")
    return torch.nan_to_num(features.float())


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


def farthest_point_indices(vertices, count, seed):
    rng = np.random.default_rng(seed)
    count = min(count, len(vertices))
    center = vertices.mean(axis=0, keepdims=True)
    first = int(np.argmax(np.sum((vertices - center) ** 2, axis=1)))
    selected = np.empty(count, dtype=np.int64)
    selected[0] = first
    min_dist2 = np.sum((vertices - vertices[first]) ** 2, axis=1)

    for i in range(1, count):
        if i % 10 == 0:
            # Small randomness prevents all samples sitting on one long limb for very elongated scans.
            top = np.argpartition(min_dist2, -min(64, len(min_dist2)))[-min(64, len(min_dist2)) :]
            chosen = int(rng.choice(top))
        else:
            chosen = int(np.argmax(min_dist2))
        selected[i] = chosen
        dist2 = np.sum((vertices - vertices[chosen]) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, dist2)
    return selected


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


def nearest_feature_indices(query_features, target_features, device, query_chunk_size, target_chunk_size):
    target_features = torch.nn.functional.normalize(target_features, dim=1)
    query_features = torch.nn.functional.normalize(query_features, dim=1)
    best_indices = torch.empty(len(query_features), dtype=torch.long)
    best_scores = torch.full((len(query_features),), -float("inf"), dtype=torch.float32)

    for q_start in tqdm(range(0, len(query_features), query_chunk_size), desc="Query chunks"):
        q_stop = min(q_start + query_chunk_size, len(query_features))
        query = query_features[q_start:q_stop].to(device)
        chunk_best_scores = torch.full((q_stop - q_start,), -float("inf"), device=device)
        chunk_best_indices = torch.zeros((q_stop - q_start,), dtype=torch.long, device=device)

        for t_start in range(0, len(target_features), target_chunk_size):
            t_stop = min(t_start + target_chunk_size, len(target_features))
            target = target_features[t_start:t_stop].to(device)
            similarity = query @ target.T
            scores, local_indices = similarity.max(dim=1)
            update = scores > chunk_best_scores
            chunk_best_scores[update] = scores[update]
            chunk_best_indices[update] = local_indices[update] + t_start

        best_indices[q_start:q_stop] = chunk_best_indices.cpu()
        best_scores[q_start:q_stop] = chunk_best_scores.cpu()

    return best_indices.numpy(), best_scores.numpy()


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
    if source_features.shape[1] != target_features.shape[1]:
        raise ValueError("Source and target feature dimensions differ.")

    source_indices = source_indices_from_args(args, source_vertices)
    query_features = source_features[torch.from_numpy(source_indices)]
    target_indices, scores = nearest_feature_indices(
        query_features,
        target_features,
        device,
        args.query_chunk_size,
        args.target_chunk_size,
    )

    mutual = np.full(len(source_indices), False)
    if args.mutual_check:
        reverse_indices, _ = nearest_feature_indices(
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
