"""Score Diff3F correspondences against manually placed landmarks.

For every label present in both landmark CSVs (label,x,y,z, from
export_blender_landmarks.py): snap the source landmark to its nearest vertex,
find the target vertex with the most similar descriptor, and measure how far
that prediction is from the manual target landmark, as a fraction of the
target's bounding-box diagonal. Also reports where the true target vertex
ranks among all target vertices by similarity.

Outputs <source>_to_<target>_landmark_benchmark.csv (one row per landmark),
..._landmark_summary.json (PCK and averages) and ..._predicted_matches.csv.
The direction matters: run it once per direction.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bugnist_tools.features import load_features
from bugnist_tools.geometry import bbox_diagonal, check_rows, load_vertices
from bugnist_tools.util import pick_device, safe_name


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Diff3F correspondences against manual landmark labels."
    )
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--source-geometry", required=True, help="Source mesh or point cloud.")
    parser.add_argument("--source-features", required=True)
    parser.add_argument("--source-landmarks", required=True, help="CSV with columns label,x,y,z.")
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--target-geometry", required=True, help="Target mesh or point cloud.")
    parser.add_argument("--target-features", required=True)
    parser.add_argument("--target-landmarks", required=True, help="CSV with columns label,x,y,z.")
    parser.add_argument("--outdir", default="visualizations/landmark_benchmark")
    parser.add_argument("--device", default=None)
    parser.add_argument("--geometry-chunk-size", type=int, default=200000)
    parser.add_argument("--target-chunk-size", type=int, default=8192)
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.01, 0.02, 0.05, 0.10],
        help="PCK thresholds as fractions of target bounding-box diagonal.",
    )
    return parser.parse_args()


def canonical_column(columns, candidates):
    lower = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def load_landmarks(path):
    path = Path(path)
    df = pd.read_csv(path)
    label_col = canonical_column(df.columns, ("label", "landmark", "name"))
    x_col = canonical_column(df.columns, ("x", "local_x"))
    y_col = canonical_column(df.columns, ("y", "local_y"))
    z_col = canonical_column(df.columns, ("z", "local_z"))
    missing = [
        name
        for name, column in (("label", label_col), ("x", x_col), ("y", y_col), ("z", z_col))
        if column is None
    ]
    if missing:
        raise ValueError(f"{path} is missing required landmark columns: {', '.join(missing)}")

    landmarks = {}
    for _, row in df.iterrows():
        label = str(row[label_col]).strip()
        if not label:
            continue
        if label in landmarks:
            raise ValueError(f"Duplicate landmark label {label!r} in {path}.")
        landmarks[label] = np.array([row[x_col], row[y_col], row[z_col]], dtype=np.float32)
    if not landmarks:
        raise ValueError(f"No landmarks found in {path}.")
    return landmarks


def nearest_geometry_indices(vertices, query_points, chunk_size):
    query_points = np.asarray(query_points, dtype=np.float32)
    best_indices = np.zeros(len(query_points), dtype=np.int64)
    best_dist2 = np.full(len(query_points), np.inf, dtype=np.float64)

    for start in range(0, len(vertices), max(int(chunk_size), 1)):
        stop = min(start + max(int(chunk_size), 1), len(vertices))
        chunk = vertices[start:stop]
        diff = query_points[:, None, :] - chunk[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        local_indices = np.argmin(dist2, axis=1)
        local_dist2 = dist2[np.arange(len(query_points)), local_indices]
        update = local_dist2 < best_dist2
        best_dist2[update] = local_dist2[update]
        best_indices[update] = local_indices[update] + start

    return best_indices, np.sqrt(best_dist2)


def nearest_feature_matches(query_features, target_features, gt_indices, device, target_chunk_size):
    query = torch.nn.functional.normalize(torch.nan_to_num(query_features.float()), dim=1)
    target = torch.nn.functional.normalize(torch.nan_to_num(target_features.float()), dim=1)
    gt_indices = np.asarray(gt_indices, dtype=np.int64)

    best_indices = np.zeros(len(query), dtype=np.int64)
    best_scores = np.full(len(query), -np.inf, dtype=np.float32)
    second_scores = np.full(len(query), -np.inf, dtype=np.float32)
    gt_scores = np.full(len(query), np.nan, dtype=np.float32)
    query_device = query.to(device)

    chunk_size = max(int(target_chunk_size), 1)
    with torch.no_grad():
        for start in tqdm(range(0, len(target), chunk_size), desc="Feature target chunks"):
            stop = min(start + chunk_size, len(target))
            target_chunk = target[start:stop].to(device)
            similarity = query_device @ target_chunk.T
            values, local_indices = torch.topk(similarity, k=min(2, similarity.shape[1]), dim=1)
            values = values.cpu().numpy()
            local_indices = local_indices.cpu().numpy()

            for q_idx in range(len(query)):
                for candidate_rank in range(values.shape[1]):
                    score = float(values[q_idx, candidate_rank])
                    index = int(local_indices[q_idx, candidate_rank]) + start
                    if score > best_scores[q_idx]:
                        second_scores[q_idx] = best_scores[q_idx]
                        best_scores[q_idx] = score
                        best_indices[q_idx] = index
                    elif score > second_scores[q_idx] and index != best_indices[q_idx]:
                        second_scores[q_idx] = score

                gt_index = int(gt_indices[q_idx])
                if start <= gt_index < stop:
                    gt_scores[q_idx] = float(similarity[q_idx, gt_index - start].detach().cpu())

    if np.any(np.isnan(gt_scores)):
        missing = np.where(np.isnan(gt_scores))[0].tolist()
        raise RuntimeError(f"Could not compute ground-truth scores for query rows: {missing}")

    rank_counts = np.zeros(len(query), dtype=np.int64)
    gt_scores_device = torch.from_numpy(gt_scores).to(device)
    with torch.no_grad():
        for start in tqdm(range(0, len(target), chunk_size), desc="Feature rank chunks"):
            stop = min(start + chunk_size, len(target))
            target_chunk = target[start:stop].to(device)
            similarity = query_device @ target_chunk.T
            rank_counts += (similarity > gt_scores_device[:, None]).sum(dim=1).cpu().numpy()

    gt_ranks = rank_counts + 1
    return best_indices, best_scores, second_scores, gt_scores, gt_ranks


def round_floats(value, ndigits):
    """Round floats inside nested dicts/lists. Long digit runs confuse spreadsheet imports."""
    if isinstance(value, float):
        return round(value, ndigits)
    if isinstance(value, dict):
        return {key: round_floats(item, ndigits) for key, item in value.items()}
    if isinstance(value, list):
        return [round_floats(item, ndigits) for item in value]
    return value


def round_dataframe_floats(df, ndigits):
    df = df.copy()
    float_columns = df.select_dtypes(include=["float64", "float32"]).columns
    df[float_columns] = df[float_columns].round(ndigits)
    return df


def summarize(values, prefix):
    values = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
    }


def evaluate(args):
    device = pick_device(args.device)
    print(f"Using device: {device}")

    source_vertices = load_vertices(args.source_geometry)
    target_vertices = load_vertices(args.target_geometry)
    source_features = load_features(args.source_features)
    target_features = load_features(args.target_features)
    check_rows(args.source_name, len(source_vertices), len(source_features))
    check_rows(args.target_name, len(target_vertices), len(target_features))
    if source_features.shape[1] != target_features.shape[1]:
        raise ValueError("Source and target feature dimensions differ.")

    source_landmarks = load_landmarks(args.source_landmarks)
    target_landmarks = load_landmarks(args.target_landmarks)
    common_labels = [label for label in source_landmarks if label in target_landmarks]
    missing_in_target = sorted(set(source_landmarks) - set(target_landmarks))
    missing_in_source = sorted(set(target_landmarks) - set(source_landmarks))
    if not common_labels:
        raise ValueError("No landmark labels are shared between source and target CSV files.")

    source_points = np.stack([source_landmarks[label] for label in common_labels], axis=0)
    target_points = np.stack([target_landmarks[label] for label in common_labels], axis=0)

    source_indices, source_snap_dist = nearest_geometry_indices(
        source_vertices,
        source_points,
        args.geometry_chunk_size,
    )
    target_gt_indices, target_snap_dist = nearest_geometry_indices(
        target_vertices,
        target_points,
        args.geometry_chunk_size,
    )

    predicted_indices, scores, second_scores, gt_scores, gt_ranks = nearest_feature_matches(
        source_features[torch.from_numpy(source_indices)],
        target_features,
        target_gt_indices,
        device,
        args.target_chunk_size,
    )

    target_diag = bbox_diagonal(target_vertices)
    predicted_points = target_vertices[predicted_indices]
    target_gt_vertex_points = target_vertices[target_gt_indices]
    manual_errors = np.linalg.norm(predicted_points - target_points, axis=1)
    vertex_errors = np.linalg.norm(predicted_points - target_gt_vertex_points, axis=1)
    manual_errors_bbox = manual_errors / target_diag
    vertex_errors_bbox = vertex_errors / target_diag
    margins = scores - second_scores

    rows = []
    matches_rows = []
    for i, label in enumerate(common_labels):
        source_xyz = source_vertices[source_indices[i]]
        target_pred_xyz = predicted_points[i]
        target_gt_xyz = target_gt_vertex_points[i]
        row = {
            "label": label,
            "source_index": int(source_indices[i]),
            "target_gt_index": int(target_gt_indices[i]),
            "predicted_target_index": int(predicted_indices[i]),
            "cosine_score": float(scores[i]),
            "second_cosine_score": float(second_scores[i]),
            "nn_margin": float(margins[i]),
            "gt_cosine_score": float(gt_scores[i]),
            "gt_feature_rank": int(gt_ranks[i]),
            "target_error": float(manual_errors[i]),
            "target_error_bbox": float(manual_errors_bbox[i]),
            "target_error_bbox_pct": float(manual_errors_bbox[i] * 100.0),
            "target_vertex_error": float(vertex_errors[i]),
            "target_vertex_error_bbox": float(vertex_errors_bbox[i]),
            "source_snap_distance": float(source_snap_dist[i]),
            "target_snap_distance": float(target_snap_dist[i]),
            "source_landmark_x": float(source_points[i, 0]),
            "source_landmark_y": float(source_points[i, 1]),
            "source_landmark_z": float(source_points[i, 2]),
            "target_landmark_x": float(target_points[i, 0]),
            "target_landmark_y": float(target_points[i, 1]),
            "target_landmark_z": float(target_points[i, 2]),
            "source_x": float(source_xyz[0]),
            "source_y": float(source_xyz[1]),
            "source_z": float(source_xyz[2]),
            "target_gt_x": float(target_gt_xyz[0]),
            "target_gt_y": float(target_gt_xyz[1]),
            "target_gt_z": float(target_gt_xyz[2]),
            "pred_target_x": float(target_pred_xyz[0]),
            "pred_target_y": float(target_pred_xyz[1]),
            "pred_target_z": float(target_pred_xyz[2]),
        }
        for threshold in args.thresholds:
            row[f"pck_{threshold:g}"] = bool(manual_errors_bbox[i] <= threshold)
        rows.append(row)

        matches_rows.append(
            {
                "match_id": i,
                "label": label,
                "source_name": args.source_name,
                "target_name": args.target_name,
                "source_index": int(source_indices[i]),
                "target_index": int(predicted_indices[i]),
                "cosine_score": float(scores[i]),
                "mutual": False,
                "source_x": float(source_xyz[0]),
                "source_y": float(source_xyz[1]),
                "source_z": float(source_xyz[2]),
                "target_x": float(target_pred_xyz[0]),
                "target_y": float(target_pred_xyz[1]),
                "target_z": float(target_pred_xyz[2]),
            }
        )

    summary = {
        "source": args.source_name,
        "target": args.target_name,
        "num_common_landmarks": int(len(common_labels)),
        "common_labels": common_labels,
        "missing_in_source": missing_in_source,
        "missing_in_target": missing_in_target,
        "source_geometry_rows": int(len(source_vertices)),
        "target_geometry_rows": int(len(target_vertices)),
        "feature_dim": int(source_features.shape[1]),
        "target_bbox_diagonal": float(target_diag),
        "top1_exact_index_ratio": float(np.mean(predicted_indices == target_gt_indices)),
        "gt_rank_top5_ratio": float(np.mean(gt_ranks <= 5)),
        "gt_rank_top10_ratio": float(np.mean(gt_ranks <= 10)),
        "gt_rank_top100_ratio": float(np.mean(gt_ranks <= 100)),
        "gt_rank_top1000_ratio": float(np.mean(gt_ranks <= 1000)),
    }
    for threshold in args.thresholds:
        summary[f"pck_at_{threshold:g}_bbox"] = float(np.mean(manual_errors_bbox <= threshold))
    summary.update(summarize(manual_errors_bbox, "target_error_bbox"))
    summary.update(summarize(vertex_errors_bbox, "target_vertex_error_bbox"))
    summary.update(summarize(scores, "cosine"))
    summary.update(summarize(margins, "nn_margin"))
    summary.update(summarize(gt_ranks, "gt_feature_rank"))

    return summary, pd.DataFrame(rows), pd.DataFrame(matches_rows)


def main():
    args = parse_args()
    summary, rows, matches = evaluate(args)
    pair_name = f"{safe_name(args.source_name)}_to_{safe_name(args.target_name)}"
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary_path = outdir / f"{pair_name}_landmark_summary.json"
    csv_path = outdir / f"{pair_name}_landmark_benchmark.csv"
    matches_path = outdir / f"{pair_name}_predicted_matches.csv"

    rounded_summary = round_floats(summary, 6)
    summary_path.write_text(json.dumps(rounded_summary, indent=2) + "\n", encoding="utf-8")
    round_dataframe_floats(rows, 6).to_csv(csv_path, index=False)
    round_dataframe_floats(matches, 6).to_csv(matches_path, index=False)

    print(f"Saved {summary_path}")
    print(f"Saved {csv_path}")
    print(f"Saved {matches_path}")
    print(json.dumps(rounded_summary, indent=2))


if __name__ == "__main__":
    main()
