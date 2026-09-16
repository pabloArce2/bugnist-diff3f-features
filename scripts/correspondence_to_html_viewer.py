import argparse
import csv
import html
import json
import math
import re
from pathlib import Path

import numpy as np
import trimesh


SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = SCRIPT_DIR / "templates"
HTML_TEMPLATE_PATH = TEMPLATE_DIR / "correspondence_viewer.html"
JS_TEMPLATE_PATH = TEMPLATE_DIR / "correspondence_viewer.js"

# Benchmark CSV files are rounded for readability, while geometry readers use
# floating-point vertex coordinates. This is deliberately much smaller than a
# visually meaningful mesh displacement but comfortably covers CSV rounding.
COORDINATE_ATOL = 1e-4

BENCHMARK_REQUIRED_COLUMNS = {
    "label",
    "source_index",
    "target_gt_index",
    "predicted_target_index",
    "source_x",
    "source_y",
    "source_z",
    "target_landmark_x",
    "target_landmark_y",
    "target_landmark_z",
    "target_gt_x",
    "target_gt_y",
    "target_gt_z",
    "pred_target_x",
    "pred_target_y",
    "pred_target_z",
}

MATCH_REQUIRED_COLUMNS = {
    "source_index",
    "target_index",
    "source_x",
    "source_y",
    "source_z",
    "target_x",
    "target_y",
    "target_z",
}

NON_METRIC_COLUMNS = {
    "label",
    "source_name",
    "target_name",
    "mutual",
    "source_index",
    "target_index",
    "target_gt_index",
    "predicted_target_index",
    "source_x",
    "source_y",
    "source_z",
    "target_x",
    "target_y",
    "target_z",
    "target_landmark_x",
    "target_landmark_y",
    "target_landmark_z",
    "target_gt_x",
    "target_gt_y",
    "target_gt_z",
    "pred_target_x",
    "pred_target_y",
    "pred_target_z",
}

INTEGER_TEXT = re.compile(r"^[+-]?\d+$")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Create a standalone interactive HTML viewer for point-to-point correspondences."
    )
    parser.add_argument("--source-geometry", required=True, help="Source mesh or point-cloud geometry.")
    parser.add_argument("--target-geometry", required=True, help="Target mesh or point-cloud geometry.")
    parser.add_argument("--source-label", help="Display label; defaults to the source filename stem.")
    parser.add_argument("--target-label", help="Display label; defaults to the target filename stem.")
    csv_group = parser.add_mutually_exclusive_group(required=True)
    csv_group.add_argument(
        "--benchmark-csv",
        help="Detailed CSV produced by evaluate_landmark_benchmark.py.",
    )
    csv_group.add_argument(
        "--matches",
        help="Correspondence CSV produced by compute_feature_correspondences.py or the landmark evaluator.",
    )
    parser.add_argument("--out", required=True, help="Output standalone .html path.")
    parser.add_argument("--title", help="Viewer title; defaults to '<source> to <target> correspondences'.")
    return parser.parse_args(argv)


def _validate_vertices(vertices, path):
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Expected Nx3 geometry coordinates in {path}, got shape {vertices.shape}.")
    if len(vertices) == 0:
        raise ValueError(f"{path} contains no vertices or points.")
    if not np.isfinite(vertices).all():
        bad = np.argwhere(~np.isfinite(vertices))[0]
        raise ValueError(
            f"{path} contains a non-finite coordinate at vertex {int(bad[0])}, axis {int(bad[1])}."
        )
    return vertices


def _validate_faces(faces, vertex_count, path):
    if faces is None:
        return np.empty((0, 3), dtype=np.int64)

    faces = np.asarray(faces)
    if faces.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Expected triangular faces in {path}, got shape {faces.shape}.")
    if not np.issubdtype(faces.dtype, np.integer):
        if not np.isfinite(faces).all() or not np.equal(faces, np.floor(faces)).all():
            raise ValueError(f"{path} contains non-integer face indices.")
    faces = faces.astype(np.int64, copy=False)
    if int(faces.min()) < 0 or int(faces.max()) >= vertex_count:
        raise ValueError(
            f"{path} contains a face index outside [0, {vertex_count - 1}]: "
            f"range is [{int(faces.min())}, {int(faces.max())}]."
        )
    return faces


def _normalize_rgb(colors, vertex_count, path):
    if colors is None:
        return None
    colors = np.asarray(colors)
    if colors.ndim != 2 or len(colors) != vertex_count or colors.shape[1] < 3:
        return None
    colors = colors[:, :3]
    if not np.isfinite(colors).all():
        raise ValueError(f"{path} contains non-finite vertex colors.")

    if np.issubdtype(colors.dtype, np.floating) and colors.size and float(colors.max()) <= 1.0:
        colors = colors * 255.0
    colors = np.rint(np.clip(colors, 0, 255)).astype(np.uint8)
    return colors


def _vertex_colors(geometry, vertex_count, path):
    # PointCloud stores colors directly. Trimesh may synthesize a default visual,
    # so only `kind == "vertex"` is treated as real per-vertex color data.
    if isinstance(geometry, trimesh.points.PointCloud):
        return _normalize_rgb(getattr(geometry, "colors", None), vertex_count, path)

    visual = getattr(geometry, "visual", None)
    if visual is None or getattr(visual, "kind", None) != "vertex":
        return None
    return _normalize_rgb(getattr(visual, "vertex_colors", None), vertex_count, path)


def _geometry_components(loaded, path):
    if isinstance(loaded, trimesh.Scene):
        components = [geometry for geometry in loaded.geometry.values() if hasattr(geometry, "vertices")]
        if not components:
            raise ValueError(f"Could not read vertices or points from scene {path}.")
        return components
    if not hasattr(loaded, "vertices"):
        raise ValueError(f"Could not read vertices or points from {path}.")
    return [loaded]


def _load_trimesh_geometry(path):
    loaded = trimesh.load(path, process=False, maintain_order=True)
    components = _geometry_components(loaded, path)

    vertex_chunks = []
    face_chunks = []
    color_chunks = []
    all_components_colored = True
    vertex_offset = 0

    for component in components:
        vertices = _validate_vertices(component.vertices, path)
        faces = _validate_faces(getattr(component, "faces", None), len(vertices), path)
        colors = _vertex_colors(component, len(vertices), path)

        vertex_chunks.append(vertices)
        if len(faces):
            face_chunks.append(faces + vertex_offset)
        if colors is None:
            all_components_colored = False
        else:
            color_chunks.append(colors)
        vertex_offset += len(vertices)

    vertices = np.concatenate(vertex_chunks, axis=0)
    faces = np.concatenate(face_chunks, axis=0) if face_chunks else np.empty((0, 3), dtype=np.int64)
    colors = np.concatenate(color_chunks, axis=0) if all_components_colored else None
    return vertices, faces, colors


def load_geometry(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Geometry file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        vertices = _validate_vertices(np.load(path, allow_pickle=False), path)
        faces = np.empty((0, 3), dtype=np.int64)
        colors = None
    elif suffix in {".xyz", ".txt"}:
        vertices = _validate_vertices(np.loadtxt(path, dtype=np.float64, ndmin=2), path)
        faces = np.empty((0, 3), dtype=np.int64)
        colors = None
    else:
        vertices, faces, colors = _load_trimesh_geometry(path)

    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    center = (minimum + maximum) / 2.0
    diagonal = float(np.linalg.norm(maximum - minimum))
    return {
        "path": str(path),
        "vertices_array": vertices,
        "faces_array": faces,
        "colors_array": colors,
        "center_array": center,
        "diagonal": diagonal,
    }


def _required_columns(reader, required, path, kind):
    fieldnames = reader.fieldnames
    if not fieldnames:
        raise ValueError(f"{path} has no CSV header.")
    missing = sorted(required - set(fieldnames))
    if missing:
        raise ValueError(f"{kind} CSV {path} is missing required columns: {', '.join(missing)}")


def _row_value(row, column, path, row_number):
    value = row.get(column)
    if value is None or not str(value).strip():
        raise ValueError(f"{path} row {row_number} has an empty {column!r} value.")
    return str(value).strip()


def _parse_float(row, column, path, row_number):
    text = _row_value(row, column, path, row_number)
    try:
        value = float(text)
    except ValueError as error:
        raise ValueError(f"{path} row {row_number} has invalid numeric {column}={text!r}.") from error
    if not math.isfinite(value):
        raise ValueError(f"{path} row {row_number} has non-finite {column}={text!r}.")
    return value


def _parse_index(row, column, path, row_number):
    value = _parse_float(row, column, path, row_number)
    if not value.is_integer():
        raise ValueError(f"{path} row {row_number} has non-integer {column}={value!r}.")
    return int(value)


def _parse_xyz(row, columns, path, row_number):
    return np.array([_parse_float(row, column, path, row_number) for column in columns], dtype=np.float64)


def _parse_mutual(row, path, row_number):
    raw = row.get("mutual")
    if raw is None or not str(raw).strip():
        return None
    text = str(raw).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"{path} row {row_number} has invalid mutual={raw!r}; expected true or false.")


def _numeric_metrics(row, path, row_number):
    metrics = {}
    for column, raw in row.items():
        if column is None or column in NON_METRIC_COLUMNS or raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        try:
            value = float(text)
        except ValueError:
            continue
        if not math.isfinite(value):
            raise ValueError(f"{path} row {row_number} has non-finite metric {column}={text!r}.")
        metrics[column] = int(text) if INTEGER_TEXT.fullmatch(text) else value
    return metrics


def _validate_index(index, vertex_count, column, path, row_number):
    if index < 0 or index >= vertex_count:
        raise ValueError(
            f"{path} row {row_number} has {column}={index}, outside geometry index range "
            f"[0, {vertex_count - 1}]."
        )


def _validate_endpoint(csv_point, geometry_point, role, index, path, row_number, label):
    if np.allclose(csv_point, geometry_point, rtol=0.0, atol=COORDINATE_ATOL):
        return
    delta = float(np.max(np.abs(csv_point - geometry_point)))
    csv_text = ", ".join(f"{value:.9g}" for value in csv_point)
    geometry_text = ", ".join(f"{value:.9g}" for value in geometry_point)
    raise ValueError(
        f"{path} row {row_number} ({label!r}) {role} coordinates [{csv_text}] do not match "
        f"geometry vertex {index} [{geometry_text}] (max difference {delta:.6g}, allowed "
        f"{COORDINATE_ATOL:g}). The CSV was likely generated for a different or reordered geometry "
        f"(wrong mesh)."
    )


def _match_label(row, row_number):
    label = str(row.get("label") or "").strip()
    if label:
        return label
    match_id = str(row.get("match_id") or "").strip()
    return f"match {match_id}" if match_id else f"match {row_number - 2}"


def _benchmark_match(row, source_vertices, target_vertices, path, row_number):
    label = _match_label(row, row_number)
    source_index = _parse_index(row, "source_index", path, row_number)
    target_index = _parse_index(row, "predicted_target_index", path, row_number)
    gt_index = _parse_index(row, "target_gt_index", path, row_number)
    _validate_index(source_index, len(source_vertices), "source_index", path, row_number)
    _validate_index(target_index, len(target_vertices), "predicted_target_index", path, row_number)
    _validate_index(gt_index, len(target_vertices), "target_gt_index", path, row_number)

    source_csv = _parse_xyz(row, ("source_x", "source_y", "source_z"), path, row_number)
    target_csv = _parse_xyz(row, ("pred_target_x", "pred_target_y", "pred_target_z"), path, row_number)
    gt_vertex_csv = _parse_xyz(row, ("target_gt_x", "target_gt_y", "target_gt_z"), path, row_number)
    # This is the unsnapped manual annotation. It is intentionally used for the
    # error line; using target_gt_* would measure a different, vertex-space error.
    manual_gt = _parse_xyz(
        row,
        ("target_landmark_x", "target_landmark_y", "target_landmark_z"),
        path,
        row_number,
    )

    source_point = source_vertices[source_index]
    target_point = target_vertices[target_index]
    gt_vertex = target_vertices[gt_index]
    _validate_endpoint(source_csv, source_point, "source", source_index, path, row_number, label)
    _validate_endpoint(target_csv, target_point, "predicted target", target_index, path, row_number, label)
    _validate_endpoint(gt_vertex_csv, gt_vertex, "snapped target ground-truth", gt_index, path, row_number, label)

    return {
        "label": label,
        "sourceIndex": source_index,
        "targetIndex": target_index,
        "gtIndex": gt_index,
        "source": source_point.tolist(),
        "target": target_point.tolist(),
        "gt": manual_gt.tolist(),
        "gtVertex": gt_vertex.tolist(),
        "metrics": _numeric_metrics(row, path, row_number),
        "mutual": _parse_mutual(row, path, row_number),
    }


def _plain_match(row, source_vertices, target_vertices, path, row_number):
    label = _match_label(row, row_number)
    source_index = _parse_index(row, "source_index", path, row_number)
    target_index = _parse_index(row, "target_index", path, row_number)
    _validate_index(source_index, len(source_vertices), "source_index", path, row_number)
    _validate_index(target_index, len(target_vertices), "target_index", path, row_number)

    source_csv = _parse_xyz(row, ("source_x", "source_y", "source_z"), path, row_number)
    target_csv = _parse_xyz(row, ("target_x", "target_y", "target_z"), path, row_number)
    source_point = source_vertices[source_index]
    target_point = target_vertices[target_index]
    _validate_endpoint(source_csv, source_point, "source", source_index, path, row_number, label)
    _validate_endpoint(target_csv, target_point, "target", target_index, path, row_number, label)

    return {
        "label": label,
        "sourceIndex": source_index,
        "targetIndex": target_index,
        "gtIndex": None,
        "source": source_point.tolist(),
        "target": target_point.tolist(),
        "gt": None,
        "gtVertex": None,
        "metrics": _numeric_metrics(row, path, row_number),
        "mutual": _parse_mutual(row, path, row_number),
    }


def load_matches(path, kind, source_vertices, target_vertices):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Correspondence CSV does not exist: {path}")
    if kind not in {"benchmark", "matches"}:
        raise ValueError(f"Unsupported correspondence CSV kind: {kind!r}")

    rows = []
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = BENCHMARK_REQUIRED_COLUMNS if kind == "benchmark" else MATCH_REQUIRED_COLUMNS
        _required_columns(reader, required, path, kind)
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"{path} row {row_number} has more values than the CSV header.")
            parser = _benchmark_match if kind == "benchmark" else _plain_match
            rows.append(parser(row, source_vertices, target_vertices, path, row_number))

    if not rows:
        raise ValueError(f"{path} contains no correspondence rows.")
    return rows


def _payload_geometry(geometry, name):
    colors = geometry["colors_array"]
    return {
        "name": name,
        "path": geometry["path"],
        "vertices": geometry["vertices_array"].tolist(),
        "faces": geometry["faces_array"].tolist(),
        "colors": colors.tolist() if colors is not None else None,
        "center": geometry["center_array"].tolist(),
    }


def build_payload(source_geometry, target_geometry, source_name, target_name, matches, csv_path, kind, title):
    scale = max(source_geometry["diagonal"], target_geometry["diagonal"])
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("Source and target geometry have zero shared spatial extent; cannot build a 3-D view.")
    return {
        "title": title,
        "source": _payload_geometry(source_geometry, source_name),
        "target": _payload_geometry(target_geometry, target_name),
        "scale": scale,
        "matches": matches,
        "meta": {
            "csv": str(Path(csv_path)),
            "kind": kind,
            "coordinateSystem": "Original geometry coordinates",
        },
    }


def render_html(payload, title, html_template_path=HTML_TEMPLATE_PATH, js_template_path=JS_TEMPLATE_PATH):
    html_template_path = Path(html_template_path)
    js_template_path = Path(js_template_path)
    if not html_template_path.is_file():
        raise FileNotFoundError(f"HTML viewer template does not exist: {html_template_path}")
    if not js_template_path.is_file():
        raise FileNotFoundError(f"JavaScript viewer template does not exist: {js_template_path}")

    template = html_template_path.read_text(encoding="utf-8")
    script = js_template_path.read_text(encoding="utf-8")
    for token in ("__TITLE__", "__DATA__", "__SCRIPT__"):
        if token not in template:
            raise ValueError(f"HTML viewer template {html_template_path} is missing placeholder {token}.")

    data_json = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    safe_data_json = data_json.replace("<", "\\u003c")
    replacements = {
        "__TITLE__": html.escape(title, quote=True),
        "__DATA__": safe_data_json,
        "__SCRIPT__": script,
    }
    # A one-pass substitution prevents a label or path containing a literal
    # placeholder such as "__SCRIPT__" from being rewritten after insertion.
    return re.sub(r"__TITLE__|__DATA__|__SCRIPT__", lambda match: replacements[match.group(0)], template)


def main(argv=None):
    args = parse_args(argv)
    source_path = Path(args.source_geometry)
    target_path = Path(args.target_geometry)
    source_name = args.source_label or source_path.stem
    target_name = args.target_label or target_path.stem
    title = args.title or f"{source_name} to {target_name} correspondences"
    csv_path = Path(args.benchmark_csv or args.matches)
    kind = "benchmark" if args.benchmark_csv else "matches"

    source_geometry = load_geometry(source_path)
    target_geometry = load_geometry(target_path)
    matches = load_matches(
        csv_path,
        kind,
        source_geometry["vertices_array"],
        target_geometry["vertices_array"],
    )
    payload = build_payload(
        source_geometry,
        target_geometry,
        source_name,
        target_name,
        matches,
        csv_path,
        kind,
        title,
    )
    content = render_html(payload, title)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    print(f"Saved interactive correspondence viewer: {out_path}")
    print(
        f"Embedded source: {len(source_geometry['vertices_array']):,} vertices, "
        f"{len(source_geometry['faces_array']):,} faces"
    )
    print(
        f"Embedded target: {len(target_geometry['vertices_array']):,} vertices, "
        f"{len(target_geometry['faces_array']):,} faces"
    )
    print(f"Embedded correspondences: {len(matches):,} ({kind})")


if __name__ == "__main__":
    main()
