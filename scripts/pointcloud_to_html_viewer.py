"""A single self-contained HTML file for looking at a coloured point cloud in a browser.

Drag to rotate, scroll to zoom, click a point to see its row index and coordinates.
"""

import argparse
import html
from pathlib import Path

import numpy as np
import trimesh


def parse_args():
    parser = argparse.ArgumentParser(description="Create a standalone interactive HTML viewer for a colored point cloud.")
    parser.add_argument("--pointcloud", required=True, help="Input colored .ply point cloud.")
    parser.add_argument("--out", required=True, help="Output .html path.")
    parser.add_argument("--title", default="Diff3F point-cloud features")
    parser.add_argument("--max-points", type=int, default=50000, help="Random subset for lighter browser rendering.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--point-size", type=float, default=2.2)
    parser.add_argument("--background", default="#f8f8f6")
    return parser.parse_args()


def load_colored_points(path):
    loaded = trimesh.load(path, process=False, maintain_order=True)
    if not hasattr(loaded, "vertices"):
        raise ValueError(f"Could not read point positions from {path}.")

    points = np.asarray(loaded.vertices, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected Nx3 point coordinates, got shape {points.shape}.")
    if len(points) == 0:
        raise ValueError(f"{path} contains no points.")

    colors = getattr(loaded, "colors", None)
    if colors is None or len(colors) != len(points):
        visual = getattr(loaded, "visual", None)
        colors = getattr(visual, "vertex_colors", None)
    if colors is None or len(colors) != len(points):
        colors = np.full((len(points), 3), 178, dtype=np.uint8)
    else:
        colors = np.asarray(colors, dtype=np.uint8)
        if colors.ndim != 2 or colors.shape[1] < 3:
            colors = np.full((len(points), 3), 178, dtype=np.uint8)
        else:
            colors = colors[:, :3]
    return points, colors


def subset_points(points, colors, max_points, seed, return_indices=False):
    indices = np.arange(len(points), dtype=np.uint32)
    if max_points is None or len(points) <= max_points:
        subset = (points, colors, indices)
    else:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(points), size=max_points, replace=False)
        subset = (points[indices], colors[indices], indices.astype(np.uint32, copy=False))
    return subset if return_indices else subset[:2]


def normalize_points(points):
    center = (points.min(axis=0) + points.max(axis=0)) / 2.0
    centered = points - center
    scale = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
    scale = max(float(scale), 1e-6)
    return centered / scale, center, scale


def pack_colors(colors):
    colors = np.asarray(colors, dtype=np.uint32)
    return (colors[:, 0] << 16) | (colors[:, 1] << 8) | colors[:, 2]


def js_array(values, formatter):
    return ",".join(formatter(value) for value in values)


def build_html(
    title,
    points,
    colors,
    background,
    point_size,
    source_name,
    original_count,
    center,
    scale,
    original_points=None,
    original_indices=None,
):
    flat_points = points.reshape(-1)
    if original_points is None:
        original_points = np.asarray(points, dtype=np.float64) * scale + center
    if original_indices is None:
        original_indices = np.arange(len(points), dtype=np.uint32)
    flat_original_points = np.asarray(original_points).reshape(-1)
    original_indices = np.asarray(original_indices).reshape(-1)
    if len(original_points) != len(points) or len(original_indices) != len(points):
        raise ValueError("Point, original-coordinate, and original-index counts must match.")

    packed_colors = pack_colors(colors)
    safe_title = html.escape(title)
    safe_source = html.escape(source_name)
    point_values = js_array(flat_points, lambda value: f"{float(value):.7g}")
    original_point_values = js_array(flat_original_points, lambda value: f"{float(value):.9g}")
    original_index_values = js_array(original_indices, lambda value: str(int(value)))
    color_values = js_array(packed_colors, lambda value: str(int(value)))
    center_text = ", ".join(f"{float(value):.3f}" for value in center)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{safe_title}</title>
<style>
  html, body {{
    margin: 0;
    width: 100%;
    height: 100%;
    overflow: hidden;
    background: {background};
    color: #25282c;
    font-family: Arial, Helvetica, sans-serif;
  }}
  #viewer {{
    width: 100vw;
    height: 100vh;
    display: block;
    cursor: grab;
    touch-action: none;
  }}
  #viewer:active {{
    cursor: grabbing;
  }}
  #panel {{
    position: fixed;
    top: 14px;
    left: 14px;
    width: min(360px, calc(100vw - 28px));
    padding: 12px 14px;
    background: rgba(255, 255, 255, 0.82);
    border: 1px solid rgba(32, 36, 42, 0.14);
    border-radius: 7px;
    box-shadow: 0 8px 28px rgba(20, 24, 30, 0.10);
    backdrop-filter: blur(8px);
    font-size: 13px;
    line-height: 1.4;
  }}
  #panel h1 {{
    margin: 0 0 6px 0;
    font-size: 15px;
    font-weight: 700;
  }}
  .row {{
    display: flex;
    justify-content: space-between;
    gap: 14px;
  }}
  label {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 9px;
  }}
  input[type="range"] {{
    width: 140px;
  }}
  button {{
    margin-top: 9px;
    padding: 5px 9px;
    border: 1px solid rgba(32, 36, 42, 0.22);
    border-radius: 6px;
    background: #fff;
    color: #25282c;
  }}
  button:disabled {{
    color: #8b8f94;
    cursor: default;
  }}
  #selection {{
    margin-top: 10px;
    padding-top: 9px;
    border-top: 1px solid rgba(32, 36, 42, 0.14);
  }}
  #selection .row strong {{
    font-family: Consolas, "Courier New", monospace;
    font-weight: 600;
  }}
  .hint {{
    margin-top: 7px;
    color: #5f646b;
  }}
</style>
</head>
<body>
<canvas id="viewer"></canvas>
<div id="panel">
  <h1>{safe_title}</h1>
  <div>{safe_source}</div>
  <div class="row"><span>visible points</span><strong>{len(points):,}</strong></div>
  <div class="row"><span>original points</span><strong>{original_count:,}</strong></div>
  <div class="row"><span>center</span><strong>{center_text}</strong></div>
  <div class="row"><span>scale</span><strong>{scale:.3f}</strong></div>
  <label>point size <input id="size" type="range" min="1" max="7" step="0.2" value="{point_size}"></label>
  <label><input id="depth" type="checkbox" checked> depth sort</label>
  <button id="reset">reset view</button>
  <div id="selection">
    <div class="row"><span>original index</span><strong id="selected-index">none</strong></div>
    <div class="row"><span>x</span><strong id="selected-x">&mdash;</strong></div>
    <div class="row"><span>y</span><strong id="selected-y">&mdash;</strong></div>
    <div class="row"><span>z</span><strong id="selected-z">&mdash;</strong></div>
    <div class="row"><span>RGB</span><strong id="selected-rgb">&mdash;</strong></div>
    <button id="clear-selection" disabled>clear selection</button>
    <div class="hint">Click a point to inspect it; drag to rotate.</div>
  </div>
</div>
<script>
const points = new Float32Array([{point_values}]);
const originalPoints = new Float64Array([{original_point_values}]);
const originalIndices = new Uint32Array([{original_index_values}]);
const colors = new Uint32Array([{color_values}]);
const canvas = document.getElementById("viewer");
const ctx = canvas.getContext("2d");
const sizeControl = document.getElementById("size");
const depthControl = document.getElementById("depth");
const resetButton = document.getElementById("reset");
const clearSelectionButton = document.getElementById("clear-selection");
const selectedIndexText = document.getElementById("selected-index");
const selectedXText = document.getElementById("selected-x");
const selectedYText = document.getElementById("selected-y");
const selectedZText = document.getElementById("selected-z");
const selectedRgbText = document.getElementById("selected-rgb");
const n = colors.length;
let width = 1;
let height = 1;
let yaw = -0.65;
let pitch = 0.35;
let zoom = 1.45;
let dragging = false;
let activePointerId = null;
let pointerMoved = false;
let pointerDownX = 0;
let pointerDownY = 0;
let lastX = 0;
let lastY = 0;
let selectedPoint = -1;
const projected = new Float32Array(n * 3);
const order = Array.from({{length: n}}, (_, i) => i);
const dragThreshold = 4;
const pickTolerance = 5;

function resize() {{
  const dpr = window.devicePixelRatio || 1;
  width = window.innerWidth;
  height = window.innerHeight;
  canvas.width = Math.max(1, Math.floor(width * dpr));
  canvas.height = Math.max(1, Math.floor(height * dpr));
  canvas.style.width = width + "px";
  canvas.style.height = height + "px";
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}}

function project() {{
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const base = Math.min(width, height) * zoom;
  for (let i = 0; i < n; i++) {{
    const j = i * 3;
    const x = points[j];
    const y = points[j + 1];
    const z = points[j + 2];
    const x1 = cy * x + sy * z;
    const z1 = -sy * x + cy * z;
    const y2 = cp * y - sp * z1;
    const z2 = sp * y + cp * z1;
    projected[j] = width * 0.5 + x1 * base;
    projected[j + 1] = height * 0.5 - y2 * base;
    projected[j + 2] = z2;
  }}
}}

function draw() {{
  project();
  ctx.fillStyle = "{background}";
  ctx.fillRect(0, 0, width, height);
  if (depthControl.checked) {{
    order.sort((a, b) => projected[a * 3 + 2] - projected[b * 3 + 2]);
  }}
  const pointSize = Number(sizeControl.value);
  const half = pointSize * 0.5;
  for (let k = 0; k < n; k++) {{
    const i = depthControl.checked ? order[k] : k;
    const j = i * 3;
    const x = projected[j];
    const y = projected[j + 1];
    if (x < -8 || x > width + 8 || y < -8 || y > height + 8) continue;
    ctx.fillStyle = "#" + colors[i].toString(16).padStart(6, "0");
    ctx.fillRect(x - half, y - half, pointSize, pointSize);
  }}

  if (selectedPoint >= 0) {{
    const j = selectedPoint * 3;
    const x = projected[j];
    const y = projected[j + 1];
    const radius = Math.max(5, pointSize + 3);
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.lineWidth = 4;
    ctx.strokeStyle = "rgba(15, 18, 22, 0.9)";
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#ffd23f";
    ctx.stroke();
  }}
}}

function formatCoordinate(value) {{
  return Number(value).toPrecision(9);
}}

function updateSelectionPanel() {{
  if (selectedPoint < 0) {{
    selectedIndexText.textContent = "none";
    selectedXText.textContent = "\u2014";
    selectedYText.textContent = "\u2014";
    selectedZText.textContent = "\u2014";
    selectedRgbText.textContent = "\u2014";
    clearSelectionButton.disabled = true;
    return;
  }}

  const j = selectedPoint * 3;
  const color = colors[selectedPoint];
  const red = (color >>> 16) & 255;
  const green = (color >>> 8) & 255;
  const blue = color & 255;
  selectedIndexText.textContent = String(originalIndices[selectedPoint]);
  selectedXText.textContent = formatCoordinate(originalPoints[j]);
  selectedYText.textContent = formatCoordinate(originalPoints[j + 1]);
  selectedZText.textContent = formatCoordinate(originalPoints[j + 2]);
  selectedRgbText.textContent = `${{red}}, ${{green}}, ${{blue}}`;
  clearSelectionButton.disabled = false;
}}

function pickPoint(clientX, clientY) {{
  const rect = canvas.getBoundingClientRect();
  const pointerX = clientX - rect.left;
  const pointerY = clientY - rect.top;
  const halfSize = Number(sizeControl.value) * 0.5;
  const pickRadius = halfSize + pickTolerance;
  let bestIndex = -1;
  let bestDepth = -Infinity;
  let bestDistance2 = Infinity;
  let bestDirect = false;

  project();
  for (let i = 0; i < n; i++) {{
    const j = i * 3;
    const dx = pointerX - projected[j];
    const dy = pointerY - projected[j + 1];
    if (Math.abs(dx) > pickRadius || Math.abs(dy) > pickRadius) continue;

    const direct = Math.abs(dx) <= halfSize && Math.abs(dy) <= halfSize;
    const depth = depthControl.checked ? projected[j + 2] : i;
    const distance2 = dx * dx + dy * dy;
    const sameClassBetter = direct
      ? (depth > bestDepth || (depth === bestDepth && distance2 < bestDistance2))
      : (distance2 < bestDistance2 || (distance2 === bestDistance2 && depth > bestDepth));
    if (bestIndex < 0 || (direct && !bestDirect) || (direct === bestDirect && sameClassBetter)) {{
      bestIndex = i;
      bestDepth = depth;
      bestDistance2 = distance2;
      bestDirect = direct;
    }}
  }}

  if (bestIndex >= 0) {{
    selectedPoint = bestIndex;
    updateSelectionPanel();
    draw();
  }}
}}

canvas.addEventListener("pointerdown", event => {{
  if (activePointerId !== null || (event.pointerType === "mouse" && event.button !== 0)) return;
  activePointerId = event.pointerId;
  dragging = true;
  pointerMoved = false;
  pointerDownX = event.clientX;
  pointerDownY = event.clientY;
  lastX = event.clientX;
  lastY = event.clientY;
  canvas.setPointerCapture(event.pointerId);
}});

canvas.addEventListener("pointermove", event => {{
  if (!dragging || event.pointerId !== activePointerId) return;
  const totalDx = event.clientX - pointerDownX;
  const totalDy = event.clientY - pointerDownY;
  if (!pointerMoved && Math.hypot(totalDx, totalDy) <= dragThreshold) return;
  pointerMoved = true;
  const dx = event.clientX - lastX;
  const dy = event.clientY - lastY;
  lastX = event.clientX;
  lastY = event.clientY;
  yaw += dx * 0.008;
  pitch = Math.max(-1.45, Math.min(1.45, pitch + dy * 0.008));
  draw();
}});

function finishPointer(event, allowPick) {{
  if (event.pointerId !== activePointerId) return;
  const releaseDx = event.clientX - pointerDownX;
  const releaseDy = event.clientY - pointerDownY;
  const moved = pointerMoved || Math.hypot(releaseDx, releaseDy) > dragThreshold;
  const shouldPick = allowPick && !moved;
  dragging = false;
  activePointerId = null;
  pointerMoved = false;
  if (canvas.hasPointerCapture(event.pointerId)) {{
    canvas.releasePointerCapture(event.pointerId);
  }}
  if (shouldPick) pickPoint(event.clientX, event.clientY);
}}

canvas.addEventListener("pointerup", event => finishPointer(event, true));
canvas.addEventListener("pointercancel", event => finishPointer(event, false));
canvas.addEventListener("lostpointercapture", event => {{
  if (event.pointerId !== activePointerId) return;
  dragging = false;
  activePointerId = null;
  pointerMoved = false;
}});

canvas.addEventListener("wheel", event => {{
  event.preventDefault();
  zoom *= Math.exp(-event.deltaY * 0.001);
  zoom = Math.max(0.25, Math.min(8.0, zoom));
  draw();
}}, {{passive: false}});

sizeControl.addEventListener("input", draw);
depthControl.addEventListener("change", draw);
resetButton.addEventListener("click", () => {{
  yaw = -0.65;
  pitch = 0.35;
  zoom = 1.45;
  draw();
}});
clearSelectionButton.addEventListener("click", () => {{
  selectedPoint = -1;
  updateSelectionPanel();
  draw();
}});
window.addEventListener("resize", resize);
updateSelectionPanel();
resize();
</script>
</body>
</html>
"""


def main():
    args = parse_args()
    pointcloud_path = Path(args.pointcloud)
    out_path = Path(args.out)
    points, colors = load_colored_points(pointcloud_path)
    original_count = len(points)
    points, colors, original_indices = subset_points(
        points,
        colors,
        args.max_points,
        args.seed,
        return_indices=True,
    )
    original_points = points.copy()
    normalized, center, scale = normalize_points(points)
    content = build_html(
        title=args.title,
        points=normalized,
        colors=colors,
        background=args.background,
        point_size=args.point_size,
        source_name=str(pointcloud_path),
        original_count=original_count,
        center=center,
        scale=scale,
        original_points=original_points,
        original_indices=original_indices,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    print(f"Saved interactive viewer: {out_path}")
    print(f"Embedded points: {len(points)} / {original_count}")


if __name__ == "__main__":
    main()
