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
        colors = np.full((len(points), 3), 178, dtype=np.uint8)
    else:
        colors = np.asarray(colors, dtype=np.uint8)[:, :3]
    return points, colors


def subset_points(points, colors, max_points, seed):
    if max_points is None or len(points) <= max_points:
        return points, colors
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(points), size=max_points, replace=False)
    return points[indices], colors[indices]


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


def build_html(title, points, colors, background, point_size, source_name, original_count, center, scale):
    flat_points = points.reshape(-1)
    packed_colors = pack_colors(colors)
    safe_title = html.escape(title)
    safe_source = html.escape(source_name)
    point_values = js_array(flat_points, lambda value: f"{float(value):.7g}")
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
</div>
<script>
const points = new Float32Array([{point_values}]);
const colors = new Uint32Array([{color_values}]);
const canvas = document.getElementById("viewer");
const ctx = canvas.getContext("2d");
const sizeControl = document.getElementById("size");
const depthControl = document.getElementById("depth");
const resetButton = document.getElementById("reset");
const n = colors.length;
let width = 1;
let height = 1;
let yaw = -0.65;
let pitch = 0.35;
let zoom = 1.45;
let dragging = false;
let lastX = 0;
let lastY = 0;
const projected = new Float32Array(n * 3);
const order = Array.from({{length: n}}, (_, i) => i);

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
}}

canvas.addEventListener("pointerdown", event => {{
  dragging = true;
  lastX = event.clientX;
  lastY = event.clientY;
  canvas.setPointerCapture(event.pointerId);
}});

canvas.addEventListener("pointermove", event => {{
  if (!dragging) return;
  const dx = event.clientX - lastX;
  const dy = event.clientY - lastY;
  lastX = event.clientX;
  lastY = event.clientY;
  yaw += dx * 0.008;
  pitch = Math.max(-1.45, Math.min(1.45, pitch + dy * 0.008));
  draw();
}});

canvas.addEventListener("pointerup", () => {{
  dragging = false;
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
window.addEventListener("resize", resize);
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
    points, colors = subset_points(points, colors, args.max_points, args.seed)
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
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    print(f"Saved interactive viewer: {out_path}")
    print(f"Embedded points: {len(points)} / {original_count}")


if __name__ == "__main__":
    main()
