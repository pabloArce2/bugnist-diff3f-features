"""Quick point-splat PNG previews that need no GPU and no OpenGL."""

import math
from pathlib import Path

import numpy as np
from PIL import Image

BACKGROUND = 248


def rotate_points(points, elev_deg, azim_deg):
    """Turn around Z by the azimuth, then tilt around X by the elevation."""
    elev = math.radians(elev_deg)
    azim = math.radians(azim_deg)
    ca, sa = math.cos(azim), math.sin(azim)
    ce, se = math.cos(elev), math.sin(elev)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    x1 = ca * x - sa * y
    y1 = sa * x + ca * y
    return np.column_stack((x1, ce * y1 - se * z, se * y1 + ce * z))


def project_to_pixels(points, size, elev, azim, fill=0.82):
    """Pixel x, y and depth for points that land inside a size x size image."""
    points = rotate_points(points - points.mean(axis=0, keepdims=True), elev, azim)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    span = max(float((maxs[:2] - mins[:2]).max()), 1e-6)
    scale = size * fill / span

    xy = (points[:, :2] - (mins[:2] + maxs[:2]) / 2) * scale + size / 2
    x = np.rint(xy[:, 0]).astype(np.int32)
    y = np.rint(size - xy[:, 1]).astype(np.int32)
    valid = (x >= 0) & (x < size) & (y >= 0) & (y < size)
    return x[valid], y[valid], points[valid, 2], valid


def thicken(x, y, z, colors, size, radius):
    """Draw every point as a (2r+1)^2 square so sparse points stay visible. colors may be None."""
    offsets = [(dx, dy) for dx in range(-radius, radius + 1) for dy in range(-radius, radius + 1)]
    x = np.concatenate([x + dx for dx, _ in offsets])
    y = np.concatenate([y + dy for _, dy in offsets])
    z = np.tile(z, len(offsets))
    valid = (x >= 0) & (x < size) & (y >= 0) & (y < size)
    if colors is not None:
        colors = np.tile(colors, (len(offsets), 1))[valid]
    return x[valid], y[valid], z[valid], colors


def save_color_preview(points, colors, out_path, max_points=50000, size=1200, elev=24.0, azim=38.0, seed=42, dot_radius=0):
    """Save a PNG of coloured points, drawn back to front."""
    rng = np.random.default_rng(seed)
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    if len(points) > max_points:
        indices = rng.choice(len(points), size=max_points, replace=False)
        points = points[indices]
        colors = colors[indices]

    x, y, z, valid = project_to_pixels(points, size, elev, azim)
    colors = colors[valid]
    if dot_radius > 0:
        x, y, z, colors = thicken(x, y, z, colors, size, dot_radius)

    image = np.full((size, size, 3), BACKGROUND, dtype=np.uint8)
    order = np.argsort(z)
    image[y[order], x[order]] = colors[order]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(out_path)
