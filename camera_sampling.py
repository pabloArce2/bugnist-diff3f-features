import math

import torch


VIEW_SAMPLINGS = ("grid", "fibonacci", "insect")


def validate_view_count(num_views, view_sampling):
    if num_views < 1:
        raise ValueError("--num-views must be >= 1.")
    if view_sampling not in VIEW_SAMPLINGS:
        raise ValueError(f"--view-sampling must be one of: {', '.join(VIEW_SAMPLINGS)}.")
    if view_sampling == "grid":
        view_grid = math.isqrt(num_views)
        if view_grid * view_grid != num_views:
            raise ValueError(
                "--num-views must be a perfect square when --view-sampling grid is used, "
                "e.g. 4, 9, 16, 25, or 100. Use --view-sampling fibonacci or insect for arbitrary counts."
            )


def _grid_angles(num_views, device):
    steps = math.isqrt(num_views)
    end = 360.0 - 360.0 / steps
    elevation = torch.linspace(start=0.0, end=end, steps=steps, device=device).repeat(steps)
    azimuth = torch.linspace(start=0.0, end=end, steps=steps, device=device)
    azimuth = torch.repeat_interleave(azimuth, steps)
    return azimuth, elevation


def _fibonacci_angles(num_views, device):
    indices = torch.arange(num_views, device=device, dtype=torch.float32) + 0.5
    z = 1.0 - 2.0 * indices / float(num_views)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    theta = indices * golden_angle
    elevation = torch.rad2deg(torch.asin(torch.clamp(z, -1.0, 1.0)))
    azimuth = torch.rad2deg(torch.remainder(theta, 2.0 * math.pi))
    return azimuth, elevation


def _weighted_counts(num_views, weights, device):
    weights = torch.tensor(weights, device=device, dtype=torch.float32)
    raw = weights / weights.sum() * float(num_views)
    counts = torch.floor(raw).to(torch.int64)
    remaining = int(num_views - int(counts.sum()))
    if remaining > 0:
        order = torch.argsort(raw - counts.float(), descending=True)
        counts[order[:remaining]] += 1
    return counts.tolist()


def _insect_angles(num_views, device):
    # Favor side and oblique views: useful for legs, antennae, thorax, and abdomen.
    elevations = torch.tensor((0.0, 25.0, -25.0, 50.0, -50.0), device=device)
    counts = _weighted_counts(num_views, (0.40, 0.25, 0.25, 0.05, 0.05), device)

    azimuths = []
    elevs = []
    for ring_index, (count, elevation) in enumerate(zip(counts, elevations)):
        if count == 0:
            continue
        step = 360.0 / float(count)
        offset = (ring_index * 0.37 * step) % 360.0
        azimuth = torch.arange(count, device=device, dtype=torch.float32) * step + offset
        azimuths.append(torch.remainder(azimuth, 360.0))
        elevs.append(torch.full((count,), float(elevation.item()), device=device))

    return torch.cat(azimuths), torch.cat(elevs)


def get_view_angles(num_views, device, view_sampling="grid", add_angle_azi=0, add_angle_ele=0):
    validate_view_count(num_views, view_sampling)
    if view_sampling == "grid":
        azimuth, elevation = _grid_angles(num_views, device)
    elif view_sampling == "fibonacci":
        azimuth, elevation = _fibonacci_angles(num_views, device)
    else:
        azimuth, elevation = _insect_angles(num_views, device)

    return azimuth + add_angle_azi, elevation + add_angle_ele
