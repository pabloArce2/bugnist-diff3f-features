import torch
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    look_at_view_transform,
)
from pytorch3d.structures import Pointclouds

from bugnist_tools.camera_sampling import get_view_angles


def depth_to_render_images(raw_depth):
    front_depth = raw_depth[..., 0]
    batch_size, height, width = front_depth.shape
    images = torch.ones((batch_size, height, width, 3), device=raw_depth.device, dtype=torch.float32)

    for idx in range(batch_size):
        depth = front_depth[idx]
        valid = depth != -1
        if not torch.any(valid):
            continue

        valid_depth = depth[valid]
        near = valid_depth.min()
        far = valid_depth.max()
        span = torch.clamp(far - near, min=1e-6)
        normalized = (depth - near) / span
        shade = 0.88 - 0.48 * normalized
        images[idx][valid] = shade[valid].unsqueeze(-1).repeat(1, 3)

    return images


def depth_to_normal_maps(raw_depth):
    front_depth = raw_depth[..., 0]
    batch_size, height, width = front_depth.shape
    normal_maps = torch.zeros((batch_size, height, width, 1, 3), device=raw_depth.device, dtype=torch.float32)

    for idx in range(batch_size):
        depth = front_depth[idx]
        valid = depth != -1
        if not torch.any(valid):
            continue

        valid_depth = depth[valid]
        near = valid_depth.min()
        far = valid_depth.max()
        span = torch.clamp(far - near, min=1e-6)
        filled_depth = depth.clone()
        filled_depth[~valid] = far
        filled_depth = (filled_depth - near) / span

        dzdx = torch.zeros_like(filled_depth)
        dzdy = torch.zeros_like(filled_depth)
        dzdx[:, 1:-1] = (filled_depth[:, 2:] - filled_depth[:, :-2]) * 0.5
        dzdx[:, 0] = filled_depth[:, 1] - filled_depth[:, 0]
        dzdx[:, -1] = filled_depth[:, -1] - filled_depth[:, -2]
        dzdy[1:-1, :] = (filled_depth[2:, :] - filled_depth[:-2, :]) * 0.5
        dzdy[0, :] = filled_depth[1, :] - filled_depth[0, :]
        dzdy[-1, :] = filled_depth[-1, :] - filled_depth[-2, :]

        normals = torch.stack((-dzdx, -dzdy, torch.ones_like(filled_depth)), dim=-1)
        normals = torch.nn.functional.normalize(normals, dim=-1)
        normals[~valid] = 0
        normal_maps[idx, :, :, 0, :] = normals

    return normal_maps


def get_colored_depth_maps(raw_depths, H, W):
    images = depth_to_render_images(raw_depths).detach().cpu().numpy()
    return [(image * 255).astype("uint8") for image in images.reshape(-1, H, W, 3)]


@torch.no_grad()
def run_rendering(
    device,
    points,
    num_views,
    H,
    W,
    add_angle_azi=0,
    add_angle_ele=0,
    use_normal_map=False,
    return_images=False,
    return_point_indices=False,
    point_radius=0.01,
    points_per_pixel=1,
    view_sampling="grid",
):
    points = points.to(device=device, dtype=torch.float32)
    features = torch.ones_like(points, device=device, dtype=torch.float32) * 0.8
    pointclouds = Pointclouds(points=[points], features=[features])
    bbox = pointclouds.get_bounding_boxes()
    bbox_min = bbox.min(dim=-1).values[0]
    bbox_max = bbox.max(dim=-1).values[0]
    bb_diff = bbox_max - bbox_min
    bbox_center = (bbox_min + bbox_max) / 2.0
    scaling_factor = 0.65
    distance = torch.sqrt((bb_diff * bb_diff).sum())
    distance *= scaling_factor
    azimuth, elevation = get_view_angles(
        num_views,
        device,
        view_sampling=view_sampling,
        add_angle_azi=add_angle_azi,
        add_angle_ele=add_angle_ele,
    )
    bbox_center = bbox_center.unsqueeze(0)
    rotation, translation = look_at_view_transform(
        dist=distance, azim=azimuth, elev=elevation, device=device, at=bbox_center
    )
    camera = PerspectiveCameras(R=rotation, T=translation, device=device)

    rasterization_settings = PointsRasterizationSettings(
        image_size=(H, W),
        radius=point_radius,
        points_per_pixel=points_per_pixel,
        bin_size=0,
        max_points_per_bin=0,
    )
    rasterizer = PointsRasterizer(cameras=camera, raster_settings=rasterization_settings)
    batch_points = pointclouds.extend(num_views)
    fragments = rasterizer(batch_points)
    raw_depth = fragments.zbuf
    point_indices = fragments.idx
    point_offsets = (torch.arange(num_views, device=device, dtype=point_indices.dtype) * len(points)).view(
        num_views, 1, 1, 1
    )
    point_indices = torch.where(point_indices != -1, point_indices - point_offsets, point_indices)

    if return_images and return_point_indices:
        rendered_images = depth_to_render_images(raw_depth)
        normal_images = depth_to_normal_maps(raw_depth) if use_normal_map else None
        return rendered_images, normal_images, camera, raw_depth, point_indices

    if return_images:
        list_depth_images_np = get_colored_depth_maps(raw_depth, H, W)
        return None, None, camera, raw_depth, list_depth_images_np

    if return_point_indices:
        normal_images = depth_to_normal_maps(raw_depth) if use_normal_map else None
        return None, normal_images, camera, raw_depth, point_indices

    return None, None, camera, raw_depth


def batch_render(
    device,
    points,
    num_views,
    H,
    W,
    use_normal_map=False,
    return_images=False,
    return_point_indices=False,
    point_radius=0.01,
    points_per_pixel=1,
    view_sampling="grid",
):
    trials = 0
    add_angle_azi = 0
    add_angle_ele = 0
    while trials < 5:
        try:
            return run_rendering(
                device,
                points,
                num_views,
                H,
                W,
                add_angle_azi=add_angle_azi,
                add_angle_ele=add_angle_ele,
                use_normal_map=use_normal_map,
                return_images=return_images,
                return_point_indices=return_point_indices,
                point_radius=point_radius,
                points_per_pixel=points_per_pixel,
                view_sampling=view_sampling,
            )
        except torch.linalg.LinAlgError:
            trials += 1
            print("lin alg exception at rendering, retrying ", trials)
            add_angle_azi = torch.randn(1, device=device)
            add_angle_ele = torch.randn(1, device=device)
            continue

    raise RuntimeError("Point cloud rendering failed after 5 attempts.")
