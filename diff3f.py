import torch
from PIL import Image
from torchvision.utils import make_grid
import numpy as np
from diffusion import add_texture_to_render
from dino import get_dino_features
from render import batch_render
from render_point_cloud import batch_render as batch_render_point_cloud
from pytorch3d.ops import ball_query
from tqdm import tqdm
from time import time
import random


FEATURE_DIMS = 1280+768 # diffusion unet + dino
VERTEX_GPU_LIMIT = 35000
FILL_DISTANCE_TARGET = 20_000_000


def arange_pixels(
    resolution=(128, 128),
    batch_size=1,
    subsample_to=None,
    invert_y_axis=False,
    margin=0,
    corner_aligned=True,
    jitter=None,
):
    h, w = resolution
    n_points = resolution[0] * resolution[1]
    uh = 1 if corner_aligned else 1 - (1 / h)
    uw = 1 if corner_aligned else 1 - (1 / w)
    if margin > 0:
        uh = uh + (2 / h) * margin
        uw = uw + (2 / w) * margin
        w, h = w + margin * 2, h + margin * 2

    x, y = torch.linspace(-uw, uw, w), torch.linspace(-uh, uh, h)
    if jitter is not None:
        dx = (torch.ones_like(x).uniform_() - 0.5) * 2 / w * jitter
        dy = (torch.ones_like(y).uniform_() - 0.5) * 2 / h * jitter
        x, y = x + dx, y + dy
    x, y = torch.meshgrid(x, y, indexing="ij")
    pixel_scaled = (
        torch.stack([x, y], -1)
        .permute(1, 0, 2)
        .reshape(1, -1, 2)
        .repeat(batch_size, 1, 1)
    )

    if subsample_to is not None and subsample_to > 0 and subsample_to < n_points:
        idx = np.random.choice(
            pixel_scaled.shape[1], size=(subsample_to,), replace=False
        )
        pixel_scaled = pixel_scaled[:, idx]

    if invert_y_axis:
        pixel_scaled[..., -1] *= -1.0

    return pixel_scaled


def copy_missing_features_from_nearest(
    ft_per_vertex,
    ft_per_vertex_count,
    mesh_vertices,
    distance_target=FILL_DISTANCE_TARGET,
    label="vertices",
):
    filled_indices = ft_per_vertex_count[:, 0] != 0
    missing_indices = ft_per_vertex_count[:, 0] == 0
    missing_count = int(missing_indices.sum())
    filled_count = int(filled_indices.sum())

    if missing_count == 0:
        return
    if filled_count == 0:
        raise ValueError(f"No {label} received features; try more views, a larger tolerance, or a smaller shape.")

    vertices_cpu = mesh_vertices.detach().float().cpu()
    filled_vertices = vertices_cpu[filled_indices]
    missing_vertices = vertices_cpu[missing_indices]
    filled_features = ft_per_vertex[filled_indices]
    missing_positions = torch.where(missing_indices)[0]
    chunk_size = max(1, distance_target // filled_count)

    print(
        f"Filling {missing_count} missing {label} from {filled_count} feature {label} "
        f"in chunks of {chunk_size}"
    )
    for start in tqdm(range(0, missing_count, chunk_size), desc="Filling missing features"):
        stop = min(start + chunk_size, missing_count)
        distances = torch.cdist(missing_vertices[start:stop], filled_vertices, p=2)
        closest_vertex_indices = torch.argmin(distances, dim=1)
        ft_per_vertex[missing_positions[start:stop], :] = filled_features[closest_vertex_indices]


def get_features_per_vertex(
    device,
    pipe,
    dino_model,
    mesh,
    prompt,
    num_views=100,
    H=512,
    W=512,
    tolerance=0.01,
    use_latent=False,
    use_normal_map=True,
    num_images_per_prompt=1,
    mesh_vertices=None,
    return_image=True,
    bq=True,
    prompts_list=None,
    view_sampling="grid",
):
    t1 = time()
    if mesh_vertices is None:
        mesh_vertices = mesh.verts_list()[0]
    if len(mesh_vertices) > VERTEX_GPU_LIMIT:
        samples = random.sample(range(len(mesh_vertices)), 10000)
        maximal_distance = torch.cdist(mesh_vertices[samples], mesh_vertices[samples]).max()
    else:
        maximal_distance = torch.cdist(mesh_vertices, mesh_vertices).max()  # .cpu()
    ball_drop_radius = maximal_distance * tolerance
    batched_renderings, normal_batched_renderings, camera, depth = batch_render(
        device,
        mesh,
        mesh.verts_list()[0],
        num_views,
        H,
        W,
        use_normal_map,
        view_sampling=view_sampling,
    )
    print("Rendering complete")
    if use_normal_map:
        normal_batched_renderings = normal_batched_renderings.cpu()
    batched_renderings = batched_renderings.cpu()
    pixel_coords = arange_pixels((H, W), invert_y_axis=True)[0]
    pixel_coords[:, 0] = torch.flip(pixel_coords[:, 0], dims=[0])
    grid = arange_pixels((H, W), invert_y_axis=False)[0].to(device).reshape(1, H, W, 2).half()
    camera = camera.cpu()
    normal_map_input = None
    depth = depth.cpu()
    torch.cuda.empty_cache()
    ft_per_vertex = torch.zeros((len(mesh_vertices), FEATURE_DIMS)).half()  # .to(device)
    ft_per_vertex_count = torch.zeros((len(mesh_vertices), 1)).half()  # .to(device)
    for idx in tqdm(range(len(batched_renderings))):
        dp = depth[idx].flatten().unsqueeze(1)
        xy_depth = torch.cat((pixel_coords, dp), dim=1)
        indices = xy_depth[:, 2] != -1
        xy_depth = xy_depth[indices]
        world_coords = (
            camera[idx].unproject_points(
                xy_depth, world_coordinates=True, from_ndc=True
            )  # .cpu()
        ).to(device)
        diffusion_input_img = (
            batched_renderings[idx, :, :, :3].cpu().numpy() * 255
        ).astype(np.uint8)
        if use_normal_map:
            normal_map_input = normal_batched_renderings[idx]
        depth_map = depth[idx, :, :, 0].unsqueeze(0).to(device)
        if prompts_list is not None:
            prompt = random.choice(prompts_list)
        diffusion_output = add_texture_to_render(
            pipe,
            diffusion_input_img,
            depth_map,
            prompt,
            normal_map_input=normal_map_input,
            use_latent=use_latent,
            num_images_per_prompt=num_images_per_prompt,
            return_image=return_image
        )
        aligned_dino_features = get_dino_features(device, dino_model, diffusion_output[1][0], grid)
        aligned_features = None
        with torch.no_grad():
            ft = torch.nn.Upsample(size=(H,W), mode="bilinear")(diffusion_output[0].unsqueeze(0)).to(device)
            ft_dim = ft.size(1)
            aligned_features = torch.nn.functional.grid_sample(
                ft, grid, align_corners=False
            ).reshape(1, ft_dim, -1)
            aligned_features = torch.nn.functional.normalize(aligned_features, dim=1)
        # this is feature per pixel in the grid
        aligned_features = torch.hstack([aligned_features*0.5, aligned_dino_features*0.5])
        features_per_pixel = aligned_features[0, :, indices].cpu()
        # map pixel to vertex on mesh
        if bq:
            queried_indices = (
                ball_query(
                    world_coords.unsqueeze(0),
                    mesh_vertices.unsqueeze(0),
                    K=100,
                    radius=ball_drop_radius,
                    return_nn=False,
                )
                .idx[0]
                .cpu()
            )
            mask = queried_indices != -1
            repeat = mask.sum(dim=1)
            ft_per_vertex_count[queried_indices[mask]] += 1
            ft_per_vertex[queried_indices[mask]] += features_per_pixel.repeat_interleave(
                repeat, dim=1
            ).T
        else:
            distances = torch.cdist(
            world_coords, mesh_vertices, p=2
            )
            closest_vertex_indices = torch.argmin(distances, dim=1).cpu()
            ft_per_vertex[closest_vertex_indices] += features_per_pixel.T
            ft_per_vertex_count[closest_vertex_indices] += 1

    idxs = (ft_per_vertex_count != 0)[:, 0]
    ft_per_vertex[idxs, :] = ft_per_vertex[idxs, :] / ft_per_vertex_count[idxs, :]
    missing_features = len(ft_per_vertex_count[ft_per_vertex_count == 0])
    print("Number of missing features: ", missing_features)
    print("Copied features from nearest vertices")

    if missing_features > 0:
        copy_missing_features_from_nearest(ft_per_vertex, ft_per_vertex_count, mesh_vertices)
    t2 = time() - t1
    t2 = t2 / 60
    print("Time taken in mins: ", t2)
    return ft_per_vertex


def get_features_per_point_cloud(
    device,
    pipe,
    dino_model,
    points,
    prompt,
    num_views=100,
    H=512,
    W=512,
    point_radius=0.01,
    points_per_pixel=1,
    use_latent=False,
    use_normal_map=True,
    num_images_per_prompt=1,
    return_image=True,
    prompts_list=None,
    view_sampling="grid",
):
    t1 = time()
    points = points.to(device=device, dtype=torch.float32)
    if len(points) == 0:
        raise ValueError("Cannot compute features for an empty point cloud.")

    batched_renderings, normal_batched_renderings, _, depth, point_indices = batch_render_point_cloud(
        device,
        points,
        num_views,
        H,
        W,
        use_normal_map=use_normal_map,
        return_images=True,
        return_point_indices=True,
        point_radius=point_radius,
        points_per_pixel=points_per_pixel,
        view_sampling=view_sampling,
    )
    print("Point cloud rendering complete")

    batched_renderings = batched_renderings.cpu()
    if use_normal_map and normal_batched_renderings is not None:
        normal_batched_renderings = normal_batched_renderings.cpu()
    depth = depth.cpu()
    point_indices = point_indices.cpu()

    grid = arange_pixels((H, W), invert_y_axis=False)[0].to(device).reshape(1, H, W, 2).half()
    torch.cuda.empty_cache()

    ft_per_point = torch.zeros((len(points), FEATURE_DIMS)).half()
    ft_per_point_count = torch.zeros((len(points), 1)).half()

    for idx in tqdm(range(len(batched_renderings))):
        view_point_indices = point_indices[idx].reshape(H * W, -1)
        valid_slots = view_point_indices != -1
        repeat = valid_slots.sum(dim=1)
        if int(repeat.sum()) == 0:
            continue

        visible_point_indices = view_point_indices[valid_slots].long()
        pixel_ids = torch.arange(H * W).repeat_interleave(repeat)

        diffusion_input_img = (
            batched_renderings[idx, :, :, :3].cpu().numpy() * 255
        ).astype(np.uint8)
        normal_map_input = None
        if use_normal_map and normal_batched_renderings is not None:
            normal_map_input = normal_batched_renderings[idx]
        depth_map = depth[idx, :, :, 0].unsqueeze(0).to(device)
        if prompts_list is not None:
            prompt = random.choice(prompts_list)

        diffusion_output = add_texture_to_render(
            pipe,
            diffusion_input_img,
            depth_map,
            prompt,
            normal_map_input=normal_map_input,
            use_latent=use_latent,
            num_images_per_prompt=num_images_per_prompt,
            return_image=return_image,
        )
        aligned_dino_features = get_dino_features(device, dino_model, diffusion_output[1][0], grid)
        with torch.no_grad():
            ft = torch.nn.Upsample(size=(H, W), mode="bilinear")(diffusion_output[0].unsqueeze(0)).to(device)
            ft_dim = ft.size(1)
            aligned_features = torch.nn.functional.grid_sample(
                ft, grid, align_corners=False
            ).reshape(1, ft_dim, -1)
            aligned_features = torch.nn.functional.normalize(aligned_features, dim=1)

        aligned_features = torch.hstack([aligned_features * 0.5, aligned_dino_features * 0.5])
        features_per_slot = aligned_features[0, :, pixel_ids.to(device)].cpu().T
        ft_per_point.index_add_(0, visible_point_indices, features_per_slot)
        ft_per_point_count.index_add_(
            0,
            visible_point_indices,
            torch.ones((len(visible_point_indices), 1), dtype=ft_per_point_count.dtype),
        )

    idxs = (ft_per_point_count != 0)[:, 0]
    ft_per_point[idxs, :] = ft_per_point[idxs, :] / ft_per_point_count[idxs, :]
    missing_features = len(ft_per_point_count[ft_per_point_count == 0])
    print("Number of missing point features: ", missing_features)
    print("Copied features from nearest points")

    if missing_features > 0:
        copy_missing_features_from_nearest(ft_per_point, ft_per_point_count, points, label="points")

    t2 = time() - t1
    t2 = t2 / 60
    print("Time taken in mins: ", t2)
    return ft_per_point
