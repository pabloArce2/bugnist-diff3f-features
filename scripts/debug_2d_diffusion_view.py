import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloaders.mesh_container import MeshContainer
from camera_sampling import VIEW_SAMPLINGS, validate_view_count
from diff3f import arange_pixels
from diffusion import init_pipe, process_depth_map, rgb2normalmap
from dino import get_dino_features, init_dino
from render import batch_render as batch_render_mesh
from render_point_cloud import batch_render as batch_render_point_cloud
from utils import convert_mesh_container_to_torch_mesh


def parse_args():
    parser = argparse.ArgumentParser(
        description="Save the 2D render, ControlNet inputs, denoising frames, and feature maps for one Diff3F view."
    )
    parser.add_argument("--kind", choices=("mesh", "pointcloud"), required=True)
    parser.add_argument("--input", required=True, help="Input mesh or point-cloud file.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--outdir", default="debug/diff3f_2d_view")
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument(
        "--view-sampling",
        choices=VIEW_SAMPLINGS,
        default="grid",
        help="Camera sampling strategy. grid is the original Diff3F behavior.",
    )
    parser.add_argument("--view-index", type=int, default=0)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--tosca", action="store_true", help="Scale mesh as TOSCA mesh.")
    parser.add_argument("--no-normal-map", action="store_true", help="Use depth-only ControlNet.")
    parser.add_argument("--point-radius", type=float, default=0.012)
    parser.add_argument("--points-per-pixel", type=int, default=1)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--denoise-interval", type=int, default=5, help="Save a denoising frame every N steps.")
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--fit-sample", type=int, default=12000, help="Pixels used to fit PCA feature colors.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all-views", action="store_true", help="With --render-only, save every rendered camera view.")
    parser.add_argument("--render-only", action="store_true", help="Only save the selected 2D render view.")
    parser.add_argument("--skip-ai", action="store_true", help="Only save render/control images; do not run diffusion.")
    return parser.parse_args()


def image_from_float_rgb(rgb):
    rgb = np.asarray(rgb)
    rgb = np.clip(rgb, 0.0, 1.0)
    return Image.fromarray((rgb * 255).astype(np.uint8))


def save_tensor_rgb(path, tensor):
    array = tensor.detach().float().cpu().numpy()
    image_from_float_rgb(array).save(path)


def load_points(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        points = np.load(path)
    elif suffix in (".xyz", ".txt"):
        points = np.loadtxt(path, dtype=np.float32)
    else:
        loaded = trimesh.load(path, process=False, maintain_order=True)
        if not hasattr(loaded, "vertices"):
            raise ValueError(f"Could not read point positions from {path}.")
        points = np.asarray(loaded.vertices)

    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected Nx3 point coordinates in {path}, got shape {points.shape}.")
    return torch.from_numpy(points)


def render_debug_view(args, device, use_normal_map):
    if args.kind == "mesh":
        mesh_container = MeshContainer().load_from_file(args.input)
        mesh = convert_mesh_container_to_torch_mesh(mesh_container, device=device, is_tosca=args.tosca)
        rendered, normals, _, depth = batch_render_mesh(
            device,
            mesh,
            mesh.verts_list()[0],
            args.num_views,
            args.height,
            args.width,
            use_normal_map=use_normal_map,
            view_sampling=args.view_sampling,
        )
        point_indices = None
    else:
        points = load_points(args.input)
        rendered, normals, _, depth, point_indices = batch_render_point_cloud(
            device,
            points,
            args.num_views,
            args.height,
            args.width,
            use_normal_map=use_normal_map,
            return_images=True,
            return_point_indices=True,
            point_radius=args.point_radius,
            points_per_pixel=args.points_per_pixel,
            view_sampling=args.view_sampling,
        )

    rendered = rendered.cpu()
    depth = depth.cpu()
    if normals is not None:
        normals = normals.cpu()
    if point_indices is not None:
        point_indices = point_indices.cpu()

    if args.all_views:
        return {
            "renders": rendered[:, :, :, :3],
        }

    if args.view_index < 0 or args.view_index >= args.num_views:
        raise ValueError(f"--view-index must be between 0 and {args.num_views - 1}.")

    view = {
        "render": rendered[args.view_index, :, :, :3],
        "depth": depth[args.view_index, :, :, 0].unsqueeze(0),
        "normal": normals[args.view_index] if normals is not None else None,
        "point_indices": point_indices[args.view_index] if point_indices is not None else None,
    }
    return view


def pca_rgb_from_rows(rows, height, width, fit_sample, seed, normalize=True, mask=None):
    rows = torch.nan_to_num(rows.float())
    if normalize:
        rows = torch.nn.functional.normalize(rows, dim=1)

    fit_source = rows
    if mask is not None:
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if mask.any():
            fit_source = rows[torch.from_numpy(mask)]

    rng = np.random.default_rng(seed)
    if len(fit_source) > fit_sample:
        indices = torch.from_numpy(rng.choice(len(fit_source), size=fit_sample, replace=False)).long()
        fit_rows = fit_source[indices]
    else:
        fit_rows = fit_source

    mean = fit_rows.mean(dim=0, keepdim=True)
    fit_centered = fit_rows - mean
    _, _, basis = torch.pca_lowrank(fit_centered, q=3, center=False, niter=4)
    projected = ((rows - mean) @ basis[:, :3]).cpu().numpy()
    scale_source = projected[mask] if mask is not None and mask.any() else projected
    low, high = np.percentile(scale_source, (1.0, 99.0), axis=0)
    span = np.maximum(high - low, 1e-6)
    colors = np.clip((projected - low) / span, 0.0, 1.0)
    image = (colors.reshape(height, width, 3) * 255).astype(np.uint8)
    if mask is not None:
        masked = np.full((height * width, 3), 248, dtype=np.uint8)
        masked[mask] = image.reshape(-1, 3)[mask]
        image = masked.reshape(height, width, 3)
    return Image.fromarray(image)


def save_feature_maps(outdir, diffusion_feature_map, generated_image, device, height, width, fit_sample, seed, mask):
    grid = arange_pixels((height, width), invert_y_axis=False)[0].to(device).reshape(1, height, width, 2)
    dino_grid = grid.half()
    unet_grid = grid.float()
    dino_model = init_dino(device)

    diffusion_feature_map = diffusion_feature_map.detach().float()
    upsampled = torch.nn.Upsample(size=(height, width), mode="bilinear")(
        diffusion_feature_map.unsqueeze(0).to(device)
    )
    sampled_unet = torch.nn.functional.grid_sample(
        upsampled,
        unet_grid,
        align_corners=False,
    ).reshape(1, upsampled.shape[1], -1)
    sampled_unet = torch.nn.functional.normalize(sampled_unet, dim=1)
    unet_rows = sampled_unet[0].T.cpu()

    dino_features = get_dino_features(device, dino_model, generated_image, dino_grid)
    dino_rows = dino_features[0].T.cpu()

    combined_rows = torch.hstack((unet_rows * 0.5, dino_rows * 0.5))

    pca_rgb_from_rows(unet_rows, height, width, fit_sample, seed, mask=mask).save(outdir / "08_unet_feature_pca.png")
    pca_rgb_from_rows(dino_rows, height, width, fit_sample, seed, mask=mask).save(outdir / "09_dino_feature_pca.png")
    pca_rgb_from_rows(combined_rows, height, width, fit_sample, seed, mask=mask).save(
        outdir / "10_combined_diff3f_feature_pca.png"
    )


def decode_latents_to_image(pipe, latents):
    decoded = pipe.decode_latents(latents.detach())
    image = (decoded[0] * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(image)


def make_contact_sheet(outdir, image_paths):
    loaded = []
    for title, path in image_paths:
        if path.exists():
            image = Image.open(path).convert("RGB")
            image.thumbnail((256, 256))
            loaded.append((title, image.copy()))

    if not loaded:
        return

    cell_w = 292
    cell_h = 326
    cols = 3
    rows = math.ceil(len(loaded) / cols)
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (248, 248, 246))
    draw = ImageDraw.Draw(sheet)

    for idx, (title, image) in enumerate(loaded):
        col = idx % cols
        row = idx // cols
        x = col * cell_w + (cell_w - image.width) // 2
        y = row * cell_h + 36
        draw.text((col * cell_w + 18, row * cell_h + 12), title, fill=(28, 32, 38))
        sheet.paste(image, (x, y))

    sheet.save(outdir / "contact_sheet.png")


def save_all_render_views(outdir, renders):
    render_dir = outdir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    image_paths = []
    for view_index, render in enumerate(renders):
        path = render_dir / f"view_{view_index:03d}.png"
        image_from_float_rgb(render.numpy()).save(path)
        image_paths.append((f"view {view_index}", path))
    return image_paths


def write_readme(outdir, args, saved_denoising_steps):
    if args.all_views:
        file_lines = ["- `renders/view_*.png`: the plain 2D renders for every camera view."]
    else:
        file_lines = [
            "- `01_input_render.png`: the plain 2D render given to image-to-image Stable Diffusion.",
        ]
    if not args.render_only:
        file_lines.extend(
            [
                "- `02_depth_control.png`: the depth ControlNet conditioning image.",
                "- `03_normal_control.png`: the normal ControlNet conditioning image, if enabled.",
                "- `04_visible_mask.png`: white pixels are geometry pixels; black pixels are background.",
            ]
        )
    if not args.render_only and not args.skip_ai:
        file_lines.extend(
            [
                "- `05_denoise_step_*.png`: decoded latent snapshots during Stable Diffusion denoising.",
                "- `06_final_generated.png`: the final image produced by the 2D AI.",
                "- `07_ai_change_map.png`: amplified pixel difference between the input render and final generated image.",
                "- `08_unet_feature_pca.png`: PCA visualization of the 1280-D diffusion UNet feature map.",
                "- `09_dino_feature_pca.png`: PCA visualization of the 768-D DINOv2 feature map.",
                "- `10_combined_diff3f_feature_pca.png`: PCA visualization of the full 2048-D pixel descriptor.",
            ]
        )
    file_lines.append("- `contact_sheet.png`: a compact visual summary.")

    lines = [
        "# Diff3F 2D View Debug",
        "",
        "This folder captures one 2D view from the Diff3F pipeline.",
        "",
        "## Files",
        "",
        *file_lines,
        "",
        "## Run",
        "",
        "```text",
        json.dumps(vars(args), indent=2),
        "```",
        "",
        f"Saved denoising steps: {saved_denoising_steps}",
        "",
        "The PCA feature images are only visual shadows of high-dimensional features. "
        "They help us inspect structure, but matching still happens in the full descriptor space.",
    ]
    (outdir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    validate_view_count(args.num_views, args.view_sampling)
    if args.all_views and not args.render_only:
        raise ValueError("--all-views is only supported together with --render-only.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    use_normal_map = False if args.render_only else not args.no_normal_map

    print(f"Using device: {device}")
    print(f"View sampling: {args.view_sampling} ({args.num_views} views)")
    if args.all_views:
        print(f"Rendering all {args.num_views} {args.kind} views")
    else:
        print(f"Rendering {args.kind} view {args.view_index}/{args.num_views - 1}")
    view = render_debug_view(args, device, use_normal_map)

    if args.all_views:
        image_paths = save_all_render_views(outdir, view["renders"])
        make_contact_sheet(outdir, image_paths)
        write_readme(outdir, args, [])
        print(f"Saved {len(image_paths)} render-only views to {outdir}")
        return

    input_render = image_from_float_rgb(view["render"].numpy())
    input_render.save(outdir / "01_input_render.png")

    if args.render_only:
        make_contact_sheet(outdir, [("input render", outdir / "01_input_render.png")])
        write_readme(outdir, args, [])
        print(f"Saved render-only debug image to {outdir}")
        return

    depth_image = process_depth_map(view["depth"].clone())
    depth_image.save(outdir / "02_depth_control.png")

    normal_image = None
    if use_normal_map and view["normal"] is not None:
        normal_image = Image.fromarray(rgb2normalmap(view["normal"]))
        normal_image.save(outdir / "03_normal_control.png")

    visible_mask = (view["depth"][0] != -1).numpy()
    Image.fromarray(visible_mask.astype(np.uint8) * 255).save(outdir / "04_visible_mask.png")

    if args.skip_ai:
        make_contact_sheet(
            outdir,
            [
                ("input render", outdir / "01_input_render.png"),
                ("depth control", outdir / "02_depth_control.png"),
                ("normal control", outdir / "03_normal_control.png"),
                ("visible mask", outdir / "04_visible_mask.png"),
            ],
        )
        write_readme(outdir, args, [])
        print(f"Saved render/control debug images to {outdir}")
        return

    pipe = init_pipe(device, use_normal_map=use_normal_map)
    saved_denoising_steps = []

    def callback(step, timestep, latents):
        should_save = step == 0 or step == args.num_inference_steps - 1 or step % args.denoise_interval == 0
        if not should_save:
            return
        path = outdir / f"05_denoise_step_{step:03d}_t{int(timestep):04d}.png"
        decode_latents_to_image(pipe, latents).save(path)
        saved_denoising_steps.append(int(step))

    control_image = [depth_image, normal_image] if normal_image is not None else depth_image
    positive_prompt = f"{args.prompt},best quality,highly detailed,photorealistic,photo"
    negative_prompt = "lowres,low quality,monochrome,watermark"
    output = pipe(
        positive_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=args.num_inference_steps,
        image=input_render,
        control_image=control_image,
        guidance_scale=args.guidance_scale,
        eta=args.eta,
        output_type="pil",
        return_image=True,
        callback=callback,
        callback_steps=1,
    ).images

    diffusion_feature_map = output[0]
    generated_image = output[1][0]
    generated_image.save(outdir / "06_final_generated.png")
    input_array = np.asarray(input_render.convert("RGB"), dtype=np.float32)
    generated_array = np.asarray(generated_image.resize(input_render.size).convert("RGB"), dtype=np.float32)
    change = np.clip(np.abs(generated_array - input_array) * 2.5, 0, 255).astype(np.uint8)
    Image.fromarray(change).save(outdir / "07_ai_change_map.png")

    save_feature_maps(
        outdir,
        diffusion_feature_map,
        generated_image,
        device,
        args.height,
        args.width,
        args.fit_sample,
        args.seed,
        visible_mask,
    )

    contact_items = [
        ("input render", outdir / "01_input_render.png"),
        ("depth control", outdir / "02_depth_control.png"),
        ("normal control", outdir / "03_normal_control.png"),
        ("visible mask", outdir / "04_visible_mask.png"),
        ("final generated", outdir / "06_final_generated.png"),
        ("AI change map", outdir / "07_ai_change_map.png"),
        ("unet feature PCA", outdir / "08_unet_feature_pca.png"),
        ("dino feature PCA", outdir / "09_dino_feature_pca.png"),
        ("combined feature PCA", outdir / "10_combined_diff3f_feature_pca.png"),
    ]
    denoise_paths = sorted(outdir.glob("05_denoise_step_*.png"))
    if denoise_paths:
        contact_items.insert(4, ("mid denoise", denoise_paths[len(denoise_paths) // 2]))
    make_contact_sheet(outdir, contact_items)
    write_readme(outdir, args, saved_denoising_steps)

    print(f"Saved debug images to {outdir}")
    print(f"Saved denoising steps: {saved_denoising_steps}")


if __name__ == "__main__":
    main()
