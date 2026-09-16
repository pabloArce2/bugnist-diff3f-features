import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from camera_sampling import VIEW_SAMPLINGS, validate_view_count
from descriptor_debug_capture import DescriptorDebugWriter, create_debug_run_directory, parse_debug_views
from diff3f import get_features_per_point_cloud
from diffusion import init_pipe
from dino import init_dino


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Diff3F features for one or more point clouds.")
    parser.add_argument("--pointcloud", nargs="+", required=True, help="Point cloud files to process: .ply, .npy, .xyz.")
    parser.add_argument(
        "--prompt",
        nargs="+",
        required=True,
        help="Prompt per point cloud, or one prompt reused for all point clouds.",
    )
    parser.add_argument("--outdir", default="output/pointcloud_features", help="Directory for .pt outputs.")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda:0 or cpu.")
    parser.add_argument("--num-views", type=int, default=100)
    parser.add_argument(
        "--view-sampling",
        choices=VIEW_SAMPLINGS,
        default="grid",
        help="Camera sampling strategy. grid is the original Diff3F behavior.",
    )
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--point-radius", type=float, default=0.01, help="PyTorch3D point radius in NDC units.")
    parser.add_argument("--points-per-pixel", type=int, default=1)
    parser.add_argument(
        "--debug-outdir",
        default=None,
        help="Capture the exact per-view images used by this descriptor under a new run folder.",
    )
    parser.add_argument(
        "--debug-views",
        default="all",
        help="Views to capture: 'all', indices such as '0 4 8', or ranges such as '0-3'.",
    )
    parser.add_argument("--no-normal-map", action="store_true", help="Use only depth ControlNet.")
    parser.add_argument("--skip-existing", action="store_true", help="Do not recompute existing outputs.")
    return parser.parse_args()


def prompts_for_pointclouds(prompts, pointcloud_count):
    if len(prompts) == 1:
        return prompts * pointcloud_count
    if len(prompts) != pointcloud_count:
        raise ValueError("--prompt must have either one value or the same number of values as --pointcloud.")
    return prompts


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
        points = np.asarray(loaded.vertices, dtype=np.float32)

    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected Nx3 point coordinates in {path}, got shape {points.shape}.")
    if len(points) == 0:
        raise ValueError(f"{path} contains no points.")
    return torch.from_numpy(points)


def main():
    args = parse_args()
    validate_view_count(args.num_views, args.view_sampling)
    if args.debug_outdir:
        parse_debug_views(args.debug_views, args.num_views)

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_device(device)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    prompts = prompts_for_pointclouds(args.prompt, len(args.pointcloud))
    use_normal_map = not args.no_normal_map

    print(f"Using device: {device}")
    print(f"View sampling: {args.view_sampling} ({args.num_views} views)")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    pipe = init_pipe(device, use_normal_map=use_normal_map)
    dino_model = init_dino(device)
    debug_run_dir = None
    if args.debug_outdir:
        debug_run_dir = create_debug_run_directory(args.debug_outdir, "pointcloud")
        print(f"Capturing exact descriptor-run images in {debug_run_dir}")
        if args.skip_existing:
            print("Ignoring --skip-existing because image capture requires a fresh descriptor run.")

    for pointcloud_index, (pointcloud_path, prompt) in enumerate(zip(args.pointcloud, prompts)):
        pointcloud_path = Path(pointcloud_path)
        save_path = outdir / f"{pointcloud_path.stem}_diff3f.pt"
        if args.skip_existing and not args.debug_outdir and save_path.exists():
            print(f"Skipping existing output: {save_path}")
            continue

        print(f"Processing {pointcloud_path} with prompt {prompt!r}")
        points = load_points(pointcloud_path)
        debug_writer = None
        if debug_run_dir is not None:
            debug_writer = DescriptorDebugWriter(
                run_dir=debug_run_dir,
                asset_index=pointcloud_index,
                kind="pointcloud",
                input_path=pointcloud_path,
                descriptor_path=save_path,
                prompt=prompt,
                num_views=args.num_views,
                view_sampling=args.view_sampling,
                image_size=(args.height, args.width),
                selected_views=args.debug_views,
                mode="full",
                extra_settings={
                    "normalMap": use_normal_map,
                    "pointRadius": args.point_radius,
                    "pointsPerPixel": args.points_per_pixel,
                },
            )
        try:
            features = get_features_per_point_cloud(
                device=device,
                pipe=pipe,
                dino_model=dino_model,
                points=points,
                prompt=prompt,
                num_views=args.num_views,
                H=args.height,
                W=args.width,
                point_radius=args.point_radius,
                points_per_pixel=args.points_per_pixel,
                use_normal_map=use_normal_map,
                view_sampling=args.view_sampling,
                view_observer=debug_writer,
            )
            torch.save(features, save_path)
            if debug_writer is not None:
                debug_writer.complete(features.shape)
            print(f"Saved {save_path}")
            print(f"Feature tensor shape: {tuple(features.shape)}")
        except Exception as error:
            if debug_writer is not None:
                debug_writer.fail(error)
            raise


if __name__ == "__main__":
    main()
