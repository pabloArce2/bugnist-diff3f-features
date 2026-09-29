"""Compute Diff3F descriptors for one or more meshes.

Writes <outdir>/<mesh stem>_diff3f.pt, a float16 tensor of shape
[num_vertices, 2048]. Row i belongs to vertex i of the input OBJ/PLY.
"""

import argparse
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bugnist_tools.camera_sampling import VIEW_SAMPLINGS, validate_view_count
from bugnist_tools.debug_capture import DescriptorDebugWriter, create_debug_run_directory, parse_debug_views
from bugnist_tools.util import pick_device, prompts_per_input
from dataloaders.mesh_container import MeshContainer
from diff3f import get_features_per_vertex
from diffusion import init_pipe
from dino import init_dino
from utils import convert_mesh_container_to_torch_mesh


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Diff3F features for one or more meshes.")
    parser.add_argument("--mesh", nargs="+", required=True, help="Mesh files to process.")
    parser.add_argument("--prompt", nargs="+", required=True, help="One prompt per mesh, or one prompt for all.")
    parser.add_argument("--outdir", default="output/hpc_features", help="Directory for the .pt files.")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda:0 or cpu.")
    parser.add_argument("--num-views", type=int, default=100)
    parser.add_argument(
        "--view-sampling",
        choices=VIEW_SAMPLINGS,
        default="grid",
        help="Camera layout; grid is the original Diff3F one.",
    )
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Ball-query radius for assigning pixels to vertices, as a fraction of the mesh diameter.",
    )
    parser.add_argument("--tosca", action="store_true", help="Divide coordinates by 10, as for the TOSCA meshes.")
    parser.add_argument("--debug-outdir", default=None, help="Also save the per-view images of this run here.")
    parser.add_argument(
        "--debug-views",
        default="all",
        help="Views to save with --debug-outdir: 'all', indices like '0 4 8', or ranges like '0-3'.",
    )
    parser.add_argument("--no-normal-map", action="store_true", help="Use only the depth ControlNet.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip meshes whose .pt already exists.")
    return parser.parse_args()


def main():
    args = parse_args()
    validate_view_count(args.num_views, args.view_sampling)
    if args.debug_outdir:
        parse_debug_views(args.debug_views, args.num_views)

    device = pick_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    prompts = prompts_per_input(args.prompt, len(args.mesh), "--mesh")

    print(f"Using device: {device}")
    print(f"View sampling: {args.view_sampling} ({args.num_views} views)")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    pipe = init_pipe(device, use_normal_map=not args.no_normal_map)
    dino_model = init_dino(device)
    debug_run_dir = None
    if args.debug_outdir:
        debug_run_dir = create_debug_run_directory(args.debug_outdir, "mesh")
        print(f"Saving per-view images in {debug_run_dir}")
        if args.skip_existing:
            print("Ignoring --skip-existing: saving images needs a fresh descriptor run.")

    for mesh_index, (mesh_path, prompt) in enumerate(zip(args.mesh, prompts)):
        mesh_path = Path(mesh_path)
        save_path = outdir / f"{mesh_path.stem}_diff3f.pt"
        if args.skip_existing and not args.debug_outdir and save_path.exists():
            print(f"Skipping existing output: {save_path}")
            continue

        print(f"Processing {mesh_path} with prompt {prompt!r}")
        mesh_container = MeshContainer().load_from_file(str(mesh_path))
        mesh = convert_mesh_container_to_torch_mesh(mesh_container, device=device, is_tosca=args.tosca)

        debug_writer = None
        if debug_run_dir is not None:
            debug_writer = DescriptorDebugWriter(
                run_dir=debug_run_dir,
                asset_index=mesh_index,
                kind="mesh",
                input_path=mesh_path,
                descriptor_path=save_path,
                prompt=prompt,
                num_views=args.num_views,
                view_sampling=args.view_sampling,
                image_size=(args.height, args.width),
                selected_views=args.debug_views,
                mode="full",
                extra_settings={
                    "normalMap": not args.no_normal_map,
                    "tolerance": args.tolerance,
                    "toscaScaling": args.tosca,
                },
            )
        try:
            features = get_features_per_vertex(
                device=device,
                pipe=pipe,
                dino_model=dino_model,
                mesh=mesh,
                prompt=prompt,
                mesh_vertices=mesh.verts_list()[0],
                num_views=args.num_views,
                H=args.height,
                W=args.width,
                tolerance=args.tolerance,
                use_normal_map=not args.no_normal_map,
                view_sampling=args.view_sampling,
                view_observer=debug_writer,
            )
            torch.save(features, save_path)
            if debug_writer is not None:
                debug_writer.complete(features.shape)
            print(f"Saved {save_path}")
        except Exception as error:
            if debug_writer is not None:
                debug_writer.fail(error)
            raise


if __name__ == "__main__":
    main()
