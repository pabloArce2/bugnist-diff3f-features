import argparse
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloaders.mesh_container import MeshContainer
from camera_sampling import VIEW_SAMPLINGS, validate_view_count
from diff3f import get_features_per_vertex
from diffusion import init_pipe
from dino import init_dino
from utils import convert_mesh_container_to_torch_mesh


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Diff3F features for one or more meshes.")
    parser.add_argument("--mesh", nargs="+", required=True, help="Mesh files to process.")
    parser.add_argument(
        "--prompt",
        nargs="+",
        required=True,
        help="Prompt per mesh, or one prompt reused for all meshes.",
    )
    parser.add_argument("--outdir", default="output/hpc_features", help="Directory for .pt outputs.")
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
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--tosca", action="store_true", help="Scale meshes as TOSCA meshes.")
    parser.add_argument("--no-normal-map", action="store_true", help="Disable normal-map ControlNet input.")
    parser.add_argument("--skip-existing", action="store_true", help="Do not recompute existing outputs.")
    return parser.parse_args()


def prompts_for_meshes(prompts, mesh_count):
    if len(prompts) == 1:
        return prompts * mesh_count
    if len(prompts) != mesh_count:
        raise ValueError("--prompt must have either one value or the same number of values as --mesh.")
    return prompts


def main():
    args = parse_args()
    validate_view_count(args.num_views, args.view_sampling)

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_device(device)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    prompts = prompts_for_meshes(args.prompt, len(args.mesh))

    print(f"Using device: {device}")
    print(f"View sampling: {args.view_sampling} ({args.num_views} views)")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    pipe = init_pipe(device, use_normal_map=not args.no_normal_map)
    dino_model = init_dino(device)

    for mesh_path, prompt in zip(args.mesh, prompts):
        mesh_path = Path(mesh_path)
        save_path = outdir / f"{mesh_path.stem}_diff3f.pt"
        if args.skip_existing and save_path.exists():
            print(f"Skipping existing output: {save_path}")
            continue

        print(f"Processing {mesh_path} with prompt {prompt!r}")
        mesh_container = MeshContainer().load_from_file(str(mesh_path))
        mesh = convert_mesh_container_to_torch_mesh(
            mesh_container,
            device=device,
            is_tosca=args.tosca,
        )
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
        )
        torch.save(features, save_path)
        print(f"Saved {save_path}")


if __name__ == "__main__":
    main()
