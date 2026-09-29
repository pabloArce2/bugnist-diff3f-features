"""Colour a mesh by the first three PCA components of its Diff3F descriptor.

Colours from separate runs of this script are not comparable between meshes;
use visualize_feature_comparison.py for that.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bugnist_tools.features import load_features, pca_colors
from bugnist_tools.geometry import check_rows, load_mesh
from bugnist_tools.preview import save_color_preview


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize per-vertex .pt features as colored mesh/preview files.")
    parser.add_argument("--mesh", required=True, help="The mesh the .pt file was computed from.")
    parser.add_argument("--features", required=True, help=".pt tensor with shape [num_vertices, feature_dim].")
    parser.add_argument("--out", required=True, help="Output colored .ply path.")
    parser.add_argument("--preview", help="Optional .png preview.")
    parser.add_argument("--fit-sample", type=int, default=12000, help="Rows used to fit the PCA.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clip-percentiles", type=float, nargs=2, default=(1.0, 99.0), metavar=("LOW", "HIGH"))
    parser.add_argument("--no-normalize", action="store_true", help="Do not L2-normalise the rows before PCA.")
    parser.add_argument("--invert", action="store_true", help="Invert the RGB colors.")
    parser.add_argument("--preview-max-points", type=int, default=50000)
    parser.add_argument("--preview-size", type=int, default=1200)
    parser.add_argument("--elev", type=float, default=24.0)
    parser.add_argument("--azim", type=float, default=38.0)
    return parser.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out)

    vertices, faces = load_mesh(args.mesh)
    features = load_features(args.features)
    check_rows(Path(args.mesh).name, len(vertices), features.shape[0])

    colors = pca_colors(
        features,
        fit_sample=args.fit_sample,
        seed=args.seed,
        clip_percentiles=args.clip_percentiles,
        normalize=not args.no_normalize,
        invert=args.invert,
    )
    vertex_colors = np.column_stack((colors, np.full(len(colors), 255, dtype=np.uint8)))
    colored_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors, process=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    colored_mesh.export(out_path)
    print(f"Saved colored mesh: {out_path}")

    if args.preview:
        save_color_preview(
            vertices, colors, args.preview, args.preview_max_points, args.preview_size, args.elev, args.azim, args.seed
        )
        print(f"Saved preview: {args.preview}")


if __name__ == "__main__":
    main()
