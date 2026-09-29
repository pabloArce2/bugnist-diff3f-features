"""Segment one insect in a BugNIST CT volume and save its surface as a mesh.

crop -> threshold -> clean the mask -> marching cubes at the 0.5 iso-level.
Vertices are written in scan voxel units, ordered (X, Y, Z).
"""

import argparse
from pathlib import Path
import sys

import numpy as np
from skimage import measure
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bugnist_tools.ct import add_segmentation_args, crop_from_args, load_volume, segment_from_args


def parse_args():
    parser = argparse.ArgumentParser(description="Convert a BugNIST TIFF volume to a surface mesh.")
    parser.add_argument("--tif", required=True, help="Input .tif/.tiff volume.")
    parser.add_argument("--out", required=True, help="Output .obj/.ply mesh path.")
    add_segmentation_args(parser)
    parser.add_argument("--center", action="store_true", help="Center mesh vertices around the origin.")
    return parser.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    volume, offset = crop_from_args(load_volume(args.tif), args)
    mask, threshold, threshold_label = segment_from_args(volume, args)

    verts_zyx, faces, _, _ = measure.marching_cubes(mask.astype(np.float32), level=0.5)
    verts_zyx = (verts_zyx + offset) * args.downsample
    verts_xyz = verts_zyx[:, [2, 1, 0]]
    if args.center:
        verts_xyz = verts_xyz - verts_xyz.mean(axis=0, keepdims=True)

    mesh = trimesh.Trimesh(vertices=verts_xyz, faces=faces, process=True)
    mesh.export(out_path)
    print(f"Saved {out_path}")
    print(f"Volume shape after crop/downsample: {volume.shape}")
    print(f"Volume offset before downsample: {tuple(int(v) for v in (offset * args.downsample))}")
    print(f"Segmentation threshold: {threshold:.3f} ({threshold_label})")
    print(f"Segmented voxels: {int(mask.sum())}")
    print(f"Mesh vertices: {len(mesh.vertices)}")
    print(f"Mesh faces: {len(mesh.faces)}")


if __name__ == "__main__":
    main()
