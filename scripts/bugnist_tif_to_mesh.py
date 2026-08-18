import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import filters, measure, morphology
import tifffile
import trimesh


def parse_args():
    parser = argparse.ArgumentParser(description="Convert a BugNIST TIFF volume to a surface mesh.")
    parser.add_argument("--tif", required=True, help="Input .tif/.tiff volume.")
    parser.add_argument("--out", required=True, help="Output .obj/.ply mesh path.")
    parser.add_argument("--centroids", help="Optional centroid CSV for mixture volumes.")
    parser.add_argument("--centroid-index", type=int, help="Row index in the centroid CSV to crop around.")
    parser.add_argument(
        "--crop-size",
        type=int,
        nargs=3,
        metavar=("Z", "Y", "X"),
        help="Crop size around the selected centroid, in voxels.",
    )
    parser.add_argument("--roi-start", type=int, nargs=3, metavar=("Z", "Y", "X"), help="Manual ROI crop start.")
    parser.add_argument("--roi-size", type=int, nargs=3, metavar=("Z", "Y", "X"), help="Manual ROI crop size.")
    parser.add_argument("--auto-crop", action="store_true", help="Crop to the threshold-mask bounding box before meshing.")
    parser.add_argument(
        "--auto-crop-padding",
        type=int,
        nargs=3,
        default=(8, 8, 8),
        metavar=("Z", "Y", "X"),
        help="Padding around an auto-cropped threshold-mask bounding box.",
    )
    parser.add_argument(
        "--coord-order",
        choices=("xyz", "zyx"),
        default="xyz",
        help="Coordinate order for fallback CSV parsing when columns are unnamed.",
    )
    parser.add_argument("--downsample", type=int, default=1, help="Stride downsampling factor.")
    parser.add_argument(
        "--threshold-method",
        choices=("auto", "manual", "percentile", "otsu"),
        default="auto",
        help=(
            "How to choose the segmentation threshold. "
            "auto uses --threshold if supplied, then --threshold-percentile if supplied, otherwise Otsu."
        ),
    )
    parser.add_argument("--threshold", type=float, help="Manual intensity threshold.")
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        help="Use this intensity percentile as the threshold, e.g. 95.",
    )
    parser.add_argument("--invert", action="store_true", help="Segment dark objects instead of bright objects.")
    parser.add_argument("--min-size", type=int, default=512, help="Remove components smaller than this.")
    parser.add_argument("--opening-radius", type=int, default=0, help="3D binary opening radius for removing thin noise.")
    parser.add_argument("--closing-radius", type=int, default=0, help="3D binary closing radius for smoothing gaps.")
    parser.add_argument("--fill-holes", action="store_true", help="Fill binary holes before marching cubes.")
    parser.add_argument("--keep-largest", action="store_true", help="Keep only the largest connected component.")
    parser.add_argument("--center", action="store_true", help="Center mesh vertices around the origin.")
    return parser.parse_args()


def numeric_columns(frame):
    return [col for col in frame.columns if pd.api.types.is_numeric_dtype(frame[col])]


def centroid_from_csv(path, row_index, coord_order):
    frame = pd.read_csv(path)
    if row_index < 0 or row_index >= len(frame):
        raise IndexError(f"--centroid-index {row_index} is outside CSV rows 0..{len(frame) - 1}")

    lower_to_col = {col.lower(): col for col in frame.columns}
    named = {}
    for axis in ("x", "y", "z"):
        for key in (axis, f"c{axis}", f"{axis}_centroid", f"centroid_{axis}", f"center_{axis}"):
            if key in lower_to_col:
                named[axis] = lower_to_col[key]
                break

    row = frame.iloc[row_index]
    if len(named) == 3:
        xyz = np.array([row[named["x"]], row[named["y"]], row[named["z"]]], dtype=float)
    else:
        cols = numeric_columns(frame)
        if len(cols) < 3:
            raise ValueError("Could not infer centroid columns; expected named x/y/z or at least 3 numeric columns.")
        values = np.array([row[cols[0]], row[cols[1]], row[cols[2]]], dtype=float)
        xyz = values if coord_order == "xyz" else values[[2, 1, 0]]

    return np.rint(xyz[[2, 1, 0]]).astype(int)


def crop_volume(volume, center_zyx, crop_size_zyx):
    crop_size_zyx = np.array(crop_size_zyx, dtype=int)
    start = np.maximum(center_zyx - crop_size_zyx // 2, 0)
    stop = np.minimum(start + crop_size_zyx, volume.shape)
    start = np.maximum(stop - crop_size_zyx, 0)
    slices = tuple(slice(int(s), int(e)) for s, e in zip(start, stop))
    return volume[slices], start


def crop_volume_from_start(volume, start_zyx, size_zyx):
    start = np.array(start_zyx, dtype=int)
    size = np.array(size_zyx, dtype=int)
    if np.any(start < 0) or np.any(size <= 0):
        raise ValueError("ROI start must be non-negative and ROI size must be positive.")
    stop = np.minimum(start + size, volume.shape)
    start = np.minimum(start, stop)
    slices = tuple(slice(int(s), int(e)) for s, e in zip(start, stop))
    return volume[slices], start


def crop_to_mask_bbox(volume, mask, padding_zyx):
    coords = np.argwhere(mask)
    if len(coords) == 0:
        raise ValueError("Auto-crop mask is empty. Try a lower threshold or disable --auto-crop.")

    padding = np.array(padding_zyx, dtype=int)
    start = np.maximum(coords.min(axis=0) - padding, 0)
    stop = np.minimum(coords.max(axis=0) + padding + 1, volume.shape)
    slices = tuple(slice(int(s), int(e)) for s, e in zip(start, stop))
    return volume[slices], start


def resolve_threshold(volume, threshold, threshold_percentile, threshold_method):
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        raise ValueError("Input volume has no finite voxels.")

    method = threshold_method
    if method == "auto":
        if threshold is not None:
            method = "manual"
        elif threshold_percentile is not None:
            method = "percentile"
        else:
            method = "otsu"

    if method == "manual":
        if threshold is None:
            raise ValueError("--threshold-method manual requires --threshold.")
        if threshold_percentile is not None:
            raise ValueError("Use either --threshold or --threshold-percentile, not both.")
        return float(threshold), "manual"

    if method == "percentile":
        if threshold is not None:
            raise ValueError("Use either --threshold or --threshold-percentile, not both.")
        if threshold_percentile is None:
            raise ValueError("--threshold-method percentile requires --threshold-percentile.")
        return float(np.percentile(finite, threshold_percentile)), f"p{threshold_percentile:g}"

    if threshold is not None or threshold_percentile is not None:
        raise ValueError("--threshold-method otsu chooses the threshold automatically; omit manual threshold args.")
    return float(filters.threshold_otsu(finite)), "otsu"


def make_mask(
    volume,
    threshold,
    threshold_percentile,
    threshold_method,
    invert,
    min_size,
    keep_largest,
    opening_radius=0,
    closing_radius=0,
    fill_holes=False,
):
    threshold, threshold_label = resolve_threshold(volume, threshold, threshold_percentile, threshold_method)

    mask = volume < threshold if invert else volume > threshold
    if min_size > 0:
        mask = morphology.remove_small_objects(mask, min_size=min_size)
    if opening_radius > 0:
        mask = morphology.binary_opening(mask, footprint=morphology.ball(opening_radius))
    if closing_radius > 0:
        mask = morphology.binary_closing(mask, footprint=morphology.ball(closing_radius))
    if fill_holes:
        mask = ndi.binary_fill_holes(mask)

    labels, count = ndi.label(mask)
    if count == 0:
        raise ValueError("Segmentation is empty. Try --invert, --threshold, or --threshold-percentile.")

    if keep_largest:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        mask = labels == sizes.argmax()

    return mask, threshold, threshold_label


def main():
    args = parse_args()
    tif_path = Path(args.tif)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    volume = tifffile.imread(tif_path)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D TIFF volume, got shape {volume.shape}")

    offset = np.zeros(3, dtype=float)
    if args.centroids or args.centroid_index is not None or args.crop_size is not None:
        if not (args.centroids and args.centroid_index is not None and args.crop_size):
            raise ValueError("--centroids, --centroid-index, and --crop-size must be used together.")
        if args.roi_start is not None or args.roi_size is not None:
            raise ValueError("Use either centroid crop args or manual ROI args, not both.")
        center_zyx = centroid_from_csv(args.centroids, args.centroid_index, args.coord_order)
        volume, offset = crop_volume(volume, center_zyx, np.array(args.crop_size))
    elif args.roi_start is not None or args.roi_size is not None:
        if args.roi_start is None or args.roi_size is None:
            raise ValueError("--roi-start and --roi-size must be used together.")
        volume, offset = crop_volume_from_start(volume, args.roi_start, args.roi_size)

    if args.auto_crop:
        auto_mask = make_mask(
            volume,
            args.threshold,
            args.threshold_percentile,
            args.threshold_method,
            args.invert,
            args.min_size,
            args.keep_largest,
            args.opening_radius,
            args.closing_radius,
            args.fill_holes,
        )[0]
        volume, auto_offset = crop_to_mask_bbox(volume, auto_mask, args.auto_crop_padding)
        offset = offset + auto_offset

    if args.downsample < 1:
        raise ValueError("--downsample must be >= 1")
    if args.downsample > 1:
        volume = volume[:: args.downsample, :: args.downsample, :: args.downsample]
        offset = offset / args.downsample

    mask, threshold, threshold_label = make_mask(
        volume,
        args.threshold,
        args.threshold_percentile,
        args.threshold_method,
        args.invert,
        args.min_size,
        args.keep_largest,
        args.opening_radius,
        args.closing_radius,
        args.fill_holes,
    )

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
