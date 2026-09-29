"""Loading, cropping and segmenting BugNIST CT volumes.

Volumes are indexed (Z, Y, X), as stored in the TIFF files. Geometry written by
the scripts is converted to (X, Y, Z) at the very end.
"""

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import filters, morphology
import tifffile

THRESHOLD_METHODS = ("auto", "manual", "percentile", "otsu")


def load_volume(path):
    volume = tifffile.imread(path)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D TIFF volume, got shape {volume.shape}")
    return volume


def crop_roi(volume, start_zyx, size_zyx):
    start = np.array(start_zyx, dtype=int)
    size = np.array(size_zyx, dtype=int)
    if np.any(start < 0) or np.any(size <= 0):
        raise ValueError("ROI start must be non-negative and ROI size must be positive.")
    stop = np.minimum(start + size, volume.shape)
    start = np.minimum(start, stop)
    slices = tuple(slice(int(s), int(e)) for s, e in zip(start, stop))
    return volume[slices], start


def crop_around(volume, center_zyx, size_zyx):
    size = np.array(size_zyx, dtype=int)
    start = np.maximum(center_zyx - size // 2, 0)
    stop = np.minimum(start + size, volume.shape)
    start = np.maximum(stop - size, 0)
    slices = tuple(slice(int(s), int(e)) for s, e in zip(start, stop))
    return volume[slices], start


def crop_to_mask(volume, mask, padding_zyx):
    coords = np.argwhere(mask)
    if len(coords) == 0:
        raise ValueError("Auto-crop mask is empty. Try a lower threshold or disable --auto-crop.")
    padding = np.array(padding_zyx, dtype=int)
    start = np.maximum(coords.min(axis=0) - padding, 0)
    stop = np.minimum(coords.max(axis=0) + padding + 1, volume.shape)
    slices = tuple(slice(int(s), int(e)) for s, e in zip(start, stop))
    return volume[slices], start


def centroid_from_csv(path, row_index, coord_order="xyz"):
    """Read one specimen centre from a BugNIST mixture centroid CSV. Returns (Z, Y, X)."""
    frame = pd.read_csv(path)
    if row_index < 0 or row_index >= len(frame):
        raise IndexError(f"--centroid-index {row_index} is outside CSV rows 0..{len(frame) - 1}")

    columns = {col.lower(): col for col in frame.columns}
    named = {}
    for axis in ("x", "y", "z"):
        for key in (axis, f"c{axis}", f"pos{axis}", f"{axis}_centroid", f"centroid_{axis}", f"center_{axis}"):
            if key in columns:
                named[axis] = columns[key]
                break

    row = frame.iloc[row_index]
    if len(named) == 3:
        xyz = np.array([row[named["x"]], row[named["y"]], row[named["z"]]], dtype=float)
    else:
        numeric = [col for col in frame.columns if pd.api.types.is_numeric_dtype(frame[col])]
        if len(numeric) < 3:
            raise ValueError("Could not infer centroid columns; expected named x/y/z or at least 3 numeric columns.")
        values = np.array([row[numeric[0]], row[numeric[1]], row[numeric[2]]], dtype=float)
        xyz = values if coord_order == "xyz" else values[[2, 1, 0]]

    return np.rint(xyz[[2, 1, 0]]).astype(int)


def resolve_threshold(volume, threshold=None, percentile=None, method="auto"):
    """Return (threshold, label).

    "auto" uses the manual threshold if one is given, then the percentile, and
    falls back to Otsu.
    """
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        raise ValueError("Input volume has no finite voxels.")

    if method == "auto":
        if threshold is not None:
            method = "manual"
        elif percentile is not None:
            method = "percentile"
        else:
            method = "otsu"

    if method == "manual":
        if threshold is None:
            raise ValueError("--threshold-method manual requires --threshold.")
        if percentile is not None:
            raise ValueError("Use either --threshold or --threshold-percentile, not both.")
        return float(threshold), "manual"

    if method == "percentile":
        if threshold is not None:
            raise ValueError("Use either --threshold or --threshold-percentile, not both.")
        if percentile is None:
            raise ValueError("--threshold-method percentile requires --threshold-percentile.")
        return float(np.percentile(finite, percentile)), f"p{percentile:g}"

    if threshold is not None or percentile is not None:
        raise ValueError("--threshold-method otsu chooses the threshold automatically; omit manual threshold args.")
    return float(filters.threshold_otsu(finite)), "otsu"


def segment(
    volume,
    threshold=None,
    percentile=None,
    method="auto",
    invert=False,
    min_size=512,
    keep_largest=False,
    opening_radius=0,
    closing_radius=0,
    fill_holes=False,
):
    """Threshold the volume and clean the mask. Returns (mask, threshold, label)."""
    threshold, label = resolve_threshold(volume, threshold, percentile, method)

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

    return mask, threshold, label


def add_threshold_args(parser):
    parser.add_argument(
        "--threshold-method",
        choices=THRESHOLD_METHODS,
        default="auto",
        help="auto uses --threshold if given, then --threshold-percentile, otherwise Otsu.",
    )
    parser.add_argument("--threshold", type=float, help="Manual intensity threshold.")
    parser.add_argument("--threshold-percentile", type=float, help="Use this intensity percentile as threshold, e.g. 95.")


def add_segmentation_args(parser):
    """Arguments shared by the TIFF-to-mesh and TIFF-to-point-cloud converters."""
    parser.add_argument("--centroids", help="Centroid CSV, for BugNIST mixture volumes.")
    parser.add_argument("--centroid-index", type=int, help="Row in the centroid CSV to crop around.")
    parser.add_argument("--crop-size", type=int, nargs=3, metavar=("Z", "Y", "X"), help="Crop size around the centroid.")
    parser.add_argument(
        "--coord-order",
        choices=("xyz", "zyx"),
        default="xyz",
        help="Column order used when the centroid CSV has unnamed columns.",
    )
    parser.add_argument("--roi-start", type=int, nargs=3, metavar=("Z", "Y", "X"), help="Manual ROI crop start.")
    parser.add_argument("--roi-size", type=int, nargs=3, metavar=("Z", "Y", "X"), help="Manual ROI crop size.")
    parser.add_argument("--auto-crop", action="store_true", help="Crop to the bounding box of the thresholded mask.")
    parser.add_argument(
        "--auto-crop-padding",
        type=int,
        nargs=3,
        default=(8, 8, 8),
        metavar=("Z", "Y", "X"),
        help="Padding around the auto-crop box, in voxels.",
    )
    parser.add_argument("--downsample", type=int, default=1, help="Keep every Nth voxel along each axis.")
    add_threshold_args(parser)
    parser.add_argument("--invert", action="store_true", help="Segment dark objects instead of bright ones.")
    parser.add_argument("--min-size", type=int, default=512, help="Remove components smaller than this (voxels).")
    parser.add_argument("--opening-radius", type=int, default=0, help="Binary opening radius, removes thin noise.")
    parser.add_argument("--closing-radius", type=int, default=0, help="Binary closing radius, closes small gaps.")
    parser.add_argument("--fill-holes", action="store_true", help="Fill enclosed cavities in the mask.")
    parser.add_argument("--keep-largest", action="store_true", help="Keep only the largest connected component.")


def segment_from_args(volume, args):
    return segment(
        volume,
        threshold=args.threshold,
        percentile=args.threshold_percentile,
        method=args.threshold_method,
        invert=args.invert,
        min_size=args.min_size,
        keep_largest=args.keep_largest,
        opening_radius=args.opening_radius,
        closing_radius=args.closing_radius,
        fill_holes=args.fill_holes,
    )


def crop_from_args(volume, args, offset_dtype=np.float64):
    """Apply the centroid/ROI crop, auto-crop and downsampling from the CLI arguments.

    Returns the cropped volume and the offset of its first voxel in the original
    volume, divided by the downsampling factor. Geometry extracted from the
    cropped volume goes back to scan coordinates with (coords + offset) * downsample.
    """
    offset = np.zeros(3, dtype=offset_dtype)
    if args.centroids or args.centroid_index is not None or args.crop_size is not None:
        if not (args.centroids and args.centroid_index is not None and args.crop_size):
            raise ValueError("--centroids, --centroid-index, and --crop-size must be used together.")
        if args.roi_start is not None or args.roi_size is not None:
            raise ValueError("Use either centroid crop args or manual ROI args, not both.")
        center = centroid_from_csv(args.centroids, args.centroid_index, args.coord_order)
        volume, start = crop_around(volume, center, np.array(args.crop_size))
        offset = start.astype(offset_dtype)
    elif args.roi_start is not None or args.roi_size is not None:
        if args.roi_start is None or args.roi_size is None:
            raise ValueError("--roi-start and --roi-size must be used together.")
        volume, start = crop_roi(volume, args.roi_start, args.roi_size)
        offset = start.astype(offset_dtype)

    if args.auto_crop:
        mask = segment_from_args(volume, args)[0]
        volume, start = crop_to_mask(volume, mask, args.auto_crop_padding)
        offset = offset + start.astype(offset_dtype)

    if args.downsample < 1:
        raise ValueError("--downsample must be >= 1")
    if args.downsample > 1:
        step = args.downsample
        volume = volume[::step, ::step, ::step]
        offset = offset / step

    return volume, offset
