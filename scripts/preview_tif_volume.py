import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tifffile


AXES = {"z": 0, "y": 1, "x": 2}


def parse_args():
    parser = argparse.ArgumentParser(description="Preview raw 3D TIFF CT volumes as PNG sheets.")
    parser.add_argument("--tif", nargs="+", required=True, help="Input .tif/.tiff volume files.")
    parser.add_argument("--outdir", default="previews/bugnist_raw_ct")
    parser.add_argument("--slices", type=int, default=12, help="Number of slices per axis sheet.")
    parser.add_argument("--axis", choices=("all", "z", "y", "x"), default="all")
    parser.add_argument(
        "--contrast-percentiles",
        type=float,
        nargs=2,
        default=(1.0, 99.5),
        metavar=("LOW", "HIGH"),
        help="Percentiles used to map CT intensities to 8-bit preview images.",
    )
    parser.add_argument("--downsample", type=int, default=1, help="Preview stride in Z/Y/X.")
    parser.add_argument("--roi-start", type=int, nargs=3, metavar=("Z", "Y", "X"))
    parser.add_argument("--roi-size", type=int, nargs=3, metavar=("Z", "Y", "X"))
    parser.add_argument("--threshold", type=float, help="Manual intensity threshold for overlay.")
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=95.0,
        help="Intensity percentile for red overlay when --threshold is omitted.",
    )
    parser.add_argument("--no-overlay", action="store_true", help="Skip threshold-overlay sheets.")
    return parser.parse_args()


def apply_roi(volume, roi_start, roi_size):
    if roi_start is None and roi_size is None:
        return volume, np.zeros(3, dtype=int)
    if roi_start is None or roi_size is None:
        raise ValueError("--roi-start and --roi-size must be used together.")

    start = np.array(roi_start, dtype=int)
    size = np.array(roi_size, dtype=int)
    if np.any(start < 0) or np.any(size <= 0):
        raise ValueError("ROI start must be non-negative and ROI size must be positive.")

    stop = np.minimum(start + size, volume.shape)
    start = np.minimum(start, stop)
    slices = tuple(slice(int(s), int(e)) for s, e in zip(start, stop))
    return volume[slices], start


def normalize_to_u8(image, low, high):
    image = image.astype(np.float32, copy=False)
    if high <= low:
        high = low + 1.0
    image = np.clip((image - low) / (high - low), 0.0, 1.0)
    return (image * 255).astype(np.uint8)


def slice_along(volume, axis, index):
    axis_index = AXES[axis]
    if axis_index == 0:
        return volume[index, :, :]
    if axis_index == 1:
        return volume[:, index, :]
    return volume[:, :, index]


def fit_image(image, max_size=260):
    image = image.convert("RGB")
    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return image


def make_sheet(items, out_path, cols=4, thumb=260, label_h=26):
    rows = math.ceil(len(items) / cols)
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + label_h)), (246, 246, 243))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for item_index, (image, label) in enumerate(items):
        image = fit_image(image, thumb)
        cell_x = (item_index % cols) * thumb
        cell_y = (item_index // cols) * (thumb + label_h)
        x = cell_x + (thumb - image.width) // 2
        y = cell_y + (thumb - image.height) // 2
        sheet.paste(image, (x, y))
        draw.text((cell_x + 6, cell_y + thumb + 5), label, fill=(42, 45, 48), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def make_overlay(image_u8, mask, alpha=0.45):
    rgb = np.repeat(image_u8[:, :, None], 3, axis=2).astype(np.float32)
    color = np.array([255.0, 64.0, 24.0], dtype=np.float32)
    rgb[mask] = rgb[mask] * (1.0 - alpha) + color * alpha
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def selected_indices(length, count):
    if count <= 1:
        return [length // 2]
    return np.linspace(0, length - 1, count, dtype=int).tolist()


def threshold_value(volume, threshold, threshold_percentile):
    if threshold is not None:
        return threshold
    finite = volume[np.isfinite(volume)]
    return float(np.percentile(finite, threshold_percentile))


def write_slices(volume, outdir, stem, axes, low, high, threshold, threshold_percentile, overlay, count):
    mask = None
    threshold_label = None
    if overlay:
        threshold = threshold_value(volume, threshold, threshold_percentile)
        threshold_label = f"thr {threshold:.2f}"
        mask = volume > threshold

    for axis in axes:
        items = []
        overlay_items = []
        for index in selected_indices(volume.shape[AXES[axis]], count):
            raw_slice = slice_along(volume, axis, index)
            image_u8 = normalize_to_u8(raw_slice, low, high)
            items.append((Image.fromarray(image_u8), f"{axis}={index}"))
            if overlay:
                overlay_slice = slice_along(mask, axis, index)
                overlay_items.append((make_overlay(image_u8, overlay_slice), f"{axis}={index}"))

        make_sheet(items, outdir / f"{stem}_slices_{axis}.png")
        if overlay:
            make_sheet(overlay_items, outdir / f"{stem}_slices_{axis}_overlay_{threshold_label.replace(' ', '_')}.png")


def write_orthos(volume, outdir, stem, low, high):
    items = []
    for axis in ("z", "y", "x"):
        index = volume.shape[AXES[axis]] // 2
        image = Image.fromarray(normalize_to_u8(slice_along(volume, axis, index), low, high))
        items.append((image, f"{axis} center={index}"))
    make_sheet(items, outdir / f"{stem}_orthos.png", cols=3)


def write_mips(volume, outdir, stem, low, high, threshold, threshold_percentile, overlay):
    items = []
    overlay_items = []
    mask = None
    threshold_label = None
    if overlay:
        threshold = threshold_value(volume, threshold, threshold_percentile)
        threshold_label = f"thr {threshold:.2f}"
        mask = volume > threshold

    for axis in ("z", "y", "x"):
        axis_index = AXES[axis]
        mip = volume.max(axis=axis_index)
        image_u8 = normalize_to_u8(mip, low, high)
        items.append((Image.fromarray(image_u8), f"max along {axis}"))
        if overlay:
            overlay_mask = mask.max(axis=axis_index).astype(bool)
            overlay_items.append((make_overlay(image_u8, overlay_mask), f"max along {axis}"))

    make_sheet(items, outdir / f"{stem}_mips.png", cols=3)
    if overlay:
        make_sheet(overlay_items, outdir / f"{stem}_mips_overlay_{threshold_label.replace(' ', '_')}.png", cols=3)


def write_histogram(volume, outdir, stem, threshold, threshold_percentile):
    values = volume[np.isfinite(volume)].ravel()
    hist, edges = np.histogram(values, bins=256, range=(float(values.min()), float(values.max())))
    hist = np.log1p(hist)
    width, height = 900, 320
    margin = 46
    image = Image.new("RGB", (width, height), (248, 248, 246))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    plot_w = width - margin * 2
    plot_h = height - margin * 2
    max_h = max(float(hist.max()), 1.0)

    for i, value in enumerate(hist):
        x0 = margin + int(i * plot_w / len(hist))
        x1 = margin + int((i + 1) * plot_w / len(hist))
        y0 = height - margin
        y1 = y0 - int(value / max_h * plot_h)
        draw.rectangle((x0, y1, max(x1, x0 + 1), y0), fill=(95, 103, 112))

    threshold = threshold_value(volume, threshold, threshold_percentile)
    x_thr = margin + int((threshold - edges[0]) / max(edges[-1] - edges[0], 1e-6) * plot_w)
    draw.line((x_thr, margin, x_thr, height - margin), fill=(230, 70, 30), width=3)
    draw.text((margin, 12), f"{stem} intensity histogram, threshold={threshold:.2f}", fill=(36, 39, 42), font=font)
    draw.text((margin, height - margin + 12), f"{edges[0]:.1f}", fill=(36, 39, 42), font=font)
    draw.text((width - margin - 60, height - margin + 12), f"{edges[-1]:.1f}", fill=(36, 39, 42), font=font)

    out_path = outdir / f"{stem}_hist.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def write_summary(volume, outdir, stem, original_shape, offset, low, high, threshold, threshold_percentile):
    threshold = threshold_value(volume, threshold, threshold_percentile)
    lines = [
        f"stem: {stem}",
        f"original_shape_zyx: {tuple(original_shape)}",
        f"preview_shape_zyx: {tuple(volume.shape)}",
        f"roi_offset_zyx: {tuple(int(v) for v in offset)}",
        f"dtype: {volume.dtype}",
        f"min: {float(volume.min()):.3f}",
        f"max: {float(volume.max()):.3f}",
        f"contrast_low: {low:.3f}",
        f"contrast_high: {high:.3f}",
        f"overlay_threshold: {threshold:.3f}",
    ]
    out_path = outdir / f"{stem}_summary.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def preview_volume(path, args):
    path = Path(path)
    volume = tifffile.imread(path)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D TIFF volume, got shape {volume.shape}")

    original_shape = volume.shape
    volume, offset = apply_roi(volume, args.roi_start, args.roi_size)
    if args.downsample < 1:
        raise ValueError("--downsample must be >= 1")
    if args.downsample > 1:
        volume = volume[:: args.downsample, :: args.downsample, :: args.downsample]

    finite = volume[np.isfinite(volume)]
    low, high = np.percentile(finite, args.contrast_percentiles)
    outdir = Path(args.outdir) / path.stem
    axes = ("z", "y", "x") if args.axis == "all" else (args.axis,)
    overlay = not args.no_overlay

    write_orthos(volume, outdir, path.stem, low, high)
    write_mips(volume, outdir, path.stem, low, high, args.threshold, args.threshold_percentile, overlay)
    write_slices(volume, outdir, path.stem, axes, low, high, args.threshold, args.threshold_percentile, overlay, args.slices)
    write_histogram(volume, outdir, path.stem, args.threshold, args.threshold_percentile)
    write_summary(volume, outdir, path.stem, original_shape, offset, low, high, args.threshold, args.threshold_percentile)
    print(f"Saved previews for {path} in {outdir}")


def main():
    args = parse_args()
    for tif_path in args.tif:
        preview_volume(tif_path, args)


if __name__ == "__main__":
    main()
