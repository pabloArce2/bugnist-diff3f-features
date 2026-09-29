"""Save the 2D images produced while a descriptor is computed.

`get_features_per_vertex` and `get_features_per_point_cloud` accept a
`view_observer`. A DescriptorDebugWriter passed there writes the render, the
ControlNet inputs and the generated image of each selected view to its own
folder, together with a manifest.json describing the run.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw


SCHEMA_VERSION = 1
MAX_CONTACT_SHEET_VIEWS = 36


def parse_debug_views(spec: str | None, num_views: int) -> set[int]:
    """Parse ``all``, individual indices, or inclusive ranges such as ``2-5``."""
    if num_views <= 0:
        raise ValueError("num_views must be positive.")

    text = "all" if spec is None else str(spec).strip().lower()
    if text == "all":
        return set(range(num_views))
    if not text:
        raise ValueError("--debug-views cannot be empty; use 'all' or one or more view indices.")

    selected: set[int] = set()
    for token in re.split(r"[\s,]+", text):
        if not token:
            continue
        range_match = re.fullmatch(r"(\d+)-(\d+)", token)
        if range_match:
            start, stop = (int(value) for value in range_match.groups())
            if start > stop:
                raise ValueError(f"Invalid descending debug-view range: {token}")
            selected.update(range(start, stop + 1))
            continue
        if not re.fullmatch(r"\d+", token):
            raise ValueError(f"Invalid debug-view selection: {token!r}")
        selected.add(int(token))

    invalid = sorted(index for index in selected if index < 0 or index >= num_views)
    if invalid:
        raise ValueError(
            f"Debug view indices must be between 0 and {num_views - 1}; got "
            + ", ".join(map(str, invalid))
            + "."
        )
    if not selected:
        raise ValueError("No debug views were selected.")
    return selected


def create_debug_run_directory(base: str | Path, kind: str) -> Path:
    """A new, uniquely named folder for one descriptor run."""
    base_path = Path(base)
    base_path.mkdir(parents=True, exist_ok=True)
    safe_kind = _safe_name(kind) or "descriptor"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_path = base_path / f"{safe_kind}_{stamp}_{uuid4().hex[:6]}"
    run_path.mkdir(parents=False, exist_ok=False)
    return run_path


class DescriptorDebugWriter:
    """Writes the selected views of one mesh or point cloud."""

    def __init__(
        self,
        *,
        run_dir: str | Path,
        asset_index: int,
        kind: str,
        input_path: str | Path,
        descriptor_path: str | Path,
        prompt: str,
        num_views: int,
        view_sampling: str,
        image_size: tuple[int, int],
        selected_views: str | None = "all",
        mode: str = "full",
        extra_settings: dict[str, Any] | None = None,
    ) -> None:
        if mode not in {"generated", "full"}:
            raise ValueError("Descriptor debug mode must be 'generated' or 'full'.")

        self.selected_views = parse_debug_views(selected_views, num_views)
        self.mode = mode
        self.asset_dir = Path(run_dir) / f"{asset_index:02d}_{_safe_name(Path(input_path).stem) or 'asset'}"
        self.asset_dir.mkdir(parents=True, exist_ok=False)
        self.manifest_path = self.asset_dir / "manifest.json"
        self._generated_paths: list[tuple[int, Path]] = []
        self._view_records: dict[str, dict[str, Any]] = {}
        self._manifest: dict[str, Any] = {
            "schemaVersion": SCHEMA_VERSION,
            "status": "running",
            "kind": str(kind),
            "inputPath": str(input_path),
            "descriptorPath": str(descriptor_path),
            "prompt": str(prompt),
            "numViews": int(num_views),
            "viewSampling": str(view_sampling),
            "height": int(image_size[0]),
            "width": int(image_size[1]),
            "captureMode": mode,
            "selectedViews": sorted(self.selected_views),
            "capturedViews": [],
            "settings": dict(extra_settings or {}),
            "createdAt": _iso_now(),
            "views": self._view_records,
        }
        self._write_manifest()

    def wants_view(self, view_index: int) -> bool:
        return int(view_index) in self.selected_views

    def capture_view(
        self,
        *,
        view_index: int,
        input_image: Any,
        depth_map: Any,
        normal_map: Any | None,
        visible_mask: Any,
        generated_image: Any,
        prompt: str,
    ) -> None:
        index = int(view_index)
        if not self.wants_view(index):
            return

        view_dir = self.asset_dir / f"view_{index:03d}"
        view_dir.mkdir(parents=True, exist_ok=True)
        generated = _as_rgb_image(generated_image)
        generated_path = view_dir / "06_final_generated.png"
        generated.save(generated_path)
        files = [generated_path.name]

        if self.mode == "full":
            input_render = _as_rgb_image(input_image)
            input_path = view_dir / "01_input_render.png"
            input_render.save(input_path)
            files.insert(0, input_path.name)

            depth_path = view_dir / "02_depth_control.png"
            _depth_control_image(depth_map).save(depth_path)
            files.append(depth_path.name)

            if normal_map is not None:
                normal_path = view_dir / "03_normal_control.png"
                _normal_control_image(normal_map).save(normal_path)
                files.append(normal_path.name)

            mask_path = view_dir / "04_visible_mask.png"
            _mask_image(visible_mask).save(mask_path)
            files.append(mask_path.name)

            change_path = view_dir / "07_ai_change_map.png"
            _change_map(input_render, generated).save(change_path)
            files.append(change_path.name)

        self._generated_paths.append((index, generated_path))
        self._view_records[str(index)] = {
            "directory": view_dir.relative_to(self.asset_dir).as_posix(),
            "prompt": str(prompt),
            "files": files,
        }
        self._manifest["capturedViews"] = sorted(int(value) for value in self._view_records)
        self._write_manifest()

    def complete(self, feature_shape: Any) -> None:
        contact_sheet = self._make_contact_sheet()
        self._manifest.update(
            {
                "status": "complete",
                "featureShape": [int(value) for value in feature_shape],
                "completedAt": _iso_now(),
                "contactSheet": contact_sheet.name if contact_sheet else None,
            }
        )
        self._write_manifest()

    def fail(self, error: BaseException | str) -> None:
        self._manifest.update(
            {
                "status": "failed",
                "error": str(error),
                "failedAt": _iso_now(),
            }
        )
        self._write_manifest()

    def _make_contact_sheet(self) -> Path | None:
        if not self._generated_paths:
            return None

        chosen = _even_sample(self._generated_paths, MAX_CONTACT_SHEET_VIEWS)
        thumb_width, thumb_height = 220, 220
        label_height = 28
        columns = min(4, len(chosen))
        rows = math.ceil(len(chosen) / columns)
        sheet = Image.new("RGB", (columns * thumb_width, rows * (thumb_height + label_height)), "#F4F6F5")
        draw = ImageDraw.Draw(sheet)
        resampling = getattr(Image, "Resampling", Image).LANCZOS

        for position, (view_index, image_path) in enumerate(chosen):
            with Image.open(image_path) as source:
                thumbnail = source.convert("RGB")
                thumbnail.thumbnail((thumb_width - 12, thumb_height - 12), resampling)
            column = position % columns
            row = position // columns
            cell_x = column * thumb_width
            cell_y = row * (thumb_height + label_height)
            image_x = cell_x + (thumb_width - thumbnail.width) // 2
            image_y = cell_y + label_height + (thumb_height - thumbnail.height) // 2
            draw.text((cell_x + 8, cell_y + 7), f"View {view_index}", fill="#20262D")
            sheet.paste(thumbnail, (image_x, image_y))

        path = self.asset_dir / "generated_contact_sheet.png"
        sheet.save(path)
        return path

    def _write_manifest(self) -> None:
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self._manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.manifest_path)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _as_rgb_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    array = _as_numpy(value)
    array = np.squeeze(array)
    if array.ndim != 3 or array.shape[-1] not in {3, 4}:
        raise ValueError(f"Expected an HxWx3/4 image, got shape {array.shape}.")
    array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        if array.size and float(np.nanmax(array)) <= 1.0:
            array = array * 255.0
        array = np.nan_to_num(array)
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")


def _depth_control_image(value: Any) -> Image.Image:
    depth = np.squeeze(_as_numpy(value)).astype(np.float32, copy=True)
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2D depth map, got shape {depth.shape}.")
    invalid = depth == -1
    valid = ~invalid
    if not valid.any():
        return Image.fromarray(np.zeros(depth.shape, dtype=np.uint8), mode="L")
    transformed = float(depth[valid].max()) - depth
    transformed[invalid] = 0
    maximum = float(transformed.max())
    if maximum > 0:
        transformed /= maximum
    return Image.fromarray(np.clip(transformed * 255, 0, 255).astype(np.uint8), mode="L")


def _normal_control_image(value: Any) -> Image.Image:
    normal = _as_numpy(value).astype(np.float32, copy=False)
    if normal.ndim == 4 and normal.shape[-2] == 1:
        normal = normal[..., 0, :]
    normal = np.squeeze(normal)
    if normal.ndim != 3 or normal.shape[-1] < 3:
        raise ValueError(f"Expected an HxWx3 normal map, got shape {normal.shape}.")
    normal = normal[..., :3]
    minimum = float(np.min(normal))
    span = max(float(np.max(normal)) - minimum, 1e-6)
    normalized = np.where(normal != 0, (normal - minimum) / span, 0)
    return Image.fromarray(np.clip(normalized * 255, 0, 255).astype(np.uint8), mode="RGB")


def _mask_image(value: Any) -> Image.Image:
    mask = np.squeeze(_as_numpy(value)).astype(bool)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D visibility mask, got shape {mask.shape}.")
    return Image.fromarray(mask.astype(np.uint8) * 255, mode="L")


def _change_map(input_image: Image.Image, generated_image: Image.Image) -> Image.Image:
    input_array = np.asarray(input_image.convert("RGB"), dtype=np.float32)
    generated_array = np.asarray(generated_image.resize(input_image.size).convert("RGB"), dtype=np.float32)
    difference = np.clip(np.abs(generated_array - input_array) * 2.5, 0, 255).astype(np.uint8)
    return Image.fromarray(difference, mode="RGB")


def _even_sample(items: list[Any], limit: int) -> list[Any]:
    if len(items) <= limit:
        return list(items)
    positions = np.linspace(0, len(items) - 1, num=limit, dtype=int)
    return [items[int(position)] for position in positions]


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value)).strip("._-")


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
