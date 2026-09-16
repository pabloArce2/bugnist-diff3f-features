import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from descriptor_debug_capture import (  # noqa: E402
    DescriptorDebugWriter,
    create_debug_run_directory,
    parse_debug_views,
)


class DebugViewSelectionTests(unittest.TestCase):
    def test_all_and_mixed_ranges(self):
        self.assertEqual(parse_debug_views("all", 4), {0, 1, 2, 3})
        self.assertEqual(parse_debug_views("0, 2-4 7", 8), {0, 2, 3, 4, 7})

    def test_rejects_bad_or_out_of_range_values(self):
        for value in ("", "left", "4-2", "8"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_debug_views(value, 8)


class DescriptorDebugWriterTests(unittest.TestCase):
    def test_writes_selected_exact_view_bundle_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = create_debug_run_directory(temporary, "mesh")
            writer = DescriptorDebugWriter(
                run_dir=run_dir,
                asset_index=0,
                kind="mesh",
                input_path="meshes/camel.obj",
                descriptor_path="output/camel_diff3f.pt",
                prompt="camel",
                num_views=3,
                view_sampling="fibonacci",
                image_size=(8, 10),
                selected_views="0,2",
                mode="full",
                extra_settings={"toscaScaling": True},
            )

            input_image = np.full((8, 10, 3), 180, dtype=np.uint8)
            generated_image = Image.fromarray(np.full((8, 10, 3), 120, dtype=np.uint8))
            depth = np.linspace(0.1, 1.0, 80, dtype=np.float32).reshape(1, 8, 10)
            depth[:, 0, :] = -1
            normal = np.ones((8, 10, 1, 3), dtype=np.float32)
            visible = depth[0] != -1

            writer.capture_view(
                view_index=0,
                input_image=input_image,
                depth_map=depth,
                normal_map=normal,
                visible_mask=visible,
                generated_image=generated_image,
                prompt="camel",
            )
            writer.capture_view(
                view_index=1,
                input_image=input_image,
                depth_map=depth,
                normal_map=normal,
                visible_mask=visible,
                generated_image=generated_image,
                prompt="camel",
            )
            writer.complete((1789, 2048))

            asset_dir = next(path for path in run_dir.iterdir() if path.is_dir())
            view_dir = asset_dir / "view_000"
            self.assertTrue((view_dir / "01_input_render.png").is_file())
            self.assertTrue((view_dir / "02_depth_control.png").is_file())
            self.assertTrue((view_dir / "03_normal_control.png").is_file())
            self.assertTrue((view_dir / "04_visible_mask.png").is_file())
            self.assertTrue((view_dir / "06_final_generated.png").is_file())
            self.assertTrue((view_dir / "07_ai_change_map.png").is_file())
            self.assertFalse((asset_dir / "view_001").exists())
            self.assertTrue((asset_dir / "generated_contact_sheet.png").is_file())

            manifest = json.loads((asset_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["selectedViews"], [0, 2])
            self.assertEqual(manifest["capturedViews"], [0])
            self.assertEqual(manifest["featureShape"], [1789, 2048])
            self.assertTrue(manifest["settings"]["toscaScaling"])


if __name__ == "__main__":
    unittest.main()
