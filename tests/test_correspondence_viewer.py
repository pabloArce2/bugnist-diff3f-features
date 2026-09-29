import csv
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import correspondence_to_html_viewer as viewer  # noqa: E402


class CorrespondenceViewerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.source_vertices = np.array(
            [[0.125, 0.0, 0.0], [1.5, 0.25, 0.0], [0.0, 2.75, 0.5]],
            dtype=np.float64,
        )
        self.target_vertices = np.array(
            [[10.25, -1.0, 0.5], [12.5, 0.75, 1.0], [9.5, 2.0, -0.25]],
            dtype=np.float64,
        )

    def benchmark_row(self):
        source = self.source_vertices[1]
        target = self.target_vertices[1]
        gt_vertex = self.target_vertices[0]
        return {
            "label": "TEST_LANDMARK",
            "source_index": "1",
            "target_gt_index": "0",
            "predicted_target_index": "1",
            "source_x": str(source[0]),
            "source_y": str(source[1]),
            "source_z": str(source[2]),
            "target_landmark_x": "10.6",
            "target_landmark_y": "-0.7",
            "target_landmark_z": "0.65",
            "target_gt_x": str(gt_vertex[0]),
            "target_gt_y": str(gt_vertex[1]),
            "target_gt_z": str(gt_vertex[2]),
            "pred_target_x": str(target[0]),
            "pred_target_y": str(target[1]),
            "pred_target_z": str(target[2]),
            "cosine_score": "0.8125",
            "target_error": "2.5",
            "target_error_bbox_pct": "3.75",
            "gt_feature_rank": "17",
        }

    def write_benchmark(self, row, name="benchmark.csv"):
        path = self.root / name
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        return path

    def write_triangle_obj(self, name, vertices):
        path = self.root / name
        lines = [f"v {x:.12g} {y:.12g} {z:.12g}" for x, y, z in vertices]
        lines.append("f 1 2 3")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_csv_coordinates_from_wrong_mesh_are_rejected(self):
        row = self.benchmark_row()
        row["source_x"] = str(float(row["source_x"]) + 0.01)
        path = self.write_benchmark(row)

        with self.assertRaisesRegex(ValueError, r"wrong mesh"):
            viewer.load_matches(
                path,
                "benchmark",
                self.source_vertices,
                self.target_vertices,
            )

    def test_out_of_range_csv_index_is_rejected(self):
        row = self.benchmark_row()
        row["predicted_target_index"] = str(len(self.target_vertices))
        path = self.write_benchmark(row)

        with self.assertRaisesRegex(ValueError, r"outside geometry index range"):
            viewer.load_matches(
                path,
                "benchmark",
                self.source_vertices,
                self.target_vertices,
            )

    def test_manual_ground_truth_remains_distinct_from_snapped_vertex(self):
        path = self.write_benchmark(self.benchmark_row())
        match = viewer.load_matches(
            path,
            "benchmark",
            self.source_vertices,
            self.target_vertices,
        )[0]

        self.assertEqual(match["gt"], [10.6, -0.7, 0.65])
        self.assertEqual(match["gtVertex"], self.target_vertices[0].tolist())
        self.assertNotEqual(match["gt"], match["gtVertex"])
        self.assertEqual(match["target"], self.target_vertices[1].tolist())

    def test_payload_preserves_geometry_faces_indices_and_metrics(self):
        source_path = self.write_triangle_obj("source.obj", self.source_vertices)
        target_path = self.write_triangle_obj("target.obj", self.target_vertices)
        source = viewer.load_geometry(source_path)
        target = viewer.load_geometry(target_path)
        csv_path = self.write_benchmark(self.benchmark_row())
        matches = viewer.load_matches(
            csv_path,
            "benchmark",
            source["vertices_array"],
            target["vertices_array"],
        )
        payload = viewer.build_payload(
            source,
            target,
            "source specimen",
            "target specimen",
            matches,
            csv_path,
            "benchmark",
            "Synthetic correspondences",
        )

        self.assertEqual(payload["source"]["vertices"], self.source_vertices.tolist())
        self.assertEqual(payload["source"]["faces"], [[0, 1, 2]])
        self.assertIsNone(payload["source"]["colors"])
        self.assertEqual(payload["target"]["vertices"], self.target_vertices.tolist())
        self.assertEqual(payload["target"]["faces"], [[0, 1, 2]])
        self.assertEqual(payload["matches"][0]["sourceIndex"], 1)
        self.assertEqual(payload["matches"][0]["targetIndex"], 1)
        self.assertEqual(payload["matches"][0]["gtIndex"], 0)
        self.assertEqual(payload["matches"][0]["metrics"]["gt_feature_rank"], 17)
        expected_scale = max(
            np.linalg.norm(np.ptp(self.source_vertices, axis=0)),
            np.linalg.norm(np.ptp(self.target_vertices, axis=0)),
        )
        self.assertAlmostEqual(payload["scale"], expected_scale)

    def test_html_embedding_escapes_markup_without_recursive_replacement(self):
        html_path = self.root / "template.html"
        js_path = self.root / "template.js"
        html_path.write_text(
            "<title>__TITLE__</title><script type='application/json'>__DATA__</script>"
            "<script>__SCRIPT__</script>",
            encoding="utf-8",
        )
        js_path.write_text("window.viewerLoaded = true;", encoding="utf-8")
        payload = {
            "title": "</script><script>alert(1)</script>__SCRIPT__",
            "path": "<unsafe>",
        }

        rendered = viewer.render_html(
            payload,
            "<unsafe & title>",
            html_template_path=html_path,
            js_template_path=js_path,
        )

        self.assertIn("&lt;unsafe &amp; title&gt;", rendered)
        self.assertIn(r"\u003c/script>\u003cscript>alert(1)\u003c/script>__SCRIPT__", rendered)
        self.assertIn(r'"path":"\u003cunsafe>"', rendered)
        self.assertNotIn("</script><script>alert(1)</script>", rendered)
        self.assertEqual(rendered.count("window.viewerLoaded = true;"), 1)


if __name__ == "__main__":
    unittest.main()
