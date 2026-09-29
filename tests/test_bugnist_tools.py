from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bugnist_tools.ct import crop_roi, crop_to_mask, resolve_threshold, segment  # noqa: E402
from bugnist_tools.features import (  # noqa: E402
    SharedPCA,
    fit_shared_kmeans,
    pca_basis,
    predict_clusters,
    shared_sample,
)
from bugnist_tools.geometry import check_rows, load_mesh, load_vertices  # noqa: E402
from bugnist_tools.matching import farthest_point_indices, nearest_neighbors, nearest_two  # noqa: E402


def two_blobs():
    volume = np.zeros((40, 40, 40), dtype=np.uint8)
    volume[5:25, 5:25, 5:25] = 200  # large cube
    volume[30:34, 30:34, 30:34] = 200  # small cube
    return volume


class ThresholdTests(unittest.TestCase):
    def test_auto_prefers_manual_then_percentile_then_otsu(self):
        volume = two_blobs()
        self.assertEqual(resolve_threshold(volume, threshold=50), (50.0, "manual"))
        self.assertEqual(resolve_threshold(volume, percentile=90)[1], "p90")
        self.assertEqual(resolve_threshold(volume)[1], "otsu")

    def test_manual_and_percentile_together_is_an_error(self):
        with self.assertRaises(ValueError):
            resolve_threshold(two_blobs(), threshold=50, percentile=90, method="manual")


class SegmentationTests(unittest.TestCase):
    def test_keep_largest_drops_the_small_component(self):
        mask, _, _ = segment(two_blobs(), threshold=100, min_size=0, keep_largest=True)
        self.assertEqual(int(mask.sum()), 20**3)
        self.assertFalse(mask[31, 31, 31])

    def test_min_size_removes_specks(self):
        mask, _, _ = segment(two_blobs(), threshold=100, min_size=100)
        self.assertEqual(int(mask.sum()), 20**3)

    def test_fill_holes_closes_a_hollow_shell(self):
        volume = np.zeros((30, 30, 30), dtype=np.uint8)
        volume[5:25, 5:25, 5:25] = 200
        volume[10:20, 10:20, 10:20] = 0
        mask, _, _ = segment(volume, threshold=100, min_size=0, fill_holes=True)
        self.assertTrue(mask[15, 15, 15])

    def test_crop_offsets_point_back_into_the_scan(self):
        volume = two_blobs()
        cropped, start = crop_roi(volume, (2, 3, 4), (10, 10, 10))
        self.assertEqual(cropped.shape, (10, 10, 10))
        np.testing.assert_array_equal(start, [2, 3, 4])
        _, start = crop_to_mask(volume, volume > 100, (1, 1, 1))
        np.testing.assert_array_equal(start, [4, 4, 4])


class GeometryTests(unittest.TestCase):
    def test_obj_vertices_keep_file_order_with_normals(self):
        obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nvn 0 0 1\nf 1//1 2//1 3//1\nf 1//1 3//1 4//1\n"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tetra.obj"
            path.write_text(obj, encoding="utf-8")
            vertices = load_vertices(path)
            _, faces = load_mesh(path)
        np.testing.assert_array_equal(vertices, [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        np.testing.assert_array_equal(faces, [[0, 1, 2], [0, 2, 3]])

    def test_row_mismatch_is_reported(self):
        with self.assertRaises(ValueError):
            check_rows("shape", 10, 11)


class FeatureTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(0)
        self.sets = [torch.randn(300, 16, generator=generator), torch.randn(250, 16, generator=generator)]

    def test_pca_is_deterministic_orthonormal_and_sign_fixed(self):
        rows = shared_sample(self.sets, 200, seed=1)
        _, first, _ = pca_basis(rows)
        _, second, _ = pca_basis(rows.clone())
        self.assertTrue(torch.equal(first, second))
        torch.testing.assert_close(first.T @ first, torch.eye(3), atol=1e-5, rtol=0)
        largest = first.abs().argmax(dim=0)
        self.assertTrue(bool((first[largest, torch.arange(3)] > 0).all()))

    def test_shared_colors_and_clusters(self):
        pca = SharedPCA(self.sets, sample_per_item=200, seed=3)
        colors = pca.colors(self.sets[0])
        self.assertEqual(colors.shape, (300, 3))
        self.assertEqual(colors.dtype, np.uint8)
        kmeans = fit_shared_kmeans(self.sets, pca, k=4, sample_per_item=200, seed=3)
        labels = predict_clusters(self.sets[1], pca, kmeans)
        self.assertEqual(labels.shape, (250,))
        self.assertTrue(set(np.unique(labels)) <= {0, 1, 2, 3})


class MatchingTests(unittest.TestCase):
    def test_nearest_neighbours_find_the_same_row(self):
        generator = torch.Generator().manual_seed(1)
        target = torch.randn(50, 8, generator=generator)
        query = target[[3, 17, 42]] * 2.0  # cosine similarity ignores the scale
        indices, scores = nearest_neighbors(query, target, "cpu", target_chunk_size=7)
        np.testing.assert_array_equal(indices, [3, 17, 42])
        np.testing.assert_allclose(scores, 1.0, atol=1e-5)
        indices2, _, second = nearest_two(query, target, "cpu", target_chunk_size=7)
        np.testing.assert_array_equal(indices2, indices)
        self.assertTrue(bool((second < 1.0).all()))

    def test_farthest_point_sampling_starts_far_from_the_centre(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [10, 0, 0], [0, 2, 0]], dtype=np.float32)
        indices = farthest_point_indices(points, 3, seed=0)
        self.assertEqual(indices[0], 2)
        self.assertEqual(len(set(indices.tolist())), 3)


if __name__ == "__main__":
    unittest.main()
