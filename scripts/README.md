# Scripts

Run from the repository root, e.g. `python scripts/preview_tif_volume.py --help`. The order
below follows the workflow in [docs/pipeline.md](../docs/pipeline.md). Shared code is in
[`bugnist_tools/`](../bugnist_tools).

## CT scan to geometry

| Script | Does |
| --- | --- |
| `preview_tif_volume.py` | slices, projections, threshold overlays and histogram of a TIFF volume |
| `bugnist_tif_to_mesh.py` | crop, threshold and clean a scan, then marching cubes to OBJ/PLY |
| `bugnist_tif_to_pointcloud.py` | the same segmentation, then sample a point cloud (PLY/XYZ, optional NPY) |
| `smooth_simplify_mesh.py` | Taubin/Laplacian smoothing and decimation to a target face count |
| `transform_geometry.py` | rotate, scale or translate a mesh or point cloud |
| `mesh_to_pointcloud.py` | sample points on existing meshes |
| `preview_geometry.py` | PNG preview of meshes or point clouds (matplotlib, or fast point render) |

## Descriptors (GPU)

| Script | Does |
| --- | --- |
| `compute_mesh_features.py` | Diff3F descriptors for one or more meshes, `.pt` per mesh |
| `compute_pointcloud_features.py` | the same for point clouds |
| `debug_2d_diffusion_view.py` | one camera view step by step: render, ControlNet inputs, denoising, feature PCA |
| `inspect_feature_quality.py` | NaNs, norms and smoothness of a `.pt`; compare two `.pt` files |

## Visualisation

| Script | Does |
| --- | --- |
| `visualize_mesh_features.py` | colour a mesh by the PCA of its descriptor |
| `visualize_pointcloud_features.py` | colour a point cloud by the PCA of its descriptor |
| `visualize_feature_comparison.py` | shared PCA colours for several shapes, optional shared k-means |
| `plot_pca_kmeans_scatter.py` | the shared k-means clusters in PCA space, with centroids |
| `plot_cluster_grid.py` | one panel per shared k-means cluster |
| `make_contact_sheet.py` | tile PNGs into a labelled grid |
| `pointcloud_to_html_viewer.py` | self-contained HTML viewer for a coloured point cloud |

## Matching and evaluation

| Script | Does |
| --- | --- |
| `compute_feature_correspondences.py` | nearest-neighbour matches from sampled source vertices |
| `evaluate_correspondence_metrics.py` | cycle consistency, target reuse and other label-free diagnostics |
| `evaluate_landmark_benchmark.py` | errors and PCK against hand-placed landmarks |
| `correspondence_to_html_viewer.py` | self-contained HTML viewer for matches or a benchmark (uses `templates/`) |

## Blender

Run these with Blender (`blender --background --python scripts/<name>.py -- <options>`),
not with the conda Python. The usage is at the top of each file.

| Script | Does |
| --- | --- |
| `make_blender_landmark_scene.py` | a scene with movable landmark markers, for labelling by hand |
| `export_blender_landmarks.py` | write the markers of a labelled scene to a landmark CSV |
| `make_blender_landmark_benchmark_scene.py` | source, prediction and manual target for every landmark |
| `make_blender_correspondence_scene.py` | two shapes side by side with a line per match |
| `make_blender_feature_comparison.py` | coloured PLYs side by side |

`pointcloud_to_blender_splats.py` runs in the conda environment and turns a coloured point
cloud into a small mesh that Blender displays with its colours.
