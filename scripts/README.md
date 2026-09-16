# Scripts Map

This folder contains the practical BugNIST/Diff3F workflow scripts.

For full commands, use:

```text
documentation/guides/COMMAND_GUIDE.md
```

## CT Preview And Geometry Creation

```text
preview_tif_volume.py        inspect TIFF CT slices, MIPs, histograms, threshold overlays
bugnist_tif_to_mesh.py       convert segmented BugNIST CT volume to OBJ/PLY mesh
bugnist_tif_to_pointcloud.py convert segmented BugNIST CT volume to PLY/NPY point cloud
mesh_to_pointcloud.py        sample point clouds from existing mesh surfaces
transform_geometry.py        rotate, scale, or translate meshes and point clouds
smooth_simplify_mesh.py      smooth CT stair-steps and optionally reduce mesh face count
```

## Geometry Preview

```text
preview_mesh.py             quick mesh PNG previews
preview_pointcloud.py       quick point-cloud PNG previews
preview_geometry_points.py  point-render preview for either meshes or point clouds
pointcloud_to_blender_splats.py convert colored points to tiny colored mesh splats for Blender
pointcloud_to_html_viewer.py    standalone colored point-cloud viewer with original row/XYZ click inspection
```

## Diff3F Descriptor Extraction

```text
compute_mesh_features.py       compute mesh-vertex Diff3F features; optional exact run-image capture
compute_pointcloud_features.py compute point-cloud Diff3F features; optional exact run-image capture
debug_2d_diffusion_view.py     save one 2D render, controls, AI trace, or feature debug images
```

## Feature Visualization

```text
visualize_mesh_features.py                  color one mesh using PCA of its features
visualize_pointcloud_features.py            color one point cloud using PCA of its features
visualize_feature_comparison.py             shared-PCA colors for several mesh feature files; optional --kmeans K for a shared discrete cluster map (default clusters full features, not the PCA projection)
visualize_pointcloud_feature_comparison.py  shared-PCA colors for several point-cloud feature files; optional --kmeans K for a shared discrete cluster map (default clusters full features, not the PCA projection)
plot_pca_kmeans_scatter.py                  scatter-plot the shared-PCA feature space (PC1/PC2/PC3 pairs) colored by shared K-means cluster, with centroids
plot_cluster_grid.py                        per-item grid isolating each shared K-means cluster one at a time against a greyed-out mesh
make_blender_feature_comparison.py          optional Blender scene for colored PLY comparison
make_contact_sheet.py                       tile existing preview PNGs into one labeled row/column grid image
```

## Correspondence And Diagnostics

```text
compute_feature_correspondences.py  nearest-neighbor feature matches between shapes
evaluate_correspondence_metrics.py  unsupervised correspondence diagnostics
inspect_feature_quality.py          basic feature tensor health and smoothness checks
make_blender_correspondence_scene.py optional Blender scene for sampled correspondences
make_blender_landmark_scene.py       create Blender scenes with movable manual landmark markers
export_blender_landmarks.py          export Blender landmark markers to label,x,y,z CSV
evaluate_landmark_benchmark.py       evaluate predicted matches against manual landmarks
make_blender_landmark_benchmark_scene.py visualize manual-vs-predicted landmark errors
correspondence_to_html_viewer.py     standalone two-mesh correspondence viewer with surface/vertex picking
```
