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
```

## Geometry Preview

```text
preview_mesh.py             quick mesh PNG previews
preview_pointcloud.py       quick point-cloud PNG previews
preview_geometry_points.py  point-render preview for either meshes or point clouds
```

## Diff3F Descriptor Extraction

```text
compute_mesh_features.py       compute Diff3F features for mesh vertices
compute_pointcloud_features.py compute Diff3F features for point-cloud points
debug_2d_diffusion_view.py     save one 2D render, controls, AI trace, or feature debug images
```

## Feature Visualization

```text
visualize_mesh_features.py                  color one mesh using PCA of its features
visualize_pointcloud_features.py            color one point cloud using PCA of its features
visualize_feature_comparison.py             shared-PCA colors for several mesh feature files
visualize_pointcloud_feature_comparison.py  shared-PCA colors for several point-cloud feature files
make_blender_feature_comparison.py          optional Blender scene for colored PLY comparison
```

## Correspondence And Diagnostics

```text
compute_feature_correspondences.py  nearest-neighbor feature matches between shapes
evaluate_correspondence_metrics.py  unsupervised correspondence diagnostics
inspect_feature_quality.py          basic feature tensor health and smoothness checks
make_blender_correspondence_scene.py optional Blender scene for sampled correspondences
```
