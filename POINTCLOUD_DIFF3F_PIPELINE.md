# Point Cloud Diff3F Pipeline

This is the point-cloud equivalent of `scripts/compute_mesh_features.py`.
It assumes point clouds were generated from the BugNIST CT volumes with the same segmentation choices documented in `..\PROJECT_CONTEXT.md`.

Run commands from the repository root:

```powershell
cd C:\Users\pablo\Documents\COPENAGUE\SECOND_SEMESTER\SPETIAL\Diffusion-3D-Features
conda activate diff3f
```

## Compute Descriptors

Start with the 20k point clouds for local testing:

```powershell
python scripts\compute_pointcloud_features.py `
  --pointcloud `
    pointclouds\bugnist_individual\bcrick_10_001\preprocessed\bcrick_10_001_thr29_roi_keeplargest_ds1_20k.ply `
    pointclouds\bugnist_individual\sfaar_10_001\preprocessed\sfaar_10_001_thr11_roi_keeplargest_ds1_20k.ply `
    pointclouds\bugnist_individual\soldat_10_002\preprocessed\soldat_10_002_thr28_roi_keeplargest_ds1_20k.ply `
  --prompt "insect" `
  --outdir output\bugnist_pointcloud_features `
  --num-views 4 `
  --height 256 `
  --width 256 `
  --point-radius 0.012
```

Output files are named `<pointcloud_stem>_diff3f.pt`.
Each `.pt` has shape `[num_points, 2048]`, where row `i` corresponds to point `i` in the exact input point cloud.

Use `--num-views 9`, `16`, or `25` for better descriptors when GPU time allows.
`--num-views` must be a perfect square.

## Visualize One Point Cloud

```powershell
python scripts\visualize_pointcloud_features.py `
  --pointcloud pointclouds\bugnist_individual\bcrick_10_001\preprocessed\bcrick_10_001_thr29_roi_keeplargest_ds1_20k.ply `
  --features output\bugnist_pointcloud_features\bcrick_10_001_thr29_roi_keeplargest_ds1_20k_diff3f.pt `
  --out visualizations\bugnist_pointcloud_features\bcrick_20k_features.ply `
  --preview visualizations\bugnist_pointcloud_features\bcrick_20k_features.png
```

## Shared PCA Comparison

Use this when you want comparable colors across multiple insects:

```powershell
python scripts\visualize_pointcloud_feature_comparison.py `
  --item bcrick pointclouds\bugnist_individual\bcrick_10_001\preprocessed\bcrick_10_001_thr29_roi_keeplargest_ds1_20k.ply output\bugnist_pointcloud_features\bcrick_10_001_thr29_roi_keeplargest_ds1_20k_diff3f.pt `
  --item sfaar pointclouds\bugnist_individual\sfaar_10_001\preprocessed\sfaar_10_001_thr11_roi_keeplargest_ds1_20k.ply output\bugnist_pointcloud_features\sfaar_10_001_thr11_roi_keeplargest_ds1_20k_diff3f.pt `
  --item soldat pointclouds\bugnist_individual\soldat_10_002\preprocessed\soldat_10_002_thr28_roi_keeplargest_ds1_20k.ply output\bugnist_pointcloud_features\soldat_10_002_thr28_roi_keeplargest_ds1_20k_diff3f.pt `
  --outdir visualizations\bugnist_pointcloud_features\shared_pca_compare
```

## Correspondence Diagnostics

The existing correspondence scripts now accept point clouds as the geometry inputs:

```powershell
python scripts\compute_feature_correspondences.py `
  --source-name bcrick `
  --source-mesh pointclouds\bugnist_individual\bcrick_10_001\preprocessed\bcrick_10_001_thr29_roi_keeplargest_ds1_20k.ply `
  --source-features output\bugnist_pointcloud_features\bcrick_10_001_thr29_roi_keeplargest_ds1_20k_diff3f.pt `
  --target-name sfaar `
  --target-mesh pointclouds\bugnist_individual\sfaar_10_001\preprocessed\sfaar_10_001_thr11_roi_keeplargest_ds1_20k.ply `
  --target-features output\bugnist_pointcloud_features\sfaar_10_001_thr11_roi_keeplargest_ds1_20k_diff3f.pt `
  --outdir visualizations\bugnist_pointcloud_correspondences `
  --num-source-points 80 `
  --mutual-check
```

```powershell
python scripts\evaluate_correspondence_metrics.py `
  --source-name bcrick `
  --source-mesh pointclouds\bugnist_individual\bcrick_10_001\preprocessed\bcrick_10_001_thr29_roi_keeplargest_ds1_20k.ply `
  --source-features output\bugnist_pointcloud_features\bcrick_10_001_thr29_roi_keeplargest_ds1_20k_diff3f.pt `
  --target-name sfaar `
  --target-mesh pointclouds\bugnist_individual\sfaar_10_001\preprocessed\sfaar_10_001_thr11_roi_keeplargest_ds1_20k.ply `
  --target-features output\bugnist_pointcloud_features\sfaar_10_001_thr11_roi_keeplargest_ds1_20k_diff3f.pt `
  --outdir visualizations\bugnist_pointcloud_correspondence_metrics `
  --num-samples 300
```
