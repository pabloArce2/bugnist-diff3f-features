# Command Guide

This is the practical command reference for the BugNIST Diff3F workflow.

Run commands from the repository root:

```powershell
cd C:\Users\pablo\Documents\COPENAGUE\SECOND_SEMESTER\SPETIAL\Diffusion-3D-Features
conda activate diff3f
```

## 1. Preview Raw CT Volumes

Use this before choosing ROI and threshold values:

```powershell
python scripts\preview_tif_volume.py `
  --tif bugNIST\raw\bcrick_10_001.tif `
  --outdir previews\bugnist_raw_ct\bcrick_10_001 `
  --threshold 29
```

The useful outputs are orthogonal slices, maximum intensity projections, histogram, threshold overlays, and summary text.

Otsu automatic-threshold preview:

```powershell
python scripts\preview_tif_volume.py `
  --tif bugNIST\raw\bcrick_10_001.tif `
  --outdir previews\bugnist_raw_ct_otsu `
  --roi-start 185 30 29 `
  --roi-size 180 220 200 `
  --threshold-method otsu
```

## 2. Convert CT To Mesh

Clean no-downsample `bcrick` example:

```powershell
python scripts\bugnist_tif_to_mesh.py `
  --tif bugNIST\raw\bcrick_10_001.tif `
  --out meshes\bugnist_individual\bcrick_10_001\preprocessed\bcrick_10_001_thr29_roi_keeplargest_ds1.obj `
  --roi-start 185 30 29 `
  --roi-size 180 220 200 `
  --threshold 29 `
  --keep-largest `
  --downsample 1
```

Otsu mesh version using the same ROI:

```powershell
python scripts\bugnist_tif_to_mesh.py `
  --tif bugNIST\raw\bcrick_10_001.tif `
  --out meshes\bugnist_individual\bcrick_10_001\otsu\bcrick_10_001_otsu_roi_keeplargest_ds1.obj `
  --roi-start 185 30 29 `
  --roi-size 180 220 200 `
  --threshold-method otsu `
  --keep-largest `
  --downsample 1
```

Known clean preprocessing values:

```text
bcrick_10_001   ROI start 185 30 29   ROI size 180 220 200   threshold 29
sfaar_10_001    ROI start 171 37 29   ROI size 313 192 164   threshold 11
soldat_10_002   ROI start 72 60 73    ROI size 358 171 123   threshold 28
guld_1_002      ROI start 92 58 49    ROI size 333 143 165   threshold 11
```

## 3. Convert CT To Point Cloud

```powershell
python scripts\bugnist_tif_to_pointcloud.py `
  --tif bugNIST\raw\bcrick_10_001.tif `
  --out pointclouds\bugnist_individual\bcrick_10_001\preprocessed\bcrick_10_001_thr29_roi_keeplargest_ds1_20k.ply `
  --npy pointclouds\bugnist_individual\bcrick_10_001\preprocessed\bcrick_10_001_thr29_roi_keeplargest_ds1_20k.npy `
  --roi-start 185 30 29 `
  --roi-size 180 220 200 `
  --threshold 29 `
  --keep-largest `
  --downsample 1 `
  --num-points 20000 `
  --method mesh-surface `
  --center
```

Otsu point-cloud version using the same ROI:

```powershell
python scripts\bugnist_tif_to_pointcloud.py `
  --tif bugNIST\raw\bcrick_10_001.tif `
  --out pointclouds\bugnist_individual\bcrick_10_001\otsu\bcrick_10_001_otsu_roi_keeplargest_ds1_20k.ply `
  --npy pointclouds\bugnist_individual\bcrick_10_001\otsu\bcrick_10_001_otsu_roi_keeplargest_ds1_20k.npy `
  --roi-start 185 30 29 `
  --roi-size 180 220 200 `
  --threshold-method otsu `
  --keep-largest `
  --downsample 1 `
  --num-points 20000 `
  --method mesh-surface `
  --center
```

## 4. Preview Geometry

Mesh preview:

```powershell
python scripts\preview_mesh.py `
  --mesh meshes\bugnist_individual\bcrick_10_001\preprocessed\bcrick_10_001_thr29_roi_keeplargest_ds1.obj `
  --outdir previews\bugnist_meshes
```

Point-cloud preview:

```powershell
python scripts\preview_pointcloud.py `
  --pointcloud pointclouds\bugnist_individual\bcrick_10_001\preprocessed\bcrick_10_001_thr29_roi_keeplargest_ds1_20k.ply `
  --outdir previews\bugnist_pointclouds
```

Quick point-render preview for either kind:

```powershell
python scripts\preview_geometry_points.py `
  --kind mesh `
  --input meshes\bugnist_individual\bcrick_10_001\preprocessed\bcrick_10_001_thr29_roi_keeplargest_ds1.obj `
  --out previews\bugnist_meshes\bcrick_quick.png
```

## 5. Compute Mesh Diff3F Features

Local smoke test:

```powershell
python scripts\compute_mesh_features.py `
  --mesh meshes\bugnist_individual\sfaar_10_001\preprocessed\sfaar_10_001_thr11_roi_keeplargest_ds1.obj `
  --prompt "insect" `
  --outdir output\bugnist_clean_features `
  --num-views 4 `
  --height 256 `
  --width 256 `
  --tolerance 0.004
```

Higher-quality local/HPC run:

```powershell
python scripts\compute_mesh_features.py `
  --mesh meshes\bugnist_individual\sfaar_10_001\preprocessed\sfaar_10_001_thr11_roi_keeplargest_ds1.obj `
  --prompt "insect" `
  --outdir output\bugnist_clean_features_hq_16v_512 `
  --num-views 16 `
  --height 512 `
  --width 512 `
  --tolerance 0.008
```

`--num-views` must be a perfect square: `4`, `9`, `16`, `25`, `100`.

## 6. Compute Point-Cloud Diff3F Features

```powershell
python scripts\compute_pointcloud_features.py `
  --pointcloud pointclouds\bugnist_individual\bcrick_10_001\preprocessed\bcrick_10_001_thr29_roi_keeplargest_ds1_20k.ply `
  --prompt "insect" `
  --outdir output\bugnist_pointcloud_features `
  --num-views 4 `
  --height 256 `
  --width 256 `
  --point-radius 0.012
```

## 7. Visualize One Feature File

Mesh descriptor as colored PLY and PNG:

```powershell
python scripts\visualize_mesh_features.py `
  --mesh meshes\bugnist_individual\sfaar_10_001\preprocessed\sfaar_10_001_thr11_roi_keeplargest_ds1.obj `
  --features output\bugnist_clean_features_hq_16v_512\sfaar_10_001_thr11_roi_keeplargest_ds1_diff3f.pt `
  --out visualizations\bugnist_clean_features_hq_16v_512\sfaar_10_001_16v512_features.ply `
  --preview visualizations\bugnist_clean_features_hq_16v_512\sfaar_10_001_16v512_features.png
```

Point-cloud descriptor as colored PLY and PNG:

```powershell
python scripts\visualize_pointcloud_features.py `
  --pointcloud pointclouds\bugnist_individual\bcrick_10_001\preprocessed\bcrick_10_001_thr29_roi_keeplargest_ds1_20k.ply `
  --features output\bugnist_pointcloud_features\bcrick_10_001_thr29_roi_keeplargest_ds1_20k_diff3f.pt `
  --out visualizations\bugnist_pointcloud_features\bcrick_20k_features.ply `
  --preview visualizations\bugnist_pointcloud_features\bcrick_20k_features.png
```

## 8. Visualize Multiple Meshes With Shared PCA

Use this when colors must be comparable between insects:

```powershell
python scripts\visualize_feature_comparison.py `
  --item bcrick meshes\bugnist_individual\bcrick_10_001\preprocessed\bcrick_10_001_thr29_roi_keeplargest_ds1.obj output\bugnist_clean_features_hq_16v_512\bcrick_10_001_thr29_roi_keeplargest_ds1_diff3f.pt `
  --item sfaar meshes\bugnist_individual\sfaar_10_001\preprocessed\sfaar_10_001_thr11_roi_keeplargest_ds1.obj output\bugnist_clean_features_hq_16v_512\sfaar_10_001_thr11_roi_keeplargest_ds1_diff3f.pt `
  --outdir visualizations\bugnist_clean_features_hq_16v_512\shared_pca_bcrick_sfaar
```

## 9. Visualize Multiple Point Clouds With Shared PCA

```powershell
python scripts\visualize_pointcloud_feature_comparison.py `
  --item bcrick pointclouds\bugnist_individual\bcrick_10_001\preprocessed\bcrick_10_001_thr29_roi_keeplargest_ds1_20k.ply output\bugnist_pointcloud_features\bcrick_10_001_thr29_roi_keeplargest_ds1_20k_diff3f.pt `
  --item sfaar pointclouds\bugnist_individual\sfaar_10_001\preprocessed\sfaar_10_001_thr11_roi_keeplargest_ds1_20k.ply output\bugnist_pointcloud_features\sfaar_10_001_thr11_roi_keeplargest_ds1_20k_diff3f.pt `
  --outdir visualizations\bugnist_pointcloud_features\shared_pca_bcrick_sfaar
```

## 10. Inspect Feature Quality

Basic health check:

```powershell
python scripts\inspect_feature_quality.py `
  --geometry meshes\bugnist_individual\sfaar_10_001\preprocessed\sfaar_10_001_thr11_roi_keeplargest_ds1.obj `
  --features output\bugnist_clean_features_hq_16v_512\sfaar_10_001_thr11_roi_keeplargest_ds1_diff3f.pt
```

Compare two feature runs on the same geometry:

```powershell
python scripts\inspect_feature_quality.py `
  --geometry meshes\bugnist_individual\sfaar_10_001\preprocessed\sfaar_10_001_thr11_roi_keeplargest_ds1.obj `
  --features output\bugnist_clean_features_hq_16v_512\sfaar_10_001_thr11_roi_keeplargest_ds1_diff3f.pt `
  --compare-features output\bugnist_clean_features\sfaar_10_001_thr11_roi_keeplargest_ds1_diff3f.pt
```

## 11. Debug One 2D Diffusion View

Render/control-only:

```powershell
python scripts\debug_2d_diffusion_view.py `
  --kind pointcloud `
  --input pointclouds\bugnist_individual\bcrick_10_001\preprocessed\bcrick_10_001_thr29_roi_keeplargest_ds1_20k.ply `
  --prompt insect `
  --outdir debug\pointcloud_bcrick_view0_controls `
  --num-views 4 `
  --view-index 0 `
  --height 128 `
  --width 128 `
  --skip-ai
```

Full AI trace:

```powershell
python scripts\debug_2d_diffusion_view.py `
  --kind pointcloud `
  --input pointclouds\bugnist_individual\bcrick_10_001\preprocessed\bcrick_10_001_thr29_roi_keeplargest_ds1_20k.ply `
  --prompt insect `
  --outdir debug\pointcloud_bcrick_view0_ai `
  --num-views 4 `
  --view-index 0 `
  --height 128 `
  --width 128 `
  --num-inference-steps 10 `
  --denoise-interval 2
```

## 12. Compute Feature Correspondences

```powershell
python scripts\compute_feature_correspondences.py `
  --source-name bcrick `
  --source-mesh meshes\bugnist_individual\bcrick_10_001\preprocessed\bcrick_10_001_thr29_roi_keeplargest_ds1.obj `
  --source-features output\bugnist_clean_features_hq_16v_512\bcrick_10_001_thr29_roi_keeplargest_ds1_diff3f.pt `
  --target-name sfaar `
  --target-mesh meshes\bugnist_individual\sfaar_10_001\preprocessed\sfaar_10_001_thr11_roi_keeplargest_ds1.obj `
  --target-features output\bugnist_clean_features_hq_16v_512\sfaar_10_001_thr11_roi_keeplargest_ds1_diff3f.pt `
  --outdir visualizations\bugnist_correspondences_hq_16v_512 `
  --num-source-points 80 `
  --mutual-check
```

## 13. Evaluate Correspondence Diagnostics

```powershell
python scripts\evaluate_correspondence_metrics.py `
  --source-name bcrick `
  --source-mesh meshes\bugnist_individual\bcrick_10_001\preprocessed\bcrick_10_001_thr29_roi_keeplargest_ds1.obj `
  --source-features output\bugnist_clean_features_hq_16v_512\bcrick_10_001_thr29_roi_keeplargest_ds1_diff3f.pt `
  --target-name sfaar `
  --target-mesh meshes\bugnist_individual\sfaar_10_001\preprocessed\sfaar_10_001_thr11_roi_keeplargest_ds1.obj `
  --target-features output\bugnist_clean_features_hq_16v_512\sfaar_10_001_thr11_roi_keeplargest_ds1_diff3f.pt `
  --outdir visualizations\bugnist_correspondence_metrics_hq_16v_512 `
  --num-samples 300
```

## 14. Create Blender Scenes

Feature comparison scene:

```powershell
& "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" `
  --background `
  --python scripts\make_blender_feature_comparison.py `
  -- `
  --item bcrick visualizations\bugnist_clean_features_hq_16v_512\shared_pca_bcrick_sfaar\bcrick_shared_pca_features.ply `
  --item sfaar visualizations\bugnist_clean_features_hq_16v_512\shared_pca_bcrick_sfaar\sfaar_shared_pca_features.ply `
  --output visualizations\bugnist_clean_features_hq_16v_512\shared_pca_bcrick_sfaar\bcrick_sfaar_shared_pca.blend
```

Correspondence scene:

```powershell
& "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" `
  --background `
  --python scripts\make_blender_correspondence_scene.py `
  -- `
  --source-mesh meshes\bugnist_individual\bcrick_10_001\preprocessed\bcrick_10_001_thr29_roi_keeplargest_ds1.obj `
  --target-mesh meshes\bugnist_individual\sfaar_10_001\preprocessed\sfaar_10_001_thr11_roi_keeplargest_ds1.obj `
  --matches visualizations\bugnist_correspondences_hq_16v_512\bcrick_to_sfaar_matches.csv `
  --output visualizations\bugnist_correspondences_hq_16v_512\bcrick_to_sfaar_correspondence.blend
```

## 15. DTU / HPC

Submit example job:

```bash
mkdir -p logs
bsub < hpc/dtu_lsf_overnight_smoke.sh
bjobs
```

Do not run heavy Diff3F extraction on the login node. Submit through LSF.
