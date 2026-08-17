# Diff3F 2D View Debug

This folder captures one 2D view from the Diff3F pipeline.

## Files

- `01_input_render.png`: the plain 2D render given to image-to-image Stable Diffusion.
- `02_depth_control.png`: the depth ControlNet conditioning image.
- `03_normal_control.png`: the normal ControlNet conditioning image, if enabled.
- `04_visible_mask.png`: white pixels are geometry pixels; black pixels are background.
- `05_denoise_step_*.png`: decoded latent snapshots during Stable Diffusion denoising.
- `06_final_generated.png`: the final image produced by the 2D AI.
- `07_ai_change_map.png`: amplified pixel difference between the input render and final generated image.
- `08_unet_feature_pca.png`: PCA visualization of the 1280-D diffusion UNet feature map.
- `09_dino_feature_pca.png`: PCA visualization of the 768-D DINOv2 feature map.
- `10_combined_diff3f_feature_pca.png`: PCA visualization of the full 2048-D pixel descriptor.
- `contact_sheet.png`: a compact visual summary.

## Run

```text
{
  "kind": "pointcloud",
  "input": "pointclouds\\bugnist_individual\\bcrick_10_001\\preprocessed\\bcrick_10_001_thr29_roi_keeplargest_ds1_20k.ply",
  "prompt": "insect",
  "outdir": "debug\\pointcloud_bcrick_view0_controls",
  "device": null,
  "num_views": 4,
  "view_index": 0,
  "height": 128,
  "width": 128,
  "tosca": false,
  "no_normal_map": false,
  "point_radius": 0.012,
  "points_per_pixel": 1,
  "num_inference_steps": 30,
  "denoise_interval": 5,
  "guidance_scale": 7.0,
  "eta": 1.0,
  "fit_sample": 12000,
  "seed": 42,
  "skip_ai": true
}
```

Saved denoising steps: []

The PCA feature images are only visual shadows of high-dimensional features. They help us inspect structure, but matching still happens in the full descriptor space.
