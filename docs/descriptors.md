# Diff3F descriptors

## What is in a `.pt` file

`compute_mesh_features.py` and `compute_pointcloud_features.py` save one tensor per input:

```python
features = torch.load("output/bugnist_crickets_features_16v_512/BrownCricket_rotated_diff3f.pt")
features.shape   # torch.Size([24683, 2048]), dtype float16
```

Row `i` is the descriptor of vertex `i` (or point `i`) of the exact file that was
processed, in file order. The file stores no geometry, so always keep the pair together:
`BrownCricket_rotated.obj` with `BrownCricket_rotated_diff3f.pt`. All scripts that take both
check that the row counts agree.

The 2048 values are the concatenation of two image features, each L2-normalised and
weighted by 0.5:

| Columns | Source | Resolution in the image |
| --- | --- | --- |
| 0-1279 | Stable Diffusion UNet, second decoder block, accumulated during denoising | 32 x 32 |
| 1280-2047 | DINOv2 ViT-B/14, last block, on the generated image | 37 x 37 |

Compare descriptors with cosine similarity. The rows are not normalised again after
averaging over views (see below), so a row's length says something about how consistently
the views agreed on that point.

## How a descriptor is computed

For every camera view (`diff3f.py`, `get_features_per_vertex`):

1. **Render** the untextured shape with PyTorch3D: a grey image, a depth map and a normal
   map. The camera looks at the centre of the bounding box from 0.65 times its diagonal.
2. **Generate an image.** Stable Diffusion 1.5 in image-to-image mode starts from the
   render, with the depth and normal maps as ControlNet conditions and the prompt
   (plus `,best quality,highly detailed,photorealistic,photo`; negative prompt
   `lowres,low quality,monochrome,watermark`). DDIM, 30 steps, guidance 7.
3. **Read features.** While denoising, the UNet decoder activations are summed over the
   last three quarters of the steps, with later (cleaner) steps weighted more. DINOv2 then
   looks at the finished image.
4. **Fuse** the two feature maps per pixel as described above.
5. **Project back.** Each foreground pixel is unprojected with its depth, and all vertices
   within `--tolerance` (a fraction of the mesh diameter) of that 3D point receive the
   pixel's descriptor.

After all views, each vertex holds the average of what it received. Vertices that no camera
saw copy the descriptor of the nearest vertex that was seen.

For point clouds (`get_features_per_point_cloud`, added in this fork) the rasteriser
already reports which point is visible in each pixel, so the pixel descriptor is added
directly to that point; there is no tolerance radius. `--point-radius` sets the rendered
point size in normalised device coordinates. Too small and the render has holes, too large
and neighbouring points blur together; 0.012 worked for 20k-point insects.

## Choosing the options

**Prompt.** The only semantic input. A short prompt such as `insect` sometimes makes the
model paint wings or body shapes that are not in the geometry; a descriptive one follows
the silhouette better. The cricket prompt used in the project was
*cricket insect, full body, long antennae, large hind legs, segmented insect body, detailed
exoskeleton, macro photograph*. Avoid scene words ("on a leaf") that invite the model to add
things. Keep the prompt identical for specimens you want to compare.

**Views** (`--num-views`, `--view-sampling`, `bugnist_tools/camera_sampling.py`):

| Layout | Cameras | Notes |
| --- | --- | --- |
| `grid` | sqrt(N) azimuths x sqrt(N) elevations | original Diff3F; N must be a square. The elevation also runs through 360 degrees, so cameras repeat: 16 views are really 6 distinct ones, 100 are 50 |
| `fibonacci` | evenly spread on a sphere | any N |
| `insect` | rings at 0, +-25 and +-50 degrees elevation, 40/25/25/5/5 % of the views | any N; most cameras look at the side of the body, where legs and antennae are visible |

16 views at 512 x 512 was the project default. Because the layouts are fixed in world
coordinates, rotate specimens to a common pose first ([pipeline.md](pipeline.md)).

**Resolution.** 512 x 512 is Stable Diffusion 1.5's native size. 256 x 256 is much faster
and fine for checking that everything runs, but thin legs and antennae then cover very few
pixels.

**`--tolerance`** (meshes only). 0.004-0.01 of the mesh diameter. Larger values let a pixel
reach more vertices, which fills small gaps but blurs detail.

**`--no-normal-map`** uses only the depth ControlNet.

## Saving the images of a run

`--debug-outdir DIR --debug-views all` (or `0 4 8`, `0-3`) makes the descriptor run save, for
each selected view, the images it actually used:

```text
DIR/mesh_<timestamp>/00_<mesh name>/
    view_000/01_input_render.png      render given to Stable Diffusion
             02_depth_control.png     depth ControlNet input
             03_normal_control.png    normal ControlNet input
             04_visible_mask.png      pixels that belong to the object
             06_final_generated.png   generated image, the one DINOv2 read
             07_ai_change_map.png     |generated - render|, amplified
    generated_contact_sheet.png
    manifest.json                    prompt, settings, views, output .pt, status
```

Capturing always recomputes the descriptor (it ignores `--skip-existing`), so the images and
the `.pt` come from the same execution.

To look inside a single view in more detail, `debug_2d_diffusion_view.py` runs one camera
through the pipeline and also saves decoded latents during denoising
(`05_denoise_step_*.png`) and PCA images of the UNet, DINOv2 and combined features
(`08`-`10`). `--render-only` stops after the render (with `--all-views`: every camera), and
`--skip-diffusion` after the ControlNet inputs.

## Things worth knowing

These came out of inspecting the intermediate outputs during the project and are left as
they are, so results stay comparable with the original method:

- With image-to-image strength 0.8, only 24 of the 30 requested steps run, so the UNet
  feature is accumulated over the last 18.
- The random seed is reset to the same value before every view, so all views start from
  the same noise. Results are repeatable, but the noise is not averaged out across views.
  (Descriptors from two runs still differ very slightly because of GPU non-determinism.)
- With classifier-free guidance the UNet runs a negative-prompt and a positive-prompt
  branch; the recorded feature comes from the first, negative-prompt branch.
- On meshes, when several pixels of one view land on the same vertex, only one of them is
  counted (an indexed `+=` with repeated indices). The mesh descriptor is therefore close to
  an average over views, while the point-cloud version averages over all pixels.

## Changes to the upstream code

The method is the upstream implementation. The fork changes:

- `diff3f.py`: point-cloud aggregation (`get_features_per_point_cloud`); a `view_sampling`
  and a `view_observer` argument (for the image capture); missing vertices are filled on the
  CPU in chunks instead of with one full distance matrix on the GPU, which needed ~28 GB
  on large CT meshes.
- `render.py`, `render_point_cloud.py`: camera layouts from `bugnist_tools/camera_sampling.py`;
  the point renderer takes the point radius and image size, returns the visible point per
  pixel and shades a render and a normal map from depth.
- `pipeline_controlnet_img2img.py`: the UNet feature buffer takes its shape from the first
  feature map instead of a fixed 1280 x 32 x 32, so resolutions other than 512 work.
- `diffusion.py`: depth-only ControlNet option, a fallback when `accelerate` is missing,
  a guard against flat normal maps, and the `DIFF3F_SD_MODEL` override.
- `dataloaders/point_cloud_dataset.py`: a plain PyTorch k-NN when `torch_cluster` is not
  installed.
